"""Generalize assembly-return permission over local access geometry."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_phase147_generic_assembly_return as phase147,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _enable_relational_access,
    _new_arm,
    _observe,
    _predict,
)
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _development_stream,
)
from manifoldflow.experiments.dendritic_basin_phase102_phase_aligned_chain import (
    _advance_generated_state,
)


class GeometricPermission:
    """Local signed utility averages on overlapping receptor dendrites."""

    def __init__(
        self,
        dimensions: int,
        *,
        prior: float = 1.0,
        risk_weight: float = 4.0,
    ) -> None:
        self.dimensions = int(dimensions)
        self.prior = float(prior)
        self.risk_weight = float(risk_weight)
        self.utility_sum: dict[int, np.ndarray] = {}
        self.help_sum: dict[int, np.ndarray] = {}
        self.harm_sum: dict[int, np.ndarray] = {}
        self.evidence: dict[int, np.ndarray] = {}
        self.trials = 0
        self.positive_trials = 0
        self.negative_trials = 0
        self.positive_utility = 0.0
        self.negative_utility = 0.0

    def _features(self, query: np.ndarray) -> np.ndarray:
        values = np.abs(np.asarray(query, dtype=np.float64))
        maximum = float(np.max(values, initial=0.0))
        return values / maximum if maximum > 0.0 else values

    def observe(self, query: np.ndarray, receptor: int, utility: float) -> None:
        features = self._features(query)
        if not np.any(features) or utility == 0.0:
            return
        sums = self.utility_sum.setdefault(
            receptor, np.zeros(self.dimensions, dtype=np.float64)
        )
        help_sum = self.help_sum.setdefault(
            receptor, np.zeros(self.dimensions, dtype=np.float64)
        )
        harm_sum = self.harm_sum.setdefault(
            receptor, np.zeros(self.dimensions, dtype=np.float64)
        )
        evidence = self.evidence.setdefault(
            receptor, np.zeros(self.dimensions, dtype=np.float64)
        )
        sums += features * float(utility)
        help_sum += features * max(0.0, float(utility))
        harm_sum += features * max(0.0, -float(utility))
        evidence += features
        self.trials += 1
        if utility > 0.0:
            self.positive_trials += 1
            self.positive_utility += float(utility)
        else:
            self.negative_trials += 1
            self.negative_utility += abs(float(utility))

    def value(self, query: np.ndarray, receptor: int) -> float:
        features = self._features(query)
        help_sum = self.help_sum.get(receptor)
        harm_sum = self.harm_sum.get(receptor)
        evidence = self.evidence.get(receptor)
        if (
            help_sum is None
            or harm_sum is None
            or evidence is None
            or not np.any(features)
        ):
            return 0.0
        local = (help_sum - self.risk_weight * harm_sum) / (
            evidence + self.prior
        )
        return float(np.dot(features, local) / max(float(np.sum(features)), 1e-12))

    def permission(self, query: np.ndarray, receptor: int) -> float:
        return float(1.0 - np.exp(-max(0.0, self.value(query, receptor))))


def _develop(
    payload: dict[str, object], stream: bytes, *, risk_weight: float
) -> tuple[object, GeometricPermission]:
    arm = _new_arm(payload)
    _enable_relational_access(
        arm,
        expression_gain=0.5,
        frames_per_capsule=12,
        access_proposal_threshold=0.0,
    )
    arm.relational_access.learning_mode = "branches"
    arm.temporal_credit_mode = "receptor_relation_residual_utility"
    arm.lazy_expression_enabled = False
    permission = GeometricPermission(
        arm.field.cortical_dimensions, risk_weight=risk_weight
    )
    for byte_id in stream:
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        query = arm.relational_access.current_query.copy()
        for relation, activity in (
            arm.pending_receptor_relational_population or {}
        ).items():
            _engram, assembly, receptor = relation
            item = assembly, receptor
            candidate = phase147._assembly_overlay_distribution(
                arm.learner,
                (item,),
                support={item: float(activity)},
            )
            if np.array_equal(candidate, baseline):
                continue
            utility = float(
                np.log2(max(candidate[byte_id], 1e-300))
                - np.log2(max(baseline[byte_id], 1e-300))
            )
            permission.observe(query, receptor, utility)
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)
    return arm, permission


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
    risk_weight: float,
    permission_competition: str = "all",
) -> dict[str, object]:
    if permission_competition not in {"all", "global", "onset_global"}:
        raise ValueError("unknown permission competition")
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    tiny = tiny_path.read_bytes()
    secondary = None if secondary_path is None else secondary_path.read_bytes()
    required = development_bytes + scan_bytes
    stream = (
        tiny[:required]
        if secondary is None
        else _development_stream(
            tiny,
            secondary,
            mode="mixed",
            byte_count=required,
            chunk_bytes=256,
        )
    )
    started = perf_counter()
    arm, permission = _develop(
        payload, stream[:development_bytes], risk_weight=risk_weight
    )
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    rows = []
    proposed = 0
    permitted = 0
    values = []
    previous_relations: set[tuple[int, int, int]] = set()
    for offset, byte_id in enumerate(stream[development_bytes:required]):
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        query = arm.relational_access.current_query.copy()
        population = arm.pending_receptor_relational_population or {}
        proposed += int(bool(population))
        support: dict[tuple[int, int], float] = {}
        for relation, activity in population.items():
            if (
                permission_competition == "onset_global"
                and relation in previous_relations
            ):
                continue
            _engram, assembly, receptor = relation
            value = permission.value(query, receptor)
            values.append(value)
            amount = float(activity) * permission.permission(query, receptor)
            if amount > 0.0:
                item = assembly, receptor
                support[item] = support.get(item, 0.0) + amount
        if support and permission_competition in {"global", "onset_global"}:
            winner = max(support, key=lambda item: (support[item], -item[0], -item[1]))
            support = {winner: support[winner]}
        items = tuple(sorted(support))
        scrambled_items = phase147._scramble_assemblies(
            arm.learner, items
        ) if items else None
        if scrambled_items is not None:
            learned = phase147._assembly_overlay_distribution(
                arm.learner, items, support=support
            )
            scrambled_support = {
                replacement: support[item]
                for item, replacement in zip(items, scrambled_items, strict=True)
            }
            scrambled = phase147._assembly_overlay_distribution(
                arm.learner, scrambled_items, support=scrambled_support
            )
            variation = 0.5 * float(np.sum(np.abs(learned - baseline)))
            scrambled_variation = 0.5 * float(np.sum(np.abs(scrambled - baseline)))
            if variation > 0.0 or scrambled_variation > 0.0:
                permitted += 1
                rows.append(
                    {
                        "position": development_bytes + offset,
                        "items": len(items),
                        "learned_utility": float(
                            np.log2(max(learned[byte_id], 1e-300))
                            - np.log2(max(baseline[byte_id], 1e-300))
                        ),
                        "scrambled_utility": float(
                            np.log2(max(scrambled[byte_id], 1e-300))
                            - np.log2(max(baseline[byte_id], 1e-300))
                        ),
                        "variation": variation,
                    }
                )
        _advance_generated_state(arm, byte_id, float(baseline[byte_id]))
        previous_relations = set(population)
        arm.pending_temporal_population = {}
        arm.pending_temporal_predictions = {}
        arm.pending_temporal_global_prediction = None
        arm.pending_relational_receptors = {}
        arm.pending_receptor_relational_population = {}
    aggregate = {}
    for key in ("learned_utility", "scrambled_utility", "variation"):
        samples = np.asarray([row[key] for row in rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(samples)) if len(samples) else 0.0,
            "sum": float(np.sum(samples)) if len(samples) else 0.0,
            "positive_fraction": float(np.mean(samples > 0.0)) if len(samples) else 0.0,
        }
    difference = np.asarray(
        [row["learned_utility"] - row["scrambled_utility"] for row in rows],
        dtype=np.float64,
    )
    aggregate["identity_advantage"] = {
        "mean": float(np.mean(difference)) if len(difference) else 0.0,
        "positive_fraction": (
            float(np.mean(difference > 0.0)) if len(difference) else 0.0
        ),
    }
    return {
        "task": "phase148_geometric_permission",
        "state": str(state),
        "tiny_path": str(tiny_path),
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "risk_weight": risk_weight,
        "permission_competition": permission_competition,
        "permission": {
            "trials": permission.trials,
            "positive_trials": permission.positive_trials,
            "negative_trials": permission.negative_trials,
            "positive_utility": permission.positive_utility,
            "negative_utility": permission.negative_utility,
            "receptors": len(permission.utility_sum),
            "scan_value_mean": float(np.mean(values)) if values else 0.0,
            "scan_value_positive_fraction": (
                float(np.mean(np.asarray(values) > 0.0)) if values else 0.0
            ),
        },
        "proposed_returns": proposed,
        "permitted_returns": permitted,
        "aggregate": aggregate,
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("tiny", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--development-bytes", type=int, default=1024)
    parser.add_argument("--scan-bytes", type=int, default=1024)
    parser.add_argument("--risk-weight", type=float, default=4.0)
    parser.add_argument(
        "--permission-competition",
        choices=("all", "global", "onset_global"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        scan_bytes=args.scan_bytes,
        risk_weight=args.risk_weight,
        permission_competition=args.permission_competition,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
