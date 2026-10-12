"""Test reusable geometric permission with online identity-specific residual routes."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_phase145_residual_route_codevelopment as phase145,
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
from manifoldflow.experiments.dendritic_basin_phase148_geometric_permission import (
    GeometricPermission,
)


def _candidate_predictions(
    arm: object, baseline: np.ndarray, *, gain: float
) -> dict[tuple[int, int, int], np.ndarray]:
    predictions = {}
    for relation, activity in (
        arm.pending_receptor_relational_population or {}
    ).items():
        _engram, assembly, receptor = relation
        if (receptor, assembly) not in arm.relational_residual_routes:
            continue
        item = assembly, receptor
        candidate = phase145._residual_overlay_distribution(
            arm.learner,
            arm.relational_residual_routes,
            (item,),
            support={item: float(activity)},
            gain=gain,
            baseline=baseline,
        )
        if not np.array_equal(candidate, baseline):
            predictions[relation] = candidate
    return predictions


def _utility(candidate: np.ndarray, baseline: np.ndarray, byte_id: int) -> float:
    return float(
        np.log2(max(candidate[byte_id], 1e-300))
        - np.log2(max(baseline[byte_id], 1e-300))
    )


def _configure(payload: dict[str, object]) -> object:
    arm = _new_arm(payload)
    _enable_relational_access(
        arm,
        expression_gain=0.5,
        frames_per_capsule=12,
        access_proposal_threshold=0.0,
    )
    arm.relational_access.learning_mode = "branches"
    arm.temporal_credit_mode = "receptor_relation_residual_utility"
    arm.relational_residual_codevelopment = True
    arm.lazy_expression_enabled = False
    return arm


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
    gain: float,
    risk_weight: float,
    permission_competition: str,
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
            tiny, secondary, mode="mixed", byte_count=required, chunk_bytes=256
        )
    )
    started = perf_counter()
    arm = _configure(payload)
    permission = GeometricPermission(
        arm.field.cortical_dimensions, risk_weight=risk_weight
    )
    development_shadow = 0
    for byte_id in stream[:development_bytes]:
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        query = arm.relational_access.current_query.copy()
        candidates = _candidate_predictions(arm, baseline, gain=gain)
        for relation, candidate in candidates.items():
            permission.observe(
                query, relation[2], _utility(candidate, baseline, byte_id)
            )
        development_shadow += len(candidates)
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)

    rows: list[dict[str, object]] = []
    proposed = 0
    candidates_scored = 0
    released = 0
    previous_relations: set[tuple[int, int, int]] = set()
    routes_before_scan = len(arm.relational_residual_routes)
    for offset, byte_id in enumerate(stream[development_bytes:required]):
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        query = arm.relational_access.current_query.copy()
        population = arm.pending_receptor_relational_population or {}
        proposed += int(bool(population))
        candidates = _candidate_predictions(arm, baseline, gain=gain)
        candidates_scored += len(candidates)

        support: dict[tuple[int, int], float] = {}
        values: dict[tuple[int, int], float] = {}
        for relation, activity in population.items():
            if relation not in candidates:
                continue
            if (
                permission_competition == "onset_global"
                and relation in previous_relations
            ):
                continue
            _engram, assembly, receptor = relation
            value = permission.value(query, receptor)
            amount = float(activity) * permission.permission(query, receptor)
            if amount > 0.0:
                item = assembly, receptor
                support[item] = support.get(item, 0.0) + amount
                values[item] = max(values.get(item, float("-inf")), value)
        if support and permission_competition in {"global", "onset_global"}:
            winner = max(support, key=lambda item: (support[item], -item[0], -item[1]))
            support = {winner: support[winner]}
            values = {winner: values[winner]}

        items = tuple(sorted(support))
        if items:
            natural = phase145._residual_overlay_distribution(
                arm.learner,
                arm.relational_residual_routes,
                items,
                support=support,
                gain=gain,
                baseline=baseline,
            )
            variation = 0.5 * float(np.sum(np.abs(natural - baseline)))
            if variation > 0.0:
                released += 1
                scrambled_items = phase145._scramble_residual_items(
                    arm.relational_residual_routes, items
                )
                scrambled_utility = None
                if scrambled_items is not None:
                    scrambled_support = {
                        replacement: support[item]
                        for item, replacement in zip(
                            items, scrambled_items, strict=True
                        )
                    }
                    scrambled = phase145._residual_overlay_distribution(
                        arm.learner,
                        arm.relational_residual_routes,
                        scrambled_items,
                        support=scrambled_support,
                        gain=gain,
                        baseline=baseline,
                    )
                    scrambled_utility = _utility(scrambled, baseline, byte_id)
                rows.append(
                    {
                        "position": development_bytes + offset,
                        "items": len(items),
                        "permission_value": max(values.values()),
                        "natural_utility": _utility(natural, baseline, byte_id),
                        "scrambled_utility": scrambled_utility,
                        "variation": variation,
                    }
                )

        # The observed byte may calibrate future decisions and grow a new
        # identity-specific route, but cannot influence the decision just scored.
        for relation, candidate in candidates.items():
            permission.observe(
                query, relation[2], _utility(candidate, baseline, byte_id)
            )
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)
        previous_relations = set(population)

    def summary(key: str) -> dict[str, float | int]:
        samples = np.asarray(
            [row[key] for row in rows if row[key] is not None], dtype=np.float64
        )
        return {
            "count": len(samples),
            "sum": float(np.sum(samples)) if len(samples) else 0.0,
            "mean": float(np.mean(samples)) if len(samples) else 0.0,
            "positive_fraction": float(np.mean(samples > 0.0)) if len(samples) else 0.0,
        }

    paired = [
        float(row["natural_utility"]) - float(row["scrambled_utility"])
        for row in rows
        if row["scrambled_utility"] is not None
    ]
    return {
        "task": "phase150_online_residual_permission",
        "state": str(state),
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "gain": gain,
        "risk_weight": risk_weight,
        "permission_competition": permission_competition,
        "development_shadow_trials": development_shadow,
        "permission_trials": permission.trials,
        "permission_positive_trials": permission.positive_trials,
        "permission_negative_trials": permission.negative_trials,
        "routes_before_scan": routes_before_scan,
        "routes_after_scan": len(arm.relational_residual_routes),
        "proposed_returns": proposed,
        "candidate_relations_scored": candidates_scored,
        "released_returns": released,
        "natural": summary("natural_utility"),
        "scrambled": summary("scrambled_utility"),
        "identity_advantage": {
            "count": len(paired),
            "mean": float(np.mean(paired)) if paired else 0.0,
            "positive_fraction": (
                float(np.mean(np.asarray(paired) > 0.0)) if paired else 0.0
            ),
        },
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
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--risk-weight", type=float, default=4.0)
    parser.add_argument(
        "--permission-competition",
        choices=("all", "global", "onset_global"),
        default="onset_global",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        scan_bytes=args.scan_bytes,
        gain=args.gain,
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
