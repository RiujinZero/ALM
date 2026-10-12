"""Give reusable cross-organ permission a fading local eligibility trace."""

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
from manifoldflow.experiments import (
    dendritic_basin_phase150_online_residual_permission as phase150,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _observe,
    _predict,
)
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _development_stream,
)


class TemporalGeometricPermission:
    """Risk-sensitive dendrites whose participation, not content, persists."""

    def __init__(
        self,
        dimensions: int,
        *,
        half_lives: tuple[float, ...] = (8.0, 32.0, 128.0, 512.0),
        prior: float = 1.0,
        risk_weight: float = 4.0,
    ) -> None:
        self.dimensions = int(dimensions)
        self.half_lives = np.asarray(half_lives, dtype=np.float64)
        self.decay = 2.0 ** (-1.0 / self.half_lives)
        self.prior = float(prior)
        self.risk_weight = float(risk_weight)
        shape = (len(self.half_lives), self.dimensions)
        self.traces: dict[int, np.ndarray] = {}
        self.help_sum: dict[int, np.ndarray] = {}
        self.harm_sum: dict[int, np.ndarray] = {}
        self.evidence: dict[int, np.ndarray] = {}
        self.confirmations = 0
        self.positive_confirmations = 0
        self.negative_confirmations = 0
        self.trace_shape = shape

    def _features(self, query: np.ndarray) -> np.ndarray:
        values = np.abs(np.asarray(query, dtype=np.float64))
        maximum = float(np.max(values, initial=0.0))
        return values / maximum if maximum > 0.0 else values

    def participate(self, query: np.ndarray, receptor: int, amount: float) -> None:
        features = self._features(query) * min(1.0, max(0.0, float(amount)))
        if not np.any(features):
            return
        trace = self.traces.setdefault(
            int(receptor), np.zeros(self.trace_shape, dtype=np.float64)
        )
        np.maximum(trace, features[None, :], out=trace)

    def confirm(self, utility: float) -> None:
        """A later external consequence modulates only participating dendrites."""
        if utility == 0.0:
            return
        for receptor, trace in self.traces.items():
            help_sum = self.help_sum.setdefault(
                receptor, np.zeros(self.trace_shape, dtype=np.float64)
            )
            harm_sum = self.harm_sum.setdefault(
                receptor, np.zeros(self.trace_shape, dtype=np.float64)
            )
            evidence = self.evidence.setdefault(
                receptor, np.zeros(self.trace_shape, dtype=np.float64)
            )
            help_sum += trace * max(0.0, float(utility))
            harm_sum += trace * max(0.0, -float(utility))
            evidence += trace
        self.confirmations += 1
        self.positive_confirmations += int(utility > 0.0)
        self.negative_confirmations += int(utility < 0.0)

    def confirm_local(
        self, query: np.ndarray, receptor: int, utility: float
    ) -> None:
        """Modulate only old trace that overlaps present local evidence."""
        receptor = int(receptor)
        trace = self.traces.get(receptor)
        features = self._features(query)
        if trace is None or utility == 0.0 or not np.any(features):
            return
        eligible = trace * features[None, :]
        help_sum = self.help_sum.setdefault(
            receptor, np.zeros(self.trace_shape, dtype=np.float64)
        )
        harm_sum = self.harm_sum.setdefault(
            receptor, np.zeros(self.trace_shape, dtype=np.float64)
        )
        evidence = self.evidence.setdefault(
            receptor, np.zeros(self.trace_shape, dtype=np.float64)
        )
        help_sum += eligible * max(0.0, float(utility))
        harm_sum += eligible * max(0.0, -float(utility))
        evidence += eligible
        self.confirmations += 1
        self.positive_confirmations += int(utility > 0.0)
        self.negative_confirmations += int(utility < 0.0)

    def advance(self) -> None:
        for receptor in tuple(self.traces):
            self.traces[receptor] *= self.decay[:, None]
            if float(np.max(self.traces[receptor], initial=0.0)) < 1e-8:
                del self.traces[receptor]

    def value(self, query: np.ndarray, receptor: int) -> float:
        features = self._features(query)
        help_sum = self.help_sum.get(int(receptor))
        harm_sum = self.harm_sum.get(int(receptor))
        evidence = self.evidence.get(int(receptor))
        if help_sum is None or harm_sum is None or evidence is None:
            return 0.0
        local = (help_sum - self.risk_weight * harm_sum) / (
            evidence + self.prior
        )
        # A single lucky biochemical clock must not dominate release.  Average
        # the independently exposed timescales so temporal persistence earns
        # permission only when it is not contradicted by the other windows.
        values = np.sum(local * features[None, :], axis=1) / max(
            float(np.sum(features)), 1e-12
        )
        return float(np.mean(values))

    def permission(self, query: np.ndarray, receptor: int) -> float:
        return float(1.0 - np.exp(-max(0.0, self.value(query, receptor))))


def _joint_shadow(
    arm: object, baseline: np.ndarray, *, gain: float
) -> tuple[np.ndarray, dict[tuple[int, int], float]] | None:
    support: dict[tuple[int, int], float] = {}
    for relation, activity in (
        arm.pending_receptor_relational_population or {}
    ).items():
        _engram, assembly, receptor = relation
        if (receptor, assembly) in arm.relational_residual_routes:
            item = assembly, receptor
            support[item] = support.get(item, 0.0) + float(activity)
    if not support:
        return None
    candidate = phase145._residual_overlay_distribution(
        arm.learner,
        arm.relational_residual_routes,
        tuple(sorted(support)),
        support=support,
        gain=gain,
        baseline=baseline,
    )
    if np.array_equal(candidate, baseline):
        return None
    return candidate, support


def _learn_from_shadow(
    permission: TemporalGeometricPermission,
    arm: object,
    query: np.ndarray,
    shadow: tuple[np.ndarray, dict[tuple[int, int], float]] | None,
    baseline: np.ndarray,
    byte_id: int,
) -> float | None:
    if shadow is None:
        permission.advance()
        return None
    candidate, support = shadow
    for (assembly, receptor), amount in support.items():
        del assembly
        permission.participate(query, receptor, amount)
    utility = phase150._utility(candidate, baseline, byte_id)
    permission.confirm(utility)
    permission.advance()
    return utility


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
    gain: float,
    risk_weight: float,
) -> dict[str, object]:
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
    arm = phase150._configure(payload)
    permission = TemporalGeometricPermission(
        arm.field.cortical_dimensions, risk_weight=risk_weight
    )
    development_shadows = 0
    for byte_id in stream[:development_bytes]:
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        query = arm.relational_access.current_query.copy()
        shadow = _joint_shadow(arm, baseline, gain=gain)
        development_shadows += int(shadow is not None)
        _learn_from_shadow(permission, arm, query, shadow, baseline, byte_id)
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)

    rows: list[dict[str, float | int | None]] = []
    routes_before = len(arm.relational_residual_routes)
    scan_shadows = 0
    for offset, byte_id in enumerate(stream[development_bytes:required]):
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        query = arm.relational_access.current_query.copy()
        population = arm.pending_receptor_relational_population or {}
        support: dict[tuple[int, int], float] = {}
        values: dict[tuple[int, int], float] = {}
        for _relation, activity in population.items():
            _engram, assembly, receptor = _relation
            if (receptor, assembly) not in arm.relational_residual_routes:
                continue
            value = permission.value(query, receptor)
            amount = float(activity) * permission.permission(query, receptor)
            if amount > 0.0:
                item = assembly, receptor
                support[item] = support.get(item, 0.0) + amount
                values[item] = max(values.get(item, float("-inf")), value)
        if support:
            winner = max(support, key=lambda item: (support[item], -item[0], -item[1]))
            support = {winner: support[winner]}
            natural = phase145._residual_overlay_distribution(
                arm.learner,
                arm.relational_residual_routes,
                (winner,),
                support=support,
                gain=gain,
                baseline=baseline,
            )
            variation = 0.5 * float(np.sum(np.abs(natural - baseline)))
            if variation > 0.0:
                rows.append(
                    {
                        "position": development_bytes + offset,
                        "permission_value": values[winner],
                        "utility": phase150._utility(natural, baseline, byte_id),
                        "variation": variation,
                    }
                )
        shadow = _joint_shadow(arm, baseline, gain=gain)
        scan_shadows += int(shadow is not None)
        _learn_from_shadow(permission, arm, query, shadow, baseline, byte_id)
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)

    utilities = np.asarray([row["utility"] for row in rows], dtype=np.float64)
    return {
        "task": "phase151_temporal_geometric_permission",
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "gain": gain,
        "risk_weight": risk_weight,
        "development_shadow_events": development_shadows,
        "scan_shadow_events": scan_shadows,
        "confirmations": permission.confirmations,
        "positive_confirmations": permission.positive_confirmations,
        "negative_confirmations": permission.negative_confirmations,
        "routes_before_scan": routes_before,
        "routes_after_scan": len(arm.relational_residual_routes),
        "released_returns": len(rows),
        "utility_sum": float(np.sum(utilities)) if len(utilities) else 0.0,
        "utility_positive_fraction": (
            float(np.mean(utilities > 0.0)) if len(utilities) else 0.0
        ),
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
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
