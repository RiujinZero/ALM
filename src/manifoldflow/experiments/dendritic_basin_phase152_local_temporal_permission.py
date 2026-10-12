"""Test strictly local temporal credit for reusable residual-return permission."""

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
from manifoldflow.experiments import (
    dendritic_basin_phase151_temporal_geometric_permission as phase151,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _observe,
    _predict,
)
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _development_stream,
)


def _learn_candidates(
    tissue: phase151.TemporalGeometricPermission,
    arm: object,
    query: np.ndarray,
    candidates: dict[tuple[int, int, int], np.ndarray],
    baseline: np.ndarray,
    byte_id: int,
) -> None:
    population = arm.pending_receptor_relational_population or {}
    for relation in candidates:
        tissue.participate(query, relation[2], float(population[relation]))
    for relation, candidate in candidates.items():
        tissue.confirm_local(
            query,
            relation[2],
            phase150._utility(candidate, baseline, byte_id),
        )
    tissue.advance()


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
    tissue = phase151.TemporalGeometricPermission(
        arm.field.cortical_dimensions, risk_weight=risk_weight
    )
    development_trials = 0
    for byte_id in stream[:development_bytes]:
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        query = arm.relational_access.current_query.copy()
        candidates = phase150._candidate_predictions(arm, baseline, gain=gain)
        development_trials += len(candidates)
        _learn_candidates(tissue, arm, query, candidates, baseline, byte_id)
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)

    rows: list[dict[str, float | int]] = []
    routes_before = len(arm.relational_residual_routes)
    scan_trials = 0
    for offset, byte_id in enumerate(stream[development_bytes:required]):
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        query = arm.relational_access.current_query.copy()
        population = arm.pending_receptor_relational_population or {}
        candidates = phase150._candidate_predictions(arm, baseline, gain=gain)
        scan_trials += len(candidates)
        support: dict[tuple[int, int], float] = {}
        values: dict[tuple[int, int], float] = {}
        for relation, activity in population.items():
            if relation not in candidates:
                continue
            _engram, assembly, receptor = relation
            value = tissue.value(query, receptor)
            amount = float(activity) * tissue.permission(query, receptor)
            if amount > 0.0:
                item = assembly, receptor
                support[item] = support.get(item, 0.0) + amount
                values[item] = max(values.get(item, float("-inf")), value)
        if support:
            winner = max(support, key=lambda item: (support[item], -item[0], -item[1]))
            natural = phase145._residual_overlay_distribution(
                arm.learner,
                arm.relational_residual_routes,
                (winner,),
                support={winner: support[winner]},
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
        _learn_candidates(tissue, arm, query, candidates, baseline, byte_id)
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)

    utilities = np.asarray([row["utility"] for row in rows], dtype=np.float64)
    return {
        "task": "phase152_local_temporal_permission",
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "gain": gain,
        "risk_weight": risk_weight,
        "development_shadow_trials": development_trials,
        "scan_shadow_trials": scan_trials,
        "confirmations": tissue.confirmations,
        "positive_confirmations": tissue.positive_confirmations,
        "negative_confirmations": tissue.negative_confirmations,
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
