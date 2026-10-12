"""Apply learned receptor-complete eligibility as cortical route support."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.experiments.dendritic_basin_phase76_generation import _predict
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _development_stream,
)
from manifoldflow.experiments.dendritic_basin_phase102_phase_aligned_chain import (
    _advance_generated_state,
    _prepare_development,
    _temporal_eligibility_statistics,
)
from manifoldflow.experiments.dendritic_basin_phase140_cortical_route_return import (
    _route_overlay_distribution,
    _scrambled_items,
)


def _learned_support(arm: object, scale: float) -> dict[tuple[int, int], float]:
    support: dict[tuple[int, int], float] = {}
    tissue = arm.temporal_relational_eligibility
    for relation, activity in (
        arm.pending_receptor_relational_population or {}
    ).items():
        _engram, assembly, receptor = relation
        amount = scale * float(activity) * tissue.permission(relation)
        item = assembly, receptor
        support[item] = support.get(item, 0.0) + amount
    return support


def _routable_support(
    learner: object,
    support: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """Keep only learned relations that already exist in the cortical route tissue."""
    routed = {}
    for (assembly, receptor), amount in support.items():
        route_receptor = learner.temporal_bins[0] if receptor == 0 else receptor
        if (route_receptor, assembly) in learner.route_counts:
            routed[assembly, receptor] = amount
    return routed


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
    support_scale: float,
    co_develop_routes: bool = False,
    replay_scan: bool = False,
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    tiny = tiny_path.read_bytes()
    secondary = None if secondary_path is None else secondary_path.read_bytes()
    required = max(development_bytes, scan_bytes) if replay_scan else (
        development_bytes + scan_bytes
    )
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
    arm = _prepare_development(
        payload,
        stream[:development_bytes],
        mode="coupled",
        access_learning_mode="branches",
        frames_per_capsule=12,
        access_proposal_threshold=0.0,
        temporal_credit_mode="receptor_relation",
        relational_route_codevelopment=co_develop_routes,
    )
    learned_before_scan = _temporal_eligibility_statistics(arm)
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    rows = []
    proposed = 0
    permitted = 0
    scan_stream = (
        stream[:scan_bytes] if replay_scan else stream[development_bytes:required]
    )
    for offset, byte_id in enumerate(scan_stream):
        position = offset if replay_scan else development_bytes + offset
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        population = arm.pending_receptor_relational_population or {}
        proposed += int(bool(population))
        learned_support = _learned_support(arm, support_scale)
        support = _routable_support(arm.learner, learned_support)
        items = tuple(sorted(item for item, value in support.items() if value > 0.0))
        scrambled_items = _scrambled_items(arm.learner, items) if items else None
        if items and scrambled_items is not None:
            learned = _route_overlay_distribution(
                arm.learner,
                items,
                support=support,
            )
            scrambled_support = {
                replacement: support[item]
                for item, replacement in zip(items, scrambled_items, strict=True)
            }
            scrambled = _route_overlay_distribution(
                arm.learner,
                scrambled_items,
                support=scrambled_support,
            )
            if learned is not None and scrambled is not None:
                permitted += 1
                rows.append(
                    {
                        "position": position,
                        "relations": len(population),
                        "learned_items": len(learned_support),
                        "supported_items": len(items),
                        "maximum_support": max(support.values()),
                        "learned_utility": float(
                            np.log2(max(learned[byte_id], 1e-300))
                            - np.log2(max(baseline[byte_id], 1e-300))
                        ),
                        "scrambled_utility": float(
                            np.log2(max(scrambled[byte_id], 1e-300))
                            - np.log2(max(baseline[byte_id], 1e-300))
                        ),
                        "variation": 0.5
                        * float(np.sum(np.abs(learned - baseline))),
                    }
                )
        _advance_generated_state(
            arm,
            byte_id,
            float(baseline[byte_id]),
        )
        arm.pending_temporal_population = {}
        arm.pending_temporal_predictions = {}
        arm.pending_temporal_global_prediction = None
        arm.pending_relational_receptors = {}
        arm.pending_receptor_relational_population = {}
    aggregate = {}
    for key in ("learned_utility", "scrambled_utility", "variation"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "sum": float(np.sum(values)) if len(values) else 0.0,
            "positive_fraction": (
                float(np.mean(values > 0.0)) if len(values) else 0.0
            ),
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
        "task": "phase143_learned_route_permission",
        "state": str(state),
        "tiny_path": str(tiny_path),
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "support_scale": support_scale,
        "co_develop_routes": co_develop_routes,
        "replay_scan": replay_scan,
        "relational_route_learning_events": arm.relational_route_learning_events,
        "proposed_returns": proposed,
        "permitted_returns": permitted,
        "learned_before_scan": learned_before_scan,
        "aggregate": aggregate,
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("tiny", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--development-bytes", type=int, default=2048)
    parser.add_argument("--scan-bytes", type=int, default=1024)
    parser.add_argument("--support-scale", type=float, default=1.0)
    parser.add_argument("--co-develop-routes", action="store_true")
    parser.add_argument("--replay-scan", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        scan_bytes=args.scan_bytes,
        support_scale=args.support_scale,
        co_develop_routes=args.co_develop_routes,
        replay_scan=args.replay_scan,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
