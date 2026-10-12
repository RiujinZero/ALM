"""Calibrate residual cross-organ permission from exact external byte utility."""

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
from manifoldflow.experiments.dendritic_basin_phase102_phase_aligned_chain import (
    _advance_generated_state,
    _temporal_eligibility_statistics,
)


def _shadow_predictions(
    arm: object,
    baseline: np.ndarray,
    *,
    gain: float,
) -> dict[object, tuple[np.ndarray, np.ndarray]]:
    predictions = {}
    for relation, activity in (
        arm.pending_receptor_relational_population or {}
    ).items():
        _engram, assembly, receptor = relation
        item = assembly, receptor
        if (receptor, assembly) not in arm.relational_residual_routes:
            continue
        candidate = phase145._residual_overlay_distribution(
            arm.learner,
            arm.relational_residual_routes,
            (item,),
            support={item: float(activity)},
            gain=gain,
            baseline=baseline,
        )
        if not np.array_equal(candidate, baseline):
            predictions[relation] = baseline, candidate
    return predictions


def _develop(
    payload: dict[str, object],
    stream: bytes,
    *,
    gain: float,
) -> tuple[object, dict[str, float]]:
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
    proposed = 0
    shadowed = 0
    shadow_variation = 0.0
    for byte_id in stream:
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        proposed += len(arm.pending_receptor_relational_population or {})
        predictions = _shadow_predictions(arm, baseline, gain=gain)
        arm.pending_temporal_predictions = predictions
        shadowed += len(predictions)
        shadow_variation += sum(
            0.5 * float(np.sum(np.abs(candidate - baseline_distribution)))
            for baseline_distribution, candidate in predictions.values()
        )
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)
    return arm, {
        "proposed_relations": proposed,
        "shadowed_relations": shadowed,
        "shadow_total_variation": shadow_variation,
    }


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
    replay_scan: bool,
    gain: float,
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
    arm, calibration = _develop(payload, stream[:development_bytes], gain=gain)
    learned_before_scan = _temporal_eligibility_statistics(arm)
    residual_route_count = len(arm.relational_residual_routes)
    residual_synapses = sum(
        len(row) for row in arm.relational_residual_routes.values()
    )
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    arm.relational_residual_codevelopment = False
    if replay_scan:
        phase145._reset_dynamic_state_for_replay(arm)
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
        learned_support = phase145._learned_support(arm, 1.0)
        support = {
            item: amount
            for item, amount in learned_support.items()
            if amount > 0.0
            and (item[1], item[0]) in arm.relational_residual_routes
        }
        items = tuple(sorted(support))
        scrambled_items = phase145._scramble_residual_items(
            arm.relational_residual_routes, items
        ) if items else None
        if scrambled_items is not None:
            learned = phase145._residual_overlay_distribution(
                arm.learner,
                arm.relational_residual_routes,
                items,
                support=support,
                gain=gain,
                baseline=baseline,
            )
            scrambled_support = {
                replacement: support[item]
                for item, replacement in zip(items, scrambled_items, strict=True)
            }
            scrambled = phase145._residual_overlay_distribution(
                arm.learner,
                arm.relational_residual_routes,
                scrambled_items,
                support=scrambled_support,
                gain=gain,
                baseline=baseline,
            )
            variation = 0.5 * float(np.sum(np.abs(learned - baseline)))
            if variation > 0.0 or not np.array_equal(scrambled, baseline):
                permitted += 1
                rows.append(
                    {
                        "position": position,
                        "items": len(items),
                        "maximum_support": max(support.values()),
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
            "positive_fraction": float(np.mean(values > 0.0)) if len(values) else 0.0,
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
        "task": "phase146_signed_residual_permission",
        "state": str(state),
        "tiny_path": str(tiny_path),
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "replay_scan": replay_scan,
        "gain": gain,
        "calibration": calibration,
        "relational_residual_learning_events": (
            arm.relational_residual_learning_events
        ),
        "residual_routes": residual_route_count,
        "residual_synapses": residual_synapses,
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
    parser.add_argument("--replay-scan", action="store_true")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        scan_bytes=args.scan_bytes,
        replay_scan=args.replay_scan,
        gain=args.gain,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
