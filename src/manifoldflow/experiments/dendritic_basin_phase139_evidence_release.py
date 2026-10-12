"""Measure whether local prediction mismatch can release a recalled trajectory."""

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
)
from manifoldflow.experiments.dendritic_basin_phase138_trajectory_shadow import (
    HORIZONS,
)

FIXED_RELEASE_STEPS = (1, 2, 4, 8, 16, 32, 128)
MISMATCH_STREAKS = (1, 2, 4)


def _shadow_with_release(
    arm: object,
    future: bytes,
    *,
    maximum_steps: int = 128,
    mismatch_streak: int = 0,
) -> dict[str, object]:
    """Score one frozen return, then release it using only observed mismatch."""
    learner = pickle.loads(pickle.dumps(arm.learner, protocol=pickle.HIGHEST_PROTOCOL))
    expression = pickle.loads(
        pickle.dumps(arm.recalled_expression, protocol=pickle.HIGHEST_PROTOCOL)
    )
    cumulative = 0.0
    savings: dict[str, float] = {}
    consecutive_harm = 0
    released_at: int | None = None
    for step, byte_id in enumerate(future[: max(HORIZONS)], start=1):
        baseline = learner.predict_next_distribution()
        result = expression.predict(
            baseline,
            maximum_gain=arm.expression_gain,
            shared_cancellation=arm.expression_cancellation_mode,
            maximum_current_norm=arm.expression_current_limit,
        )
        utility = float(
            np.log2(max(result.distribution[byte_id], 1e-300))
            - np.log2(max(baseline[byte_id], 1e-300))
        )
        cumulative += utility
        if utility < 0.0 and result.gain > 0.0:
            consecutive_harm += 1
        else:
            consecutive_harm = 0
        expression.observe(byte_id, external=False)
        learner._observe_probability_only(byte_id, learn=False)
        should_release = step >= maximum_steps or (
            mismatch_streak > 0 and consecutive_harm >= mismatch_streak
        )
        if should_release and expression.active:
            expression.reset_episode()
            released_at = step
        if step in HORIZONS:
            savings[str(step)] = cumulative
    return {"savings": savings, "released_at": released_at}


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
    maximum_probes: int,
    minimum_probe_spacing: int,
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    tiny = tiny_path.read_bytes()
    secondary = None if secondary_path is None else secondary_path.read_bytes()
    required = development_bytes + scan_bytes + max(HORIZONS)
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
        temporal_credit_mode="relation_local",
    )
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    rows = []
    last_probe = -minimum_probe_spacing
    for position in range(development_bytes, development_bytes + scan_bytes):
        baseline = arm.learner.predict_next_distribution()
        distribution = _predict(arm, live=True)
        variation = 0.5 * float(np.sum(np.abs(distribution - baseline)))
        future = stream[position : position + max(HORIZONS)]
        if (
            variation > 1e-9
            and len(future) == max(HORIZONS)
            and len(rows) < maximum_probes
            and position - last_probe >= minimum_probe_spacing
        ):
            policies = {
                f"fixed_{steps}": _shadow_with_release(
                    arm, future, maximum_steps=steps
                )
                for steps in FIXED_RELEASE_STEPS
            }
            policies.update(
                {
                    f"mismatch_{streak}": _shadow_with_release(
                        arm,
                        future,
                        mismatch_streak=streak,
                    )
                    for streak in MISMATCH_STREAKS
                }
            )
            rows.append(
                {
                    "position": position,
                    "variation": variation,
                    "policies": policies,
                }
            )
            last_probe = position
        byte_id = stream[position]
        _advance_generated_state(arm, byte_id, float(distribution[byte_id]))
    aggregate = {}
    for policy in rows[0]["policies"] if rows else ():
        aggregate[policy] = {}
        for horizon in HORIZONS:
            values = [
                row["policies"][policy]["savings"][str(horizon)] for row in rows
            ]
            aggregate[policy][str(horizon)] = {
                "mean_bits_saved": float(np.mean(values)),
                "positive_fraction": float(np.mean(np.asarray(values) > 0.0)),
            }
    return {
        "task": "phase139_evidence_sensitive_release",
        "state": str(state),
        "tiny_path": str(tiny_path),
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "probes": len(rows),
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
    parser.add_argument("--scan-bytes", type=int, default=256)
    parser.add_argument("--maximum-probes", type=int, default=8)
    parser.add_argument("--minimum-probe-spacing", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        scan_bytes=args.scan_bytes,
        maximum_probes=args.maximum_probes,
        minimum_probe_spacing=args.minimum_probe_spacing,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
