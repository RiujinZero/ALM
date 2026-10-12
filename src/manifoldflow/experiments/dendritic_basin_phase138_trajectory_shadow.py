"""Score live relation returns across their complete learned phase trajectory."""

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

HORIZONS = (1, 8, 16, 32, 64, 128)


def _scramble_active(expression: object) -> bool:
    """Swap identity while preserving receptor, phase, and return strength."""
    swapped = {}
    for item, (strength, phase) in expression.active.items():
        assembly, receptor = item
        candidates = sorted(
            candidate_assembly
            for (
                candidate_assembly,
                candidate_receptor,
                candidate_phase,
            ) in expression.evidence
            if candidate_receptor == receptor
            and candidate_phase == phase
            and candidate_assembly != assembly
            and expression.evidence[
                (candidate_assembly, candidate_receptor, candidate_phase)
            ]
            >= expression.minimum_evidence
        )
        if not candidates:
            return False
        replacement = candidates[assembly % len(candidates)], receptor
        swapped[replacement] = (strength, phase)
    expression.active = swapped
    return bool(swapped)


def _shadow_trajectory(
    arm: object,
    future: bytes,
    *,
    scramble: bool,
) -> dict[str, object] | None:
    learner = pickle.loads(pickle.dumps(arm.learner, protocol=pickle.HIGHEST_PROTOCOL))
    expression = pickle.loads(
        pickle.dumps(arm.recalled_expression, protocol=pickle.HIGHEST_PROTOCOL)
    )
    if scramble and not _scramble_active(expression):
        return None
    cumulative = 0.0
    savings = {}
    active_steps = 0
    for step, byte_id in enumerate(future[: max(HORIZONS)], start=1):
        baseline = learner.predict_next_distribution()
        result = expression.predict(
            baseline,
            maximum_gain=arm.expression_gain,
            shared_cancellation=arm.expression_cancellation_mode,
            maximum_current_norm=arm.expression_current_limit,
        )
        cumulative += float(
            np.log2(max(result.distribution[byte_id], 1e-300))
            - np.log2(max(baseline[byte_id], 1e-300))
        )
        active_steps += int(result.gain > 0.0)
        if step in HORIZONS:
            savings[str(step)] = cumulative
        expression.observe(byte_id, external=False)
        learner._observe_probability_only(byte_id, learn=False)
    return {
        "savings": savings,
        "active_steps": active_steps,
        "active_rows": len(arm.recalled_expression.active),
    }


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
    probes = []
    last_probe = -minimum_probe_spacing
    for position in range(development_bytes, development_bytes + scan_bytes):
        baseline = arm.learner.predict_next_distribution()
        distribution = _predict(arm, live=True)
        variation = 0.5 * float(np.sum(np.abs(distribution - baseline)))
        future = stream[position : position + max(HORIZONS)]
        if (
            variation > 1e-9
            and len(future) == max(HORIZONS)
            and len(probes) < maximum_probes
            and position - last_probe >= minimum_probe_spacing
        ):
            natural = _shadow_trajectory(arm, future, scramble=False)
            scrambled = _shadow_trajectory(arm, future, scramble=True)
            if natural is not None and scrambled is not None:
                probes.append(
                    {
                        "position": position,
                        "variation": variation,
                        "natural": natural,
                        "identity_scrambled": scrambled,
                    }
                )
                last_probe = position
        byte_id = stream[position]
        _advance_generated_state(arm, byte_id, float(distribution[byte_id]))
    aggregate = {}
    for horizon in HORIZONS:
        key = str(horizon)
        natural = [row["natural"]["savings"][key] for row in probes]
        scrambled = [row["identity_scrambled"]["savings"][key] for row in probes]
        aggregate[key] = {
            "natural_mean_bits_saved": float(np.mean(natural)) if natural else 0.0,
            "natural_positive_fraction": (
                float(np.mean(np.asarray(natural) > 0.0)) if natural else 0.0
            ),
            "scrambled_mean_bits_saved": (
                float(np.mean(scrambled)) if scrambled else 0.0
            ),
            "scrambled_positive_fraction": (
                float(np.mean(np.asarray(scrambled) > 0.0)) if scrambled else 0.0
            ),
        }
    return {
        "task": "phase138_trajectory_shadow",
        "state": str(state),
        "tiny_path": str(tiny_path),
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "probes": len(probes),
        "aggregate": aggregate,
        "rows": probes,
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
