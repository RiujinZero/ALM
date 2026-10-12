"""Test reusable cortical assembly return without identity-specific route rows."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_phase145_residual_route_codevelopment as phase145,
)
from manifoldflow.experiments import (
    dendritic_basin_phase146_signed_residual_permission as phase146,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import _predict
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _development_stream,
)
from manifoldflow.experiments.dendritic_basin_phase102_phase_aligned_chain import (
    _advance_generated_state,
    _temporal_eligibility_statistics,
)
from manifoldflow.experiments.dendritic_basin_phase143_learned_route_permission import (
    _learned_support,
)


def _assembly_overlay_distribution(
    learner: object,
    items: tuple[tuple[int, int], ...],
    *,
    support: Mapping[tuple[int, int], float],
) -> np.ndarray:
    """Reopen returned identities in the existing cortical motif pathway."""
    original = learner.active_word_candidates
    candidates: dict[int, float] = {}
    for item in items:
        assembly, _receptor = item
        amount = float(support.get(item, 0.0))
        if amount > 0.0 and 0 <= assembly < len(learner.word_forms):
            candidates[assembly] = candidates.get(assembly, 0.0) + amount
    if not candidates:
        return learner.predict_next_distribution().copy()
    try:
        learner.active_word_candidates = candidates
        return learner.predict_next_distribution().copy()
    finally:
        learner.active_word_candidates = original


def _scramble_assemblies(
    learner: object,
    items: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...] | None:
    prefix = bytes(learner.current_word)
    excluded = {assembly for assembly, _receptor in items}
    pool = tuple(
        assembly
        for assembly, form in enumerate(learner.word_forms)
        if assembly not in excluded
        and len(form) > len(prefix)
        and form.startswith(prefix)
    )
    if not pool:
        return None
    return tuple(
        (pool[assembly % len(pool)], receptor) for assembly, receptor in items
    )


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
    replay_scan: bool,
    permission_mode: str,
) -> dict[str, object]:
    if permission_mode not in {"learned", "activity"}:
        raise ValueError("unknown permission mode")
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
    arm, calibration = phase146._develop(
        payload, stream[:development_bytes], gain=1.0
    )
    learned_before_scan = _temporal_eligibility_statistics(arm)
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    arm.relational_residual_codevelopment = False
    if replay_scan:
        phase145._reset_dynamic_state_for_replay(arm)
    rows = []
    proposed = 0
    scan_stream = (
        stream[:scan_bytes] if replay_scan else stream[development_bytes:required]
    )
    for offset, byte_id in enumerate(scan_stream):
        position = offset if replay_scan else development_bytes + offset
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        population = arm.pending_receptor_relational_population or {}
        proposed += int(bool(population))
        support = (
            _learned_support(arm, 1.0)
            if permission_mode == "learned"
            else phase145._population_support(arm)
        )
        support = {item: amount for item, amount in support.items() if amount > 0.0}
        items = tuple(sorted(support))
        scrambled_items = _scramble_assemblies(arm.learner, items) if items else None
        if scrambled_items is not None:
            learned = _assembly_overlay_distribution(
                arm.learner, items, support=support
            )
            scrambled_support = {
                replacement: support[item]
                for item, replacement in zip(items, scrambled_items, strict=True)
            }
            scrambled = _assembly_overlay_distribution(
                arm.learner, scrambled_items, support=scrambled_support
            )
            variation = 0.5 * float(np.sum(np.abs(learned - baseline)))
            scrambled_variation = 0.5 * float(np.sum(np.abs(scrambled - baseline)))
            if variation > 0.0 or scrambled_variation > 0.0:
                rows.append(
                    {
                        "position": position,
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
        "task": "phase147_generic_assembly_return",
        "state": str(state),
        "tiny_path": str(tiny_path),
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "replay_scan": replay_scan,
        "permission_mode": permission_mode,
        "calibration": calibration,
        "learned_before_scan": learned_before_scan,
        "proposed_returns": proposed,
        "scored_returns": len(rows),
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
    parser.add_argument("--replay-scan", action="store_true")
    parser.add_argument(
        "--permission-mode", choices=("learned", "activity"), default="learned"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        scan_bytes=args.scan_bytes,
        replay_scan=args.replay_scan,
        permission_mode=args.permission_mode,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
