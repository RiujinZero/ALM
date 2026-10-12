"""Reinstate recalled assemblies as provenance-preserving cortical context."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping
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


def _history_offset(learner: object, receptor: int) -> int | None:
    if receptor == 0:
        return 0
    try:
        return tuple(learner.temporal_bins).index(receptor)
    except ValueError:
        return None


def _context_overlay_distribution(
    learner: object,
    items: tuple[tuple[int, int], ...],
    *,
    support: Mapping[tuple[int, int], float],
) -> np.ndarray:
    """Place recall on an apical history wire without making an observation."""
    baseline = learner.predict_next_distribution().copy()
    original_history = learner.event_history.copy()
    original_candidates = learner.active_word_candidates
    history = original_history.copy()
    if not history:
        return baseline
    winners: dict[int, tuple[float, int]] = {}
    for assembly, receptor in items:
        offset = _history_offset(learner, receptor)
        amount = float(support.get((assembly, receptor), 0.0))
        if offset is None or offset >= len(history) or amount <= 0.0:
            continue
        current = winners.get(offset)
        if current is None or (amount, -assembly) > (current[0], -current[1]):
            winners[offset] = amount, assembly
    if not winners:
        return baseline
    try:
        for offset, (_amount, assembly) in winners.items():
            history[-1 - offset] = assembly
        learner.event_history = history
        learner.active_word_candidates = None
        return learner.predict_next_distribution().copy()
    finally:
        learner.event_history = original_history
        learner.active_word_candidates = original_candidates


def _population_support(arm: object) -> dict[tuple[int, int], float]:
    support: dict[tuple[int, int], float] = {}
    for (_engram, assembly, receptor), activity in (
        arm.pending_receptor_relational_population or {}
    ).items():
        item = assembly, receptor
        support[item] = max(support.get(item, 0.0), float(activity))
    return support


def _scramble_items(
    learner: object, items: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...] | None:
    vocabulary = len(learner.word_forms)
    excluded = {assembly for assembly, _receptor in items}
    if vocabulary <= len(excluded):
        return None
    replacements = []
    for assembly, receptor in items:
        candidates = [
            candidate
            for candidate in range(vocabulary)
            if candidate != assembly and candidate not in excluded
        ]
        if not candidates:
            return None
        replacements.append((candidates[assembly % len(candidates)], receptor))
    return tuple(replacements)


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
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
    arm = _prepare_development(
        payload,
        stream[:development_bytes],
        mode="coupled",
        access_learning_mode="branches",
        frames_per_capsule=12,
        access_proposal_threshold=0.0,
        temporal_credit_mode="receptor_relation",
    )
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    rows = []
    proposed = 0
    for offset, byte_id in enumerate(stream[development_bytes:required]):
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        support = _population_support(arm)
        proposed += int(bool(support))
        items = tuple(sorted(support))
        scrambled_items = _scramble_items(arm.learner, items) if items else None
        if scrambled_items is not None:
            natural = _context_overlay_distribution(
                arm.learner, items, support=support
            )
            scrambled_support = {
                replacement: support[item]
                for item, replacement in zip(items, scrambled_items, strict=True)
            }
            scrambled = _context_overlay_distribution(
                arm.learner, scrambled_items, support=scrambled_support
            )
            variation = 0.5 * float(np.sum(np.abs(natural - baseline)))
            if variation > 0.0 or not np.array_equal(scrambled, baseline):
                rows.append(
                    {
                        "position": development_bytes + offset,
                        "items": len(items),
                        "utility": float(
                            np.log2(max(natural[byte_id], 1e-300))
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
    utilities = np.asarray([row["utility"] for row in rows], dtype=np.float64)
    scrambled = np.asarray(
        [row["scrambled_utility"] for row in rows], dtype=np.float64
    )
    return {
        "task": "apical_context_return",
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "proposed_returns": proposed,
        "scored_returns": len(rows),
        "utility_sum": float(np.sum(utilities)) if len(utilities) else 0.0,
        "utility_positive_fraction": (
            float(np.mean(utilities > 0.0)) if len(utilities) else 0.0
        ),
        "scrambled_utility_sum": (
            float(np.sum(scrambled)) if len(scrambled) else 0.0
        ),
        "identity_advantage_mean": (
            float(np.mean(utilities - scrambled)) if len(utilities) else 0.0
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        scan_bytes=args.scan_bytes,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
