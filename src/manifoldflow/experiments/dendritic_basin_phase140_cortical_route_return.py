"""Return recalled identities through the cortex's existing temporal routes."""

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

SUPPORT_LEVELS = (0.5, 1.0, 2.0, 4.0)


def _youngest_age_for_receptor(learner: object, receptor: int) -> int | None:
    # Capsule item receptor zero denotes the just-observed event.  The cortex's
    # first temporal receptor sees that same source on the following event.
    if receptor == 0:
        return 1
    previous = 0
    for bound in learner.temporal_bins:
        if bound == receptor:
            return previous + 1
        previous = bound
    return None


def _route_overlay_distribution(
    learner: object,
    items: tuple[tuple[int, int], ...],
    *,
    support: float | Mapping[tuple[int, int], float],
) -> np.ndarray | None:
    """Briefly restore sources on their native temporal receptor wires."""
    if not hasattr(learner, "temporal_last_seen"):
        return None
    original = learner.temporal_last_seen.copy()
    original_utility = learner.route_utility.copy()
    original_candidates = learner.active_word_candidates
    try:
        for assembly, receptor in items:
            item_support = (
                float(support.get((assembly, receptor), 0.0))
                if isinstance(support, Mapping)
                else float(support)
            )
            age = _youngest_age_for_receptor(learner, receptor)
            if age is None or item_support <= 0.0:
                return None
            learner.temporal_last_seen[assembly] = learner.temporal_clock - age + 1
            route_receptor = learner.temporal_bins[0] if receptor == 0 else receptor
            key = route_receptor, assembly
            if key in learner.route_counts:
                learner.route_utility[key] = (
                    learner.route_utility.get(key, 0.0) + item_support
                )
        # Reopen the current prefix on the temporarily supported route.  This
        # is an oracle intervention; it does not rewrite the sensory history.
        learner.active_word_candidates = None
        return learner.predict_next_distribution().copy()
    finally:
        learner.temporal_last_seen = original
        learner.route_utility = original_utility
        learner.active_word_candidates = original_candidates


def _scrambled_items(
    learner: object,
    items: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...] | None:
    excluded = {assembly for assembly, _receptor in items}
    swapped = []
    for assembly, receptor in items:
        route_receptor = learner.temporal_bins[0] if receptor == 0 else receptor
        candidates = sorted(
            source
            for candidate_receptor, source in learner.route_counts
            if candidate_receptor == route_receptor
            and source != assembly
            and source not in excluded
        )
        if not candidates:
            return None
        swapped.append((candidates[assembly % len(candidates)], receptor))
    return tuple(swapped)


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
    for position in range(development_bytes, required):
        baseline = arm.learner.predict_next_distribution()
        expression_distribution = _predict(arm, live=True)
        items = tuple(sorted((arm.pending_relational_receptors or {}).keys()))
        scrambled_items = _scrambled_items(arm.learner, items) if items else None
        byte_id = stream[position]
        support_rows = {}
        for support in SUPPORT_LEVELS:
            natural = (
                _route_overlay_distribution(arm.learner, items, support=support)
                if items
                else None
            )
            scrambled = (
                _route_overlay_distribution(
                    arm.learner,
                    scrambled_items,
                    support=support,
                )
                if scrambled_items
                else None
            )
            if natural is None or scrambled is None:
                continue
            support_rows[str(support)] = {
                "route_utility": float(
                    np.log2(max(natural[byte_id], 1e-300))
                    - np.log2(max(baseline[byte_id], 1e-300))
                ),
                "scrambled_route_utility": float(
                    np.log2(max(scrambled[byte_id], 1e-300))
                    - np.log2(max(baseline[byte_id], 1e-300))
                ),
                "route_variation": 0.5
                * float(np.sum(np.abs(natural - baseline))),
            }
        if support_rows:
            rows.append(
                {
                    "position": position,
                    "items": len(items),
                    "expression_utility": float(
                        np.log2(max(expression_distribution[byte_id], 1e-300))
                        - np.log2(max(baseline[byte_id], 1e-300))
                    ),
                    "support": support_rows,
                }
            )
        _advance_generated_state(
            arm,
            byte_id,
            float(expression_distribution[byte_id]),
        )
    aggregate = {}
    expression_values = np.asarray(
        [row["expression_utility"] for row in rows], dtype=np.float64
    )
    aggregate["expression_utility"] = {
        "mean": float(np.mean(expression_values)) if len(expression_values) else 0.0,
        "positive_fraction": (
            float(np.mean(expression_values > 0.0)) if len(expression_values) else 0.0
        ),
    }
    aggregate["support"] = {}
    for support in SUPPORT_LEVELS:
        label = str(support)
        aggregate["support"][label] = {}
        for key in (
            "route_utility",
            "scrambled_route_utility",
            "route_variation",
        ):
            values = np.asarray(
                [row["support"][label][key] for row in rows], dtype=np.float64
            )
            aggregate["support"][label][key] = {
                "mean": float(np.mean(values)) if len(values) else 0.0,
                "positive_fraction": (
                    float(np.mean(values > 0.0)) if len(values) else 0.0
                ),
            }
    return {
        "task": "phase140_cortical_route_return",
        "state": str(state),
        "tiny_path": str(tiny_path),
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
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
    parser.add_argument("--scan-bytes", type=int, default=256)
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
