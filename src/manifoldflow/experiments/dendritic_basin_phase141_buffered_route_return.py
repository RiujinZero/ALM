"""Apply a transient recalled relationship when the next cortical word opens."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
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
from manifoldflow.experiments.dendritic_basin_phase140_cortical_route_return import (
    _route_overlay_distribution,
    _scrambled_items,
)


@dataclass
class _PendingRelation:
    natural: tuple[tuple[int, int], ...]
    scrambled: tuple[tuple[int, int], ...]
    captured_at: int
    ready: bool = False
    used: bool = False


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
    support: float,
    maximum_buffer_bytes: int,
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
    pending: _PendingRelation | None = None
    captures = 0
    scored = []
    for position in range(development_bytes, required):
        baseline = arm.learner.predict_next_distribution()
        byte_id = stream[position]
        if pending is not None:
            age = position - pending.captured_at
            if age > maximum_buffer_bytes:
                pending = None
            elif pending.ready and arm.learner.current_word:
                natural = _route_overlay_distribution(
                    arm.learner,
                    pending.natural,
                    support=support,
                )
                scrambled = _route_overlay_distribution(
                    arm.learner,
                    pending.scrambled,
                    support=support,
                )
                if natural is not None and scrambled is not None:
                    pending.used = True
                    scored.append(
                        {
                            "position": position,
                            "age": age,
                            "prefix_bytes": len(arm.learner.current_word),
                            "natural_utility": float(
                                np.log2(max(natural[byte_id], 1e-300))
                                - np.log2(max(baseline[byte_id], 1e-300))
                            ),
                            "scrambled_utility": float(
                                np.log2(max(scrambled[byte_id], 1e-300))
                                - np.log2(max(baseline[byte_id], 1e-300))
                            ),
                            "natural_variation": 0.5
                            * float(np.sum(np.abs(natural - baseline))),
                        }
                    )
        expression_distribution = _predict(arm, live=True)
        fresh_items = tuple(
            sorted((arm.pending_relational_receptors or {}).keys())
        )
        if fresh_items and pending is None:
            fresh_scramble = _scrambled_items(arm.learner, fresh_items)
            if fresh_scramble is not None:
                pending = _PendingRelation(
                    fresh_items,
                    fresh_scramble,
                    position,
                    ready=not bool(arm.learner.current_word),
                )
                captures += 1
        was_in_word = bool(arm.learner.current_word)
        _advance_generated_state(
            arm,
            byte_id,
            float(expression_distribution[byte_id]),
        )
        is_letter = 65 <= byte_id <= 90 or 97 <= byte_id <= 122
        if pending is not None and was_in_word and not is_letter:
            if pending.ready and pending.used:
                pending = None
            else:
                pending.ready = True
    aggregate = {}
    for key in ("natural_utility", "scrambled_utility", "natural_variation"):
        values = np.asarray([row[key] for row in scored], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "sum": float(np.sum(values)) if len(values) else 0.0,
            "positive_fraction": (
                float(np.mean(values > 0.0)) if len(values) else 0.0
            ),
        }
    differences = np.asarray(
        [row["natural_utility"] - row["scrambled_utility"] for row in scored],
        dtype=np.float64,
    )
    aggregate["identity_advantage"] = {
        "mean": float(np.mean(differences)) if len(differences) else 0.0,
        "positive_fraction": (
            float(np.mean(differences > 0.0)) if len(differences) else 0.0
        ),
    }
    return {
        "task": "phase141_buffered_route_return",
        "state": str(state),
        "tiny_path": str(tiny_path),
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "support": support,
        "maximum_buffer_bytes": maximum_buffer_bytes,
        "captures": captures,
        "scored_bytes": len(scored),
        "aggregate": aggregate,
        "rows": scored,
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("tiny", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--development-bytes", type=int, default=1024)
    parser.add_argument("--scan-bytes", type=int, default=512)
    parser.add_argument("--support", type=float, default=0.5)
    parser.add_argument("--maximum-buffer-bytes", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        scan_bytes=args.scan_bytes,
        support=args.support,
        maximum_buffer_bytes=args.maximum_buffer_bytes,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
