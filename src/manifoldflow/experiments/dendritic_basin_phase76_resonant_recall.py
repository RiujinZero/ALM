"""Measure keyless sparse attractor completion over boundary event capsules."""

from __future__ import annotations

import argparse
import json
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.event_capsule import BoundaryEventCapsuleMemory
from manifoldflow.dendritic_basin.resonant_capsule import (
    SparseResonantCapsuleIndex,
)
from manifoldflow.experiments.dendritic_basin_generate import _read_checkpoint
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_VALID_PATH,
)
from manifoldflow.experiments.dendritic_basin_phase73_shadow_capsules import (
    _content_match,
    _NoopBindingIndex,
    _temporary_distribution,
)
from manifoldflow.experiments.dendritic_basin_phase74_risk_permission import (
    _calibration_bins,
    _utility_summary,
)


def run_probe(
    checkpoint: Path,
    eval_path: Path,
    *,
    eval_bytes: int = 100_000,
    eval_offset: int = 0,
    inverse_temperature: float = 8.0,
    completion_threshold: float = 0.02,
    capture_recency_gain: float = 1.0,
    recall_recency_gain: float = 1.0,
) -> dict[str, object]:
    learner = _read_checkpoint(checkpoint)
    learner.reset_episode()
    learner.binding_index = _NoopBindingIndex()
    if not hasattr(learner, "temporal_recalled_sources"):
        learner.temporal_recalled_sources = set()
    memory = BoundaryEventCapsuleMemory(permission_context="query_receptor")
    index = SparseResonantCapsuleIndex(
        inverse_temperature=inverse_temperature,
        completion_threshold=completion_threshold,
        capture_recency_gain=capture_recency_gain,
        recall_recency_gain=recall_recency_gain,
    )
    with eval_path.open("rb") as handle:
        handle.seek(eval_offset)
        stream = np.frombuffer(handle.read(eval_bytes), dtype=np.uint8)
    raw_window = bytearray()
    previous_routes: set[tuple[int, int]] = set()
    previous_surprise = 0.0
    baseline_bits = 0.0
    events: list[dict[str, float]] = []

    for position, value in enumerate(stream):
        byte_id = int(value)
        off = learner.predict_next_distribution()
        off_probability = float(off[byte_id])
        content_gain = {
            item: match
            for capsule in memory.capsules
            for item in capsule.items
            if item[0] not in learner.temporal_last_seen
            and learner.route_utility.get((item[1], item[0]), 0.0)
            > learner.minimum_route_utility_bits
            and (match := _content_match(learner, item)) > 0.0
        }
        completion = index.resonate(
            memory,
            tuple((assembly, receptor) for receptor, assembly in previous_routes),
            content_gain=content_gain,
        )
        if completion is not None:
            items = tuple(
                item
                for item in completion.read.items
                if item[0] not in learner.temporal_last_seen
                and learner.route_utility.get((item[1], item[0]), 0.0)
                > learner.minimum_route_utility_bits
                and content_gain.get(item, 0.0) > 0.0
            )
            if items:
                read = completion.read.__class__(
                    completion.read.capsule,
                    completion.read.query,
                    items[0][1],
                    items,
                    completion.read.permission,
                    completion.read.evidence,
                )
                on = _temporary_distribution(learner, read)
                utility = log2(
                    max(float(on[byte_id]), 1e-300)
                    / max(off_probability, 1e-300)
                )
                disruption = 0.5 * float(np.sum(np.abs(on - off)))
                winner_changed = float(np.argmax(on) != np.argmax(off))
                events.append(
                    {
                        "utility_bits": utility,
                        "permission_score": completion.confidence,
                        "confidence": completion.confidence,
                        "maximum_overlap": completion.maximum_overlap,
                        "posterior_peak": completion.posterior_peak,
                        "posterior_entropy": completion.posterior_entropy,
                        "disruption": disruption,
                        "winner_changed": winner_changed,
                        "position": float(position),
                    }
                )
                memory.touch(read)

        learner._observe_probability_only(byte_id, learn=False)
        baseline_bits += -log2(max(off_probability, 1e-300))
        surprise = -log2(max(off_probability, 1e-300))
        routes = set(learner._active_route_keys())
        union = len(routes | previous_routes)
        turnover = 0.0 if not union else 1.0 - len(routes & previous_routes) / union
        disagreement = learner._temporal_disagreement(tuple(routes))
        boundary_signal = abs(surprise - previous_surprise) + turnover + disagreement
        raw_window.append(byte_id)
        if len(raw_window) > 3:
            del raw_window[0]
        query = int.from_bytes(raw_window, "big") if len(raw_window) == 3 else None
        organ = getattr(learner, "predictive_goal_organ", None)
        basin = None if organ is None else organ.current_basin
        active_items = tuple(
            (
                receptor,
                assembly,
                len(learner.word_forms[assembly])
                / max(float(learner.motif_activity[assembly]), 1e-12) ** 0.5,
            )
            for receptor, assembly in routes
            if 0 <= assembly < len(learner.word_forms)
            and assembly not in learner.temporal_recalled_sources
        )
        memory.observe_boundary(
            boundary_signal,
            query=query,
            active_items=active_items,
            basin=basin,
        )
        previous_routes = routes
        previous_surprise = surprise

    catastrophic_threshold = 0.1
    winner_preserving = [event for event in events if not event["winner_changed"]]
    winner_changing = [event for event in events if event["winner_changed"]]
    return {
        "task": "phase76_keyless_resonant_recall",
        "checkpoint": str(checkpoint),
        "eval_path": str(eval_path),
        "eval_bytes": len(stream),
        "eval_offset": eval_offset,
        "inverse_temperature": inverse_temperature,
        "completion_threshold": completion_threshold,
        "capture_recency_gain": capture_recency_gain,
        "recall_recency_gain": recall_recency_gain,
        "capsules": len(memory.capsules),
        "baseline_bpb": baseline_bits / max(1, len(stream)),
        "completions": len(events),
        "completion_summary": _utility_summary(
            events, catastrophic_threshold=catastrophic_threshold
        ),
        "winner_preserving_summary": _utility_summary(
            winner_preserving, catastrophic_threshold=catastrophic_threshold
        ),
        "winner_changing_summary": _utility_summary(
            winner_changing, catastrophic_threshold=catastrophic_threshold
        ),
        "confidence_calibration": _calibration_bins(
            events, catastrophic_threshold=catastrophic_threshold
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--inverse-temperature", type=float, default=8.0)
    parser.add_argument("--completion-threshold", type=float, default=0.02)
    parser.add_argument("--capture-recency-gain", type=float, default=1.0)
    parser.add_argument("--recall-recency-gain", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe(
        args.checkpoint,
        args.eval_path,
        eval_bytes=args.eval_bytes,
        eval_offset=args.eval_offset,
        inverse_temperature=args.inverse_temperature,
        completion_threshold=args.completion_threshold,
        capture_recency_gain=args.capture_recency_gain,
        recall_recency_gain=args.recall_recency_gain,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
