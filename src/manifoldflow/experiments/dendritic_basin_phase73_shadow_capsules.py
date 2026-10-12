"""Calibrate boundary-capsule retrieval with exact shadow read utility."""

from __future__ import annotations

import argparse
import json
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.event_capsule import (
    BoundaryEventCapsuleMemory,
    CapsuleRead,
)
from manifoldflow.dendritic_basin.transient_binding import BindingRecall
from manifoldflow.experiments.dendritic_basin_generate import _read_checkpoint
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_VALID_PATH,
)


class _NoopBindingIndex:
    """Keep the learned query-role organ active without legacy recall."""

    live_cells = 0

    def bind(self, *_args: object, **_kwargs: object) -> int:
        return 0

    def recall(self, *_args: object, **_kwargs: object) -> BindingRecall:
        return BindingRecall(None, None, 0.0, 0)

    def reset_episode(self) -> None:
        return None


def _temporary_distribution(learner: object, read: CapsuleRead) -> np.ndarray:
    missing = object()
    previous: dict[int, object] = {
        assembly: learner.temporal_last_seen.get(assembly, missing)
        for assembly, _ in read.items
    }
    recalled = set(getattr(learner, "temporal_recalled_sources", set()))
    if not hasattr(learner, "temporal_recalled_sources"):
        learner.temporal_recalled_sources = set()
    for assembly, receptor in read.items:
        if 0 <= assembly < len(learner.word_forms):
            learner.temporal_last_seen[assembly] = learner.temporal_clock - receptor + 1
            learner.temporal_recalled_sources.add(assembly)
    learner._goal_route_projection_cache = None
    distribution = learner.predict_next_distribution().copy()
    for assembly, value in previous.items():
        if value is missing:
            learner.temporal_last_seen.pop(assembly, None)
        else:
            learner.temporal_last_seen[assembly] = value
    learner.temporal_recalled_sources = recalled
    learner._goal_route_projection_cache = None
    return distribution


def _calibration_bins(
    events: list[dict[str, float]], *, field: str = "permission"
) -> list[dict[str, float | int]]:
    if not events:
        return []
    order = np.argsort([event[field] for event in events])
    bins: list[dict[str, float | int]] = []
    for indices in np.array_split(order, min(5, len(order))):
        selected = [events[int(index)] for index in indices]
        utilities = np.asarray(
            [event["utility_bits"] for event in selected], dtype=np.float64
        )
        bins.append(
            {
                "reads": len(selected),
                f"minimum_{field}": min(event[field] for event in selected),
                f"maximum_{field}": max(event[field] for event in selected),
                f"mean_{field}": float(np.mean([event[field] for event in selected])),
                "mean_utility_bits": float(np.mean(utilities)),
                "positive_fraction": float(np.mean(utilities > 0.0)),
            }
        )
    return bins


def _content_match(learner: object, item: tuple[int, int]) -> float:
    assembly, receptor = item
    row = learner.route_counts.get((receptor, assembly))
    if not row:
        return 0.0
    prefix = bytes(learner.current_word)
    if not prefix:
        return 0.0
    matched = sum(
        observations
        for target, observations in row.items()
        if 0 <= target < len(learner.word_forms)
        and learner.word_forms[target].startswith(prefix)
    )
    return matched / max(1, learner.route_totals[(receptor, assembly)])


def run_shadow_probe(
    checkpoint: Path,
    eval_path: Path,
    *,
    eval_bytes: int = 100_000,
    max_shadow_reads: int = 5_000,
    permission_context: str = "query_receptor",
    calibration_bytes: int | None = None,
    shadow_candidates: int = 1,
) -> dict[str, object]:
    learner = _read_checkpoint(checkpoint)
    learner.reset_episode()
    learner.binding_index = _NoopBindingIndex()
    if not hasattr(learner, "temporal_recalled_sources"):
        learner.temporal_recalled_sources = set()
    memory_context = (
        "query_receptor"
        if permission_context
        in {
            "role_receptor",
            "phase_receptor",
            "binding_branch",
            "trace_surprise",
            "predictive_surprise",
            "entropy_restoration",
        }
        else permission_context
    )
    memory = BoundaryEventCapsuleMemory(permission_context=memory_context)
    stream = np.fromfile(eval_path, dtype=np.uint8, count=eval_bytes)
    raw_window = bytearray()
    previous_routes: set[tuple[int, int]] = set()
    previous_surprise = 0.0
    bits = 0.0
    scored = 0
    events: list[dict[str, float]] = []
    calibration_bytes = (
        len(stream) // 2 if calibration_bytes is None else calibration_bytes
    )
    calibration_reads = 0
    calibration_utility = 0.0
    for position, byte_id_value in enumerate(stream):
        byte_id = int(byte_id_value)
        evaluating = position >= calibration_bytes
        off = learner.predict_next_distribution()
        off_probability = float(off[byte_id])
        top_ids = np.argpartition(off, -8)[-8:]
        prediction_fingerprint = tuple(
            (int(byte), float(off[byte]))
            for byte in top_ids[np.argsort(off[top_ids])[::-1]]
        )
        query = int.from_bytes(raw_window, "big") if len(raw_window) == 3 else None
        basin_organ = getattr(learner, "predictive_goal_organ", None)
        basin = None if basin_organ is None else basin_organ.current_basin
        role_organ = getattr(learner, "binding_role_organ", None)
        permission_query = (
            0
            if role_organ is None or role_organ.current_basin is None
            else int(role_organ.current_basin)
        )
        if permission_context == "phase_receptor":
            permission_query = min(len(learner.current_word), 15)
        reads = (
            ()
            if query is None
            else memory.candidates(
                query,
                basin,
                permission_query=(
                    permission_query
                    if permission_context in {"role_receptor", "phase_receptor"}
                    else query
                ),
                context_items=tuple(
                    (assembly, receptor) for receptor, assembly in previous_routes
                ),
                context_fingerprint=prediction_fingerprint,
            )
        )
        options = [
            (match, candidate, item)
            for candidate in reads
            for item in candidate.items
            if item[0] not in learner.temporal_last_seen
            and learner.route_utility.get((item[1], item[0]), 0.0)
            > learner.minimum_route_utility_bits
            and (match := _content_match(learner, item)) > 0.0
        ]
        if permission_context == "binding_branch":
            branch_options = []
            for match, candidate, item in options:
                item_read = CapsuleRead(
                    candidate.capsule,
                    candidate.query,
                    item[1],
                    (item,),
                    0.0,
                    0,
                )
                value = memory.branch_value(item_read)
                evidence = memory.branch_evidence(item_read)
                branch_options.append((match, candidate, item, value, evidence))
            if evaluating:
                branch_options = [
                    option
                    for option in branch_options
                    if option[4] > 0 and option[3] > 0.0
                ]
                branch_options.sort(
                    key=lambda option: (option[3], option[0], -option[1].capsule),
                    reverse=True,
                )
            else:
                branch_options.sort(
                    key=lambda option: (
                        -option[4],
                        option[3],
                        option[0],
                        -option[1].capsule,
                    ),
                    reverse=True,
                )
            options = [option[:3] for option in branch_options]
        ranked_options = (
            options[:shadow_candidates]
            if permission_context == "binding_branch"
            else sorted(
                options,
                key=lambda option: (
                    (
                        -option[1].predictive_surprise
                        if permission_context == "predictive_surprise"
                        else -option[1].trace_surprise
                        if permission_context == "trace_surprise"
                        else option[0]
                    ),
                    option[0],
                    memory.permission(option[1].query, option[2][1]),
                    -option[1].capsule,
                ),
                reverse=True,
            )[:shadow_candidates]
        )
        if ranked_options:
            selected_match, selected_read, selected_item = ranked_options[0]
            read = CapsuleRead(
                selected_read.capsule,
                selected_read.query,
                selected_item[1],
                (selected_item,),
                memory.permission(selected_read.query, selected_item[1]),
                memory.evidence(selected_read.query, selected_item[1]),
                selected_read.trace_surprise,
                selected_read.predictive_surprise,
            )
        else:
            read = None
        probe_reads = tuple(
            (
                match,
                CapsuleRead(
                    candidate.capsule,
                    candidate.query,
                    item[1],
                    (item,),
                    memory.permission(candidate.query, item[1]),
                    memory.evidence(candidate.query, item[1]),
                    candidate.trace_surprise,
                    candidate.predictive_surprise,
                ),
            )
            for match, candidate, item in ranked_options
        )
        probe_distributions: dict[CapsuleRead, np.ndarray] = {}
        restorations: dict[CapsuleRead, float] = {}
        if permission_context == "entropy_restoration":
            off_entropy = -float(np.sum(off * np.log2(np.maximum(off, 1e-300))))
            for _, item_read in probe_reads:
                on = _temporary_distribution(learner, item_read)
                probe_distributions[item_read] = on
                restorations[item_read] = off_entropy + float(
                    np.sum(on * np.log2(np.maximum(on, 1e-300)))
                )
            probe_reads = tuple(
                sorted(
                    probe_reads,
                    key=lambda pair: restorations[pair[1]],
                    reverse=True,
                )
            )
            read = probe_reads[0][1] if probe_reads else None
        if read is not None and (not evaluating or len(events) < max_shadow_reads):
            for rank, (match, item_read) in enumerate(probe_reads):
                if evaluating and len(events) >= max_shadow_reads:
                    break
                on = probe_distributions.get(item_read)
                if on is None:
                    on = _temporary_distribution(learner, item_read)
                utility = log2(
                    max(float(on[byte_id]), 1e-300) / max(off_probability, 1e-300)
                )
                if evaluating:
                    branch_value = memory.branch_value(item_read)
                    events.append(
                        {
                            "permission": (
                                branch_value
                                if permission_context == "binding_branch"
                                else item_read.permission
                            ),
                            "content_match": match,
                            "trace_surprise": item_read.trace_surprise,
                            "predictive_surprise": item_read.predictive_surprise,
                            "entropy_restoration": restorations.get(item_read, 0.0),
                            "evidence": float(item_read.evidence),
                            "utility_bits": utility,
                            "position": float(position),
                            "rank": float(rank),
                            "assembly": float(item_read.items[0][0]),
                            "receptor": float(item_read.items[0][1]),
                        }
                    )
                    memory.touch(item_read)
                else:
                    if permission_context == "binding_branch":
                        memory.credit_branch(item_read, utility)
                    else:
                        memory.credit(item_read, utility)
                    calibration_reads += 1
                    calibration_utility += utility
        learner._observe_probability_only(byte_id, learn=False)
        bits += -log2(max(off_probability, 1e-300))
        scored += 1
        surprise = -log2(max(off_probability, 1e-300))
        routes = set(learner._active_route_keys())
        union = len(routes | previous_routes)
        turnover = 0.0 if not union else 1.0 - len(routes & previous_routes) / union
        disagreement = learner._temporal_disagreement(tuple(routes))
        boundary_signal = abs(surprise - previous_surprise) + turnover + disagreement
        raw_window.append(byte_id)
        if len(raw_window) > 3:
            del raw_window[0]
        current_query = (
            int.from_bytes(raw_window, "big") if len(raw_window) == 3 else None
        )
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
            query=current_query,
            active_items=active_items,
            basin=basin,
            prediction_fingerprint=prediction_fingerprint,
        )
        previous_routes = routes
        previous_surprise = surprise
    selected_events = [event for event in events if event["rank"] == 0.0]
    utilities = np.asarray(
        [event["utility_bits"] for event in selected_events], dtype=np.float64
    )
    positions = {int(event["position"]) for event in events}
    unique_interventions = {
        (
            int(event["position"]),
            int(event["assembly"]),
            int(event["receptor"]),
        )
        for event in events
    }
    oracle_utilities = np.asarray(
        [
            max(
                event["utility_bits"]
                for event in events
                if int(event["position"]) == position
            )
            for position in positions
        ],
        dtype=np.float64,
    )
    restorative_events = [
        event for event in selected_events if event["entropy_restoration"] > 0.0
    ]
    restorative_utilities = np.asarray(
        [event["utility_bits"] for event in restorative_events], dtype=np.float64
    )
    return {
        "task": "phase73_boundary_capsule_shadow_permission",
        "checkpoint": str(checkpoint),
        "eval_bytes": len(stream),
        "baseline_bpb": bits / max(1, scored),
        "permission_context": permission_context,
        "shadow_candidates": shadow_candidates,
        "calibration_bytes": calibration_bytes,
        "calibration_shadow_reads": calibration_reads,
        "calibration_shadow_utility_bits": calibration_utility,
        "capsules_captured": memory.captures,
        "live_capsules": len(memory.capsules),
        "capsule_replacements": memory.replacements,
        "shadow_reads": len(events),
        "shadow_decisions": len(selected_events),
        "unique_interventions": len(unique_interventions),
        "total_shadow_utility_bits": float(utilities.sum()),
        "positive_read_fraction": (
            float(np.mean(utilities > 0.0)) if len(utilities) else 0.0
        ),
        "oracle_candidate_utility_bits": float(oracle_utilities.sum()),
        "oracle_positive_fraction": (
            float(np.mean(oracle_utilities > 0.0)) if len(oracle_utilities) else 0.0
        ),
        "restorative_reads": len(restorative_events),
        "restorative_utility_bits": float(restorative_utilities.sum()),
        "restorative_positive_fraction": (
            float(np.mean(restorative_utilities > 0.0))
            if len(restorative_utilities)
            else 0.0
        ),
        "permission_calibration": _calibration_bins(selected_events),
        "content_match_calibration": _calibration_bins(
            selected_events, field="content_match"
        ),
        "trace_surprise_calibration": _calibration_bins(
            selected_events, field="trace_surprise"
        ),
        "predictive_surprise_calibration": _calibration_bins(
            selected_events, field="predictive_surprise"
        ),
        "entropy_restoration_calibration": _calibration_bins(
            selected_events, field="entropy_restoration"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--max-shadow-reads", type=int, default=5_000)
    parser.add_argument("--shadow-candidates", type=int, default=1)
    parser.add_argument("--calibration-bytes", type=int)
    parser.add_argument(
        "--permission-context",
        choices=(
            "query_receptor",
            "role_receptor",
            "phase_receptor",
            "binding_branch",
            "trace_surprise",
            "predictive_surprise",
            "entropy_restoration",
            "receptor",
        ),
        default="query_receptor",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_shadow_probe(
        args.checkpoint,
        args.eval_path,
        eval_bytes=args.eval_bytes,
        max_shadow_reads=args.max_shadow_reads,
        permission_context=args.permission_context,
        calibration_bytes=args.calibration_bytes,
        shadow_candidates=args.shadow_candidates,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
