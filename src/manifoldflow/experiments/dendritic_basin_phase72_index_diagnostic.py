"""Distinguish missing transient binding from autonomous basin-state drift.

The assay never changes durable anatomy.  It first corrupts only distant prefix
evidence while preserving the most recent bytes, then asks whether replaying the
basin state recorded from the intact story restores likelihood.  It separately
drives a free-running cortex with the teacher-forced basin trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
    _read_checkpoint,
    _sample_byte,
)
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_VALID_PATH,
)

DEFAULT_BOUNDARY = b"<|endoftext|>\n"


def _stories(raw: bytes, boundary: bytes) -> list[bytes]:
    return [story for story in raw.split(boundary) if story]


def scramble_distant_prefix(
    prefix: bytes, *, preserved_recent_bytes: int, seed: int
) -> bytes:
    """Destroy distant order while exactly preserving the recent suffix."""
    if preserved_recent_bytes < 0:
        raise ValueError("preserved_recent_bytes must be non-negative")
    split = max(0, len(prefix) - preserved_recent_bytes)
    distant = np.frombuffer(prefix[:split], dtype=np.uint8).copy()
    np.random.default_rng(seed).shuffle(distant)
    return distant.tobytes() + prefix[split:]


def _current_basin(learner: object) -> int | None:
    organ = getattr(learner, "predictive_goal_organ", None)
    return None if organ is None else organ.current_basin


def _score_continuation(
    learner: object,
    prefix: bytes,
    continuation: bytes,
    *,
    basin_trace: list[int | None] | None = None,
) -> tuple[np.ndarray, list[int | None]]:
    learner.reset_episode()
    learner.observe_chunk_probabilities(
        np.frombuffer(prefix, dtype=np.uint8), learn=False
    )
    probability = np.empty(len(continuation), dtype=np.float64)
    observed_basins: list[int | None] = []
    organ = getattr(learner, "predictive_goal_organ", None)
    for index, byte_id in enumerate(continuation):
        if basin_trace is not None and organ is not None:
            organ.current_basin = basin_trace[index]
        observed_basins.append(_current_basin(learner))
        observed = learner._observe_probability_only(int(byte_id), learn=False)
        probability[index] = np.nan if observed is None else observed
    return probability, observed_basins


def _generate_continuation(
    learner: object,
    prefix: bytes,
    length: int,
    *,
    seed: int,
    temperature: float,
    top_k: int,
    top_p: float,
    basin_trace: list[int | None] | None = None,
) -> tuple[bytes, list[int | None]]:
    learner.reset_episode()
    learner.observe_chunk_probabilities(
        np.frombuffer(prefix, dtype=np.uint8), learn=False
    )
    rng = np.random.default_rng(seed)
    result = bytearray()
    observed_basins: list[int | None] = []
    organ = getattr(learner, "predictive_goal_organ", None)
    for index in range(length):
        if basin_trace is not None and organ is not None:
            organ.current_basin = basin_trace[index]
        observed_basins.append(_current_basin(learner))
        byte_id = _sample_byte(
            learner.predict_next_distribution(),
            rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        result.append(byte_id)
        learner.advance_observation(byte_id, learn=False)
    return bytes(result), observed_basins


def _bpb(probability: np.ndarray) -> float:
    finite = probability[np.isfinite(probability)]
    return float(np.mean(-np.log2(np.maximum(finite, 1e-300))))


def _matching_prefix(left: bytes, right: bytes) -> int:
    for index, (left_byte, right_byte) in enumerate(zip(left, right, strict=False)):
        if left_byte != right_byte:
            return index
    return min(len(left), len(right))


def _basin_summary(traces: list[list[int | None]]) -> dict[str, object]:
    flattened = [value for trace in traces for value in trace]
    active = [value for value in flattened if value is not None]
    return {
        "observations": len(flattened),
        "active_fraction": len(active) / max(1, len(flattened)),
        "distinct_active_basins": sorted(set(active)),
    }


def run_diagnostic(
    checkpoint: Path,
    eval_path: Path,
    *,
    eval_bytes: int = 100_000,
    prefix_bytes: int = 512,
    preserved_recent_bytes: int = 128,
    continuation_bytes: int = 256,
    max_stories: int = 16,
    seed: int = 72,
    temperature: float = 0.8,
    top_k: int = 32,
    top_p: float = 0.9,
) -> dict[str, object]:
    if prefix_bytes <= preserved_recent_bytes:
        raise ValueError("prefix must include evidence older than the preserved suffix")
    if continuation_bytes < 1 or max_stories < 1:
        raise ValueError("continuation_bytes and max_stories must be positive")
    raw = eval_path.read_bytes()[:eval_bytes]
    selected = [
        story
        for story in _stories(raw, DEFAULT_BOUNDARY)
        if len(story) >= prefix_bytes + continuation_bytes
    ][:max_stories]
    if not selected:
        raise ValueError(
            "evaluation slice contains no sufficiently long complete story"
        )

    normal = _read_checkpoint(checkpoint)
    corrupted = _read_checkpoint(checkpoint)
    clamped = _read_checkpoint(checkpoint)
    free = _read_checkpoint(checkpoint)
    teacher_clamped = _read_checkpoint(checkpoint)
    boundary = getattr(normal, "story_boundary", DEFAULT_BOUNDARY)
    records: list[dict[str, object]] = []
    normal_probabilities: list[np.ndarray] = []
    corrupted_probabilities: list[np.ndarray] = []
    clamped_probabilities: list[np.ndarray] = []
    teacher_traces: list[list[int | None]] = []
    free_traces: list[list[int | None]] = []
    clamped_generation_traces: list[list[int | None]] = []

    for story_index, story in enumerate(selected):
        prefix = story[:prefix_bytes]
        continuation = story[prefix_bytes : prefix_bytes + continuation_bytes]
        corrupt_prefix = scramble_distant_prefix(
            prefix,
            preserved_recent_bytes=preserved_recent_bytes,
            seed=seed + story_index,
        )
        normal_probability, teacher_trace = _score_continuation(
            normal, prefix, continuation
        )
        corrupted_probability, _ = _score_continuation(
            corrupted, corrupt_prefix, continuation
        )
        clamped_probability, _ = _score_continuation(
            clamped,
            corrupt_prefix,
            continuation,
            basin_trace=teacher_trace,
        )
        generated_free, free_trace = _generate_continuation(
            free,
            prefix,
            continuation_bytes,
            seed=seed + story_index,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated_clamped, clamped_trace = _generate_continuation(
            teacher_clamped,
            prefix,
            continuation_bytes,
            seed=seed + story_index,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            basin_trace=teacher_trace,
        )
        normal_probabilities.append(normal_probability)
        corrupted_probabilities.append(corrupted_probability)
        clamped_probabilities.append(clamped_probability)
        teacher_traces.append(teacher_trace)
        free_traces.append(free_trace)
        clamped_generation_traces.append(clamped_trace)
        records.append(
            {
                "story_index": story_index,
                "normal_bpb": _bpb(normal_probability),
                "corrupted_bpb": _bpb(corrupted_probability),
                "corrupted_teacher_basin_bpb": _bpb(clamped_probability),
                "teacher_basin_distinct": sorted(
                    {value for value in teacher_trace if value is not None}
                ),
                "free_vs_clamped_identical": generated_free == generated_clamped,
                "free_vs_clamped_first_divergence": (
                    None
                    if generated_free == generated_clamped
                    else _matching_prefix(generated_free, generated_clamped)
                ),
                "free_reference_matching_prefix": _matching_prefix(
                    generated_free, continuation
                ),
                "clamped_reference_matching_prefix": _matching_prefix(
                    generated_clamped, continuation
                ),
                "free_reference_byte_accuracy": float(
                    np.mean(np.frombuffer(generated_free, dtype=np.uint8)
                    == np.frombuffer(continuation, dtype=np.uint8))
                ),
                "clamped_reference_byte_accuracy": float(
                    np.mean(np.frombuffer(generated_clamped, dtype=np.uint8)
                    == np.frombuffer(continuation, dtype=np.uint8))
                ),
                "reference_text": continuation.decode(
                    "utf-8", errors="backslashreplace"
                ),
                "free_text": generated_free.decode("utf-8", errors="backslashreplace"),
                "clamped_text": generated_clamped.decode(
                    "utf-8", errors="backslashreplace"
                ),
                "free_diagnostics": _generation_diagnostics(generated_free, boundary),
                "clamped_diagnostics": _generation_diagnostics(
                    generated_clamped, boundary
                ),
            }
        )

    normal_all = np.concatenate(normal_probabilities)
    corrupted_all = np.concatenate(corrupted_probabilities)
    clamped_all = np.concatenate(clamped_probabilities)
    free_accuracy = [item["free_reference_byte_accuracy"] for item in records]
    clamped_accuracy = [
        item["clamped_reference_byte_accuracy"] for item in records
    ]
    changed = [not item["free_vs_clamped_identical"] for item in records]
    return {
        "task": "phase72_binding_and_closed_loop_diagnostic",
        "checkpoint": str(checkpoint),
        "eval_path": str(eval_path),
        "eval_bytes": min(eval_bytes, len(raw)),
        "story_count": len(records),
        "prefix_bytes": prefix_bytes,
        "preserved_recent_bytes": preserved_recent_bytes,
        "continuation_bytes": continuation_bytes,
        "teacher_forced": {
            "normal_bpb": _bpb(normal_all),
            "corrupted_bpb": _bpb(corrupted_all),
            "corrupted_teacher_basin_bpb": _bpb(clamped_all),
            "corruption_cost_bits": float(
                np.sum(np.log2(np.maximum(normal_all, 1e-300))
                - np.log2(np.maximum(corrupted_all, 1e-300)))
            ),
            "bits_recovered_by_teacher_basin": float(
                np.sum(np.log2(np.maximum(clamped_all, 1e-300))
                - np.log2(np.maximum(corrupted_all, 1e-300)))
            ),
            "teacher_basin_state": _basin_summary(teacher_traces),
        },
        "free_running": {
            "changed_story_fraction": float(np.mean(changed)),
            "mean_free_reference_byte_accuracy": float(np.mean(free_accuracy)),
            "mean_clamped_reference_byte_accuracy": float(
                np.mean(clamped_accuracy)
            ),
            "teacher_basin_state": _basin_summary(teacher_traces),
            "free_basin_state": _basin_summary(free_traces),
            "clamped_basin_state": _basin_summary(clamped_generation_traces),
        },
        "stories": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--prefix-bytes", type=int, default=512)
    parser.add_argument("--preserved-recent-bytes", type=int, default=128)
    parser.add_argument("--continuation-bytes", type=int, default=256)
    parser.add_argument("--max-stories", type=int, default=16)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_diagnostic(
        args.checkpoint,
        args.eval_path,
        eval_bytes=args.eval_bytes,
        prefix_bytes=args.prefix_bytes,
        preserved_recent_bytes=args.preserved_recent_bytes,
        continuation_bytes=args.continuation_bytes,
        max_stories=args.max_stories,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
