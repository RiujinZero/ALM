"""Locate a goal circuit's causal compression gain across story horizons."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from manifoldflow.experiments.dendritic_basin_generate import _read_checkpoint
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_VALID_PATH,
)

DEFAULT_BOUNDARY = b"<|endoftext|>\n"
DEFAULT_POSITION_EDGES = (0, 64, 128, 256, 512, 1_024)


def story_positions(stream: np.ndarray, boundary: bytes) -> np.ndarray:
    """Return the causal byte position within each delimiter-separated episode."""
    raw = bytes(stream)
    positions = np.empty(len(raw), dtype=np.int64)
    position = 0
    matched = 0
    for index, byte_id in enumerate(raw):
        positions[index] = position
        position += 1
        if byte_id == boundary[matched]:
            matched += 1
            if matched == len(boundary):
                position = 0
                matched = 0
        else:
            matched = 1 if byte_id == boundary[0] else 0
    return positions


def boundary_byte_mask(stream: np.ndarray, boundary: bytes) -> np.ndarray:
    """Mark bytes that belong to a complete episode delimiter."""
    raw = bytes(stream)
    selected = np.zeros(len(raw), dtype=np.bool_)
    start = raw.find(boundary)
    while start >= 0:
        selected[start : start + len(boundary)] = True
        start = raw.find(boundary, start + len(boundary))
    return selected


def observe_with_episode_resets(
    learner: object, stream: np.ndarray, boundary: bytes
) -> np.ndarray:
    """Score delimiter-separated episodes with no transient state carryover."""
    raw = bytes(stream)
    probability = np.full(len(stream), np.nan, dtype=np.float64)
    start = 0
    while start < len(raw):
        boundary_start = raw.find(boundary, start)
        end = len(raw) if boundary_start < 0 else boundary_start + len(boundary)
        learner.reset_episode()
        probability[start:end] = learner.observe_chunk_probabilities(
            stream[start:end], learn=False
        )
        start = end
    return probability


def observe_with_regulatory_trace(
    learner: object,
    stream: np.ndarray,
    allowed_modes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score a stream while tracing locally available regulatory state."""
    learner.reset_episode()
    probability = np.full(len(stream), np.nan, dtype=np.float64)
    modes = np.full(len(stream), -1, dtype=np.int8)
    goals = np.full(len(stream), -1, dtype=np.int64)
    basins = np.full(len(stream), -1, dtype=np.int16)
    flow_onsets = np.zeros(len(stream), dtype=np.bool_)
    original_weight = float(getattr(learner, "maximum_goal_route_byte_weight", 0.0))
    for index, byte_id in enumerate(stream):
        mode = getattr(learner.goal_organ, "current_mode", None)
        goal = getattr(learner.goal_organ, "current_goal", None)
        predictive_goal = getattr(learner, "predictive_goal_organ", None)
        basin = None if predictive_goal is None else predictive_goal.current_basin
        modes[index] = -1 if mode is None else int(mode)
        goals[index] = -1 if goal is None else int(goal)
        basins[index] = -1 if basin is None else int(basin)
        before = int(getattr(learner, "goal_route_flow_uses", 0))
        if allowed_modes is not None:
            learner.maximum_goal_route_byte_weight = (
                original_weight if mode in allowed_modes else 0.0
            )
        observed = learner._observe_probability_only(int(byte_id), learn=False)
        learner.maximum_goal_route_byte_weight = original_weight
        if observed is not None:
            probability[index] = observed
        flow_onsets[index] = int(getattr(learner, "goal_route_flow_uses", 0)) > before
    return probability, modes, goals, basins, flow_onsets


def summarize_state_persistence(states: np.ndarray) -> dict[str, float | int]:
    """Describe how long one selected low-dimensional state remains active."""
    values = np.asarray(states).reshape(-1)
    active = values >= 0
    runs: list[int] = []
    start = 0
    while start < len(values):
        state = int(values[start])
        end = start + 1
        while end < len(values) and int(values[end]) == state:
            end += 1
        if state >= 0:
            runs.append(end - start)
        start = end
    lengths = np.asarray(runs, dtype=np.int64)
    return {
        "active_fraction": float(np.mean(active)) if len(values) else 0.0,
        "distinct_active_states": int(len(set(int(value) for value in values[active]))),
        "active_runs": int(len(lengths)),
        "mean_run_bytes": float(np.mean(lengths)) if len(lengths) else 0.0,
        "median_run_bytes": float(np.median(lengths)) if len(lengths) else 0.0,
        "p90_run_bytes": float(np.quantile(lengths, 0.9)) if len(lengths) else 0.0,
        "maximum_run_bytes": int(np.max(lengths)) if len(lengths) else 0,
        "runs_at_least_64_bytes": int(np.sum(lengths >= 64)),
        "runs_at_least_256_bytes": int(np.sum(lengths >= 256)),
        "runs_at_least_512_bytes": int(np.sum(lengths >= 512)),
    }


def _effect_summary(delta_bits: np.ndarray) -> dict[str, float | int]:
    finite = delta_bits[np.isfinite(delta_bits)]
    if not len(finite):
        return {
            "scored_bytes": 0,
            "mean_goal_minus_control_bits": 0.0,
            "total_bits_saved_by_goal": 0.0,
            "goal_win_fraction": 0.0,
            "goal_loss_fraction": 0.0,
            "changed_fraction": 0.0,
        }
    epsilon = 1e-15
    return {
        "scored_bytes": int(len(finite)),
        "mean_goal_minus_control_bits": float(np.mean(finite)),
        "total_bits_saved_by_goal": float(-np.sum(finite)),
        "goal_win_fraction": float(np.mean(finite < -epsilon)),
        "goal_loss_fraction": float(np.mean(finite > epsilon)),
        "changed_fraction": float(np.mean(np.abs(finite) > epsilon)),
    }


def summarize_complete_stories(
    delta_bits: np.ndarray, stream: np.ndarray, boundary: bytes
) -> dict[str, object]:
    """Measure whether aggregate savings are broad or driven by few stories."""
    if len(delta_bits) != len(stream):
        raise ValueError("delta bits and stream must have equal length")
    raw = bytes(stream)
    start = 0
    saved: list[float] = []
    lengths: list[int] = []
    compression_ratios: list[float] = []
    excerpts: list[str] = []
    story_ranges: list[tuple[int, int]] = []
    while True:
        boundary_start = raw.find(boundary, start)
        if boundary_start < 0:
            break
        finite = delta_bits[start:boundary_start]
        finite = finite[np.isfinite(finite)]
        if len(finite):
            saved.append(float(-np.sum(finite)))
            lengths.append(int(len(finite)))
            story = raw[start:boundary_start]
            compression_ratios.append(len(zlib.compress(story)) / max(1, len(story)))
            excerpts.append(story[:200].decode("utf-8", errors="backslashreplace"))
            story_ranges.append((start, boundary_start))
        start = boundary_start + len(boundary)
    if not saved:
        return {
            "complete_story_count": 0,
            "goal_win_story_fraction": 0.0,
            "goal_loss_story_fraction": 0.0,
            "mean_bits_saved_per_story": 0.0,
            "median_bits_saved_per_story": 0.0,
            "mean_story_bytes": 0.0,
            "saved_bits_story_length_correlation": 0.0,
            "saved_bits_compression_correlation": 0.0,
            "strongest_goal_wins": [],
            "strongest_goal_losses": [],
        }
    values = np.asarray(saved)
    story_lengths = np.asarray(lengths, dtype=np.float64)
    correlation = (
        0.0
        if len(values) < 2
        or float(np.std(values)) == 0.0
        or float(np.std(story_lengths)) == 0.0
        else float(np.corrcoef(values, story_lengths)[0, 1])
    )
    compression = np.asarray(compression_ratios)
    compression_correlation = (
        0.0
        if len(values) < 2
        or float(np.std(values)) == 0.0
        or float(np.std(compression)) == 0.0
        else float(np.corrcoef(values, compression)[0, 1])
    )

    def record(index: int) -> dict[str, object]:
        story_start, story_end = story_ranges[index]
        story_delta = delta_bits[story_start:story_end]
        finite_indices = np.flatnonzero(np.isfinite(story_delta))
        ordered = finite_indices[np.argsort(story_delta[finite_indices])]

        def event(offset: int) -> dict[str, float | int | str]:
            absolute = story_start + int(offset)
            context_start = max(story_start, absolute - 40)
            context_end = min(story_end, absolute + 17)
            context = raw[context_start:context_end]
            return {
                "story_byte_offset": int(offset),
                "observed_byte": int(raw[absolute]),
                "bits_saved_by_goal": float(-delta_bits[absolute]),
                "context_hex": context.hex(),
                "context_text": context.decode("utf-8", errors="backslashreplace"),
            }

        return {
            "story_index": int(index),
            "bits_saved_by_goal": float(values[index]),
            "story_bytes": int(story_lengths[index]),
            "zlib_compression_ratio": float(compression[index]),
            "excerpt": excerpts[index],
            "strongest_byte_wins": [event(int(offset)) for offset in ordered[:3]],
            "strongest_byte_losses": [
                event(int(offset)) for offset in ordered[-3:][::-1]
            ],
        }

    ranking = np.argsort(values)
    return {
        "complete_story_count": int(len(values)),
        "goal_win_story_fraction": float(np.mean(values > 0.0)),
        "goal_loss_story_fraction": float(np.mean(values < 0.0)),
        "mean_bits_saved_per_story": float(np.mean(values)),
        "median_bits_saved_per_story": float(np.median(values)),
        "p10_bits_saved_per_story": float(np.quantile(values, 0.1)),
        "p90_bits_saved_per_story": float(np.quantile(values, 0.9)),
        "mean_story_bytes": float(np.mean(story_lengths)),
        "saved_bits_story_length_correlation": correlation,
        "saved_bits_compression_correlation": compression_correlation,
        "strongest_goal_wins": [record(int(index)) for index in ranking[-5:][::-1]],
        "strongest_goal_losses": [record(int(index)) for index in ranking[:5]],
    }


def summarize_horizons(
    delta_bits: np.ndarray,
    positions: np.ndarray,
    edges: tuple[int, ...] = DEFAULT_POSITION_EDGES,
) -> dict[str, dict[str, float | int]]:
    if len(delta_bits) != len(positions):
        raise ValueError("delta bits and positions must have equal length")
    report: dict[str, dict[str, float | int]] = {}
    for lower, upper in zip(edges, edges[1:]):  # noqa: B905
        selected = (positions >= lower) & (positions < upper)
        report[f"{lower}:{upper}"] = _effect_summary(delta_bits[selected])
    selected = positions >= edges[-1]
    report[f"{edges[-1]}+"] = _effect_summary(delta_bits[selected])
    return report


def run_ablation(
    checkpoint: Path,
    *,
    valid_path: str = DEFAULT_VALID_PATH,
    eval_bytes: int = 1_000_000,
    boundary: bytes = DEFAULT_BOUNDARY,
    oracle_boundary_reset: bool = False,
    trace_regulatory_modes: bool = False,
    allowed_regulatory_modes: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if allowed_regulatory_modes is not None and any(
        mode not in (0, 1, 2) for mode in allowed_regulatory_modes
    ):
        raise ValueError("allowed regulatory modes must be 0, 1, or 2")
    goal_on = _read_checkpoint(checkpoint)
    goal_off = _read_checkpoint(checkpoint)
    learned_gain = float(goal_on.goal_apical_weight)
    goal_off.goal_apical_weight = 0.0
    stream = np.fromfile(valid_path, dtype=np.uint8, count=eval_bytes)

    on_before = int(getattr(goal_on, "goal_route_flow_uses", 0))
    modes: np.ndarray | None = None
    goals: np.ndarray | None = None
    basins: np.ndarray | None = None
    flow_onsets: np.ndarray | None = None
    if trace_regulatory_modes or allowed_regulatory_modes is not None:
        if oracle_boundary_reset:
            raise ValueError("regulatory tracing and oracle resets are separate audits")
        on_probability, modes, goals, basins, flow_onsets = (
            observe_with_regulatory_trace(
                goal_on, stream, allowed_modes=allowed_regulatory_modes
            )
        )
        goal_off.reset_episode()
        off_probability = goal_off.observe_chunk_probabilities(stream, learn=False)
    elif oracle_boundary_reset:
        on_probability = observe_with_episode_resets(goal_on, stream, boundary)
        off_probability = observe_with_episode_resets(goal_off, stream, boundary)
    else:
        goal_on.reset_episode()
        goal_off.reset_episode()
        on_probability = goal_on.observe_chunk_probabilities(stream, learn=False)
        off_probability = goal_off.observe_chunk_probabilities(stream, learn=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_bits = -np.log2(on_probability) + np.log2(off_probability)
    positions = story_positions(stream, boundary)
    delimiter = boundary_byte_mask(stream, boundary)
    finite = np.isfinite(delta_bits)

    categories = {
        "ascii_letters": ((stream >= 65) & (stream <= 90))
        | ((stream >= 97) & (stream <= 122)),
        "ascii_whitespace": np.isin(stream, np.asarray((9, 10, 13, 32))),
        "ascii_other": stream < 128,
        "high_bit": stream >= 128,
    }
    categories["ascii_other"] &= ~categories["ascii_letters"]
    categories["ascii_other"] &= ~categories["ascii_whitespace"]

    report = {
        "task": "goal_flow_story_horizon_ablation",
        "checkpoint": str(checkpoint),
        "eval_bytes": int(len(stream)),
        "learned_goal_apical_weight": learned_gain,
        "ablation_goal_apical_weight": 0.0,
        "oracle_boundary_reset": oracle_boundary_reset,
        "trace_regulatory_modes": trace_regulatory_modes,
        "allowed_regulatory_modes": allowed_regulatory_modes,
        "goal_route_flow_uses": int(goal_on.goal_route_flow_uses - on_before),
        "overall": _effect_summary(delta_bits),
        "by_story_position": summarize_horizons(delta_bits, positions),
        "by_story_position_content_only": summarize_horizons(
            delta_bits[~delimiter], positions[~delimiter]
        ),
        "by_episode_region": {
            "story_content": _effect_summary(delta_bits[~delimiter]),
            "boundary_delimiter": _effect_summary(delta_bits[delimiter]),
        },
        "by_complete_story": summarize_complete_stories(delta_bits, stream, boundary),
        "by_byte_class": {
            name: _effect_summary(delta_bits[finite & selected])
            for name, selected in categories.items()
        },
    }
    if modes is not None and flow_onsets is not None:
        labels = ("surprise", "repetition", "temporal_disagreement")
        report["by_regulatory_mode"] = {
            labels[mode]: _effect_summary(delta_bits[modes == mode])
            for mode in range(len(labels))
        }
        report["at_flow_onset"] = _effect_summary(delta_bits[flow_onsets])
        report["flow_onsets"] = int(np.sum(flow_onsets))
        report["regulatory_mode_persistence"] = summarize_state_persistence(modes)
        if goals is not None:
            report["selected_goal_persistence"] = summarize_state_persistence(goals)
            report["flow_onsets_with_selected_goal_fraction"] = float(
                np.mean(goals[flow_onsets] >= 0) if np.any(flow_onsets) else 0.0
            )
        if basins is not None and np.any(basins >= 0):
            report["predictive_basin_persistence"] = summarize_state_persistence(basins)
            report["flow_onsets_with_predictive_basin_fraction"] = float(
                np.mean(basins[flow_onsets] >= 0) if np.any(flow_onsets) else 0.0
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--eval-bytes", type=int, default=1_000_000)
    parser.add_argument("--boundary", default=DEFAULT_BOUNDARY.decode("ascii"))
    parser.add_argument("--oracle-boundary-reset", action="store_true")
    parser.add_argument("--trace-regulatory-modes", action="store_true")
    parser.add_argument("--allowed-regulatory-modes", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_ablation(
        args.checkpoint,
        valid_path=args.valid_path,
        eval_bytes=args.eval_bytes,
        boundary=args.boundary.encode("utf-8"),
        oracle_boundary_reset=args.oracle_boundary_reset,
        trace_regulatory_modes=args.trace_regulatory_modes,
        allowed_regulatory_modes=(
            None
            if args.allowed_regulatory_modes is None
            else tuple(args.allowed_regulatory_modes)
        ),
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
