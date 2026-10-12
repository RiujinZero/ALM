"""Latest-phase ALM-64MB-4C velocity-state smoke on rendered ring frames.

Each episode shows one dot moving clockwise or counter-clockwise around a ring.
The current frame identifies position but is exactly ambiguous about the next
position; recent temporal state is required to choose between the two adjacent
locations.  Frames are presented as pixels through the same raw receptor used
by the continual-vision benchmark.  Prediction precedes the next-position
teaching event, and evaluation is frozen on separately generated episodes.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from manifoldflow.dendritic_basin.continual_vision import (
    ALM64MB4CGrowingVision,
    ContinualVisionConfig,
)
from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring-size", type=int, default=8)
    parser.add_argument("--steps-per-episode", type=int, default=24)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--image-pixels", type=int, default=48)
    parser.add_argument("--receptor-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=510)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser


def render_ring_frame(position: int, ring_size: int, pixels: int) -> Image.Image:
    """Render a position without exposing direction or a numeric channel."""
    if not 0 <= position < ring_size:
        raise ValueError("position outside ring")
    image = Image.new("RGB", (pixels, pixels), (4, 7, 13))
    draw = ImageDraw.Draw(image)
    center = (pixels - 1) / 2.0
    radius = pixels * 0.34
    draw.ellipse(
        (
            center - radius,
            center - radius,
            center + radius,
            center + radius,
        ),
        outline=(46, 60, 78),
        width=max(1, pixels // 32),
    )
    angle = -np.pi / 2.0 + 2.0 * np.pi * position / ring_size
    x = center + radius * np.cos(angle)
    y = center + radius * np.sin(angle)
    dot_radius = max(2.0, pixels * 0.055)
    draw.ellipse(
        (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
        fill=(238, 242, 250),
    )
    return image


def generate_episode(
    rng: np.random.Generator, ring_size: int, steps: int
) -> tuple[np.ndarray, int]:
    direction = 1 if int(rng.integers(0, 2)) == 0 else -1
    start = int(rng.integers(0, ring_size))
    positions = np.asarray(
        [(start + direction * step) % ring_size for step in range(steps + 1)],
        dtype=np.int64,
    )
    return positions, direction


def _evaluate(
    vision: ALM64MB4CGrowingVision,
    temporal: SequenceCortexOpponentLearner,
    *,
    episodes: list[tuple[np.ndarray, int]],
    frames: list[Image.Image],
    reset_history_each_frame: bool = False,
) -> dict[str, float | int]:
    correct = 0
    identifiable_correct = 0
    direction_correct = 0
    count = 0
    identifiable_count = 0
    visual_correct = 0
    for positions, direction in episodes:
        vision.reset_episode()
        temporal.reset_episode()
        for step in range(len(positions) - 1):
            if reset_history_each_frame:
                temporal.reset_episode()
            current = int(positions[step])
            target = int(positions[step + 1])
            visual_position, _, _ = vision.predict(frames[current])
            visual_correct += int(visual_position == current)
            temporal.observe(visual_position, learn=False)
            prediction = int(np.argmax(temporal.predict_next_distribution()))
            correct += int(prediction == target)
            count += 1
            if step >= 1:
                identifiable_correct += int(prediction == target)
                predicted_direction = (
                    1
                    if prediction == (current + 1) % vision.config.classes
                    else -1
                    if prediction == (current - 1) % vision.config.classes
                    else 0
                )
                direction_correct += int(predicted_direction == direction)
                identifiable_count += 1
    return {
        "predictions": count,
        "current_frame_visual_accuracy": visual_correct / max(count, 1),
        "overall_next_position_accuracy": correct / max(count, 1),
        "velocity_identifiable_predictions": identifiable_count,
        "velocity_identifiable_next_position_accuracy": identifiable_correct
        / max(identifiable_count, 1),
        "direction_accuracy": direction_correct / max(identifiable_count, 1),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.ring_size < 4 or args.steps_per_episode < 3:
        raise ValueError("ring size >= 4 and at least three steps are required")
    config = ContinualVisionConfig(
        classes=args.ring_size,
        image_size=args.receptor_size,
        columns=4,
        sleep_interval=4096,
        max_body_bytes=64_000_000,
        maximum_visual_assemblies=4096,
    )
    vision = ALM64MB4CGrowingVision(config, device=args.device)
    temporal_columns = [
        SequenceCortexOpponentLearner(
            max_context_bytes=6,
            max_sequence_nursery_paths=50_000,
            packed_paths=True,
        )
        for _ in range(config.columns)
    ]
    frames = [
        render_ring_frame(position, args.ring_size, args.image_pixels)
        for position in range(args.ring_size)
    ]
    train_rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1_000_003)
    evaluation_episodes = [
        generate_episode(eval_rng, args.ring_size, args.steps_per_episode)
        for _ in range(args.eval_episodes)
    ]

    started = time.perf_counter()
    prequential_correct = 0
    prequential_identifiable = 0
    observations = 0
    for _episode_index in range(args.train_episodes):
        positions, _ = generate_episode(
            train_rng, args.ring_size, args.steps_per_episode
        )
        vision.reset_episode()
        temporal = temporal_columns[_episode_index % len(temporal_columns)]
        temporal.reset_episode()
        for step in range(args.steps_per_episode):
            current = int(positions[step])
            target = int(positions[step + 1])
            _visual_prediction, _, evidence = vision.predict(frames[current])
            vision.observe_external(evidence, current)
            temporal.observe(current)
            prediction = int(np.argmax(temporal.predict_next_distribution()))
            if step >= 1:
                prequential_correct += int(prediction == target)
                prequential_identifiable += 1
            observations += 1
            if args.progress_every and observations % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                prequential = prequential_correct / max(
                    prequential_identifiable, 1
                )
                print(
                    f"[ring-dot] {observations:,} observations | "
                    f"{observations / max(elapsed, 1e-9):.1f}/s | "
                    f"prequential {prequential:.4f}",
                    flush=True,
                )

    temporal = copy.deepcopy(temporal_columns[0])
    for column in temporal_columns[1:]:
        temporal.consolidate(column)
    sleep = temporal.consolidate_to_byte_budget(config.max_body_bytes)
    evaluation = _evaluate(
        vision, temporal, episodes=evaluation_episodes, frames=frames
    )
    history_ablated = _evaluate(
        vision,
        temporal,
        episodes=evaluation_episodes,
        frames=frames,
        reset_history_each_frame=True,
    )
    elapsed = time.perf_counter() - started
    developed_body = vision.embodied_bytes + temporal.embodied_bytes
    return {
        "experiment": "latest_phase_alm_64mb_4c_rendered_ring_dot",
        "architecture": {
            "name": "ALM-64MB-4C",
            "phase": (
                "post-Phase-152 sensory receiving tissue plus production "
                "surprise-grown temporal cortex"
            ),
            "body_ceiling_bytes": config.max_body_bytes,
            "serialized_body_bytes": developed_body,
            "vision_body_bytes": vision.embodied_bytes,
            "temporal_body_bytes": temporal.embodied_bytes,
            "mature_visual_assemblies": vision.live_assemblies,
            "active_visual_synapses": vision.active_synapses,
            "mature_temporal_paths": temporal.live_sequence_count,
            "nursery_temporal_paths": temporal.sequence_nursery_count,
            "columns": len(temporal_columns),
            "vision_sleep_cycles": vision.sleep_cycles,
            "structural_sleep": sleep,
        },
        "protocol": {
            "raw_input": (
                "rendered RGB frames; no direction or velocity channel"
            ),
            "causal_order": (
                "current frame -> visual assembly -> temporal prediction -> "
                "next external frame"
            ),
            "train_eval_separation": "independent deterministic episode seeds",
            "frozen_evaluation": True,
            "single_frame_ceiling": 0.5,
            "uniform_class_chance": 1.0 / args.ring_size,
        },
        "config": {**asdict(config), **vars(args), "report": str(args.report)},
        "training": {
            "observations": observations,
            "velocity_identifiable_prequential_accuracy": prequential_correct
            / max(prequential_identifiable, 1),
        },
        "evaluation": evaluation,
        "history_ablated_control": history_ablated,
        "runtime": {
            "seconds": elapsed,
            "observations_per_second": observations / max(elapsed, 1e-9),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": args.device,
        },
    }


def main() -> None:
    args = _parser().parse_args()
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
