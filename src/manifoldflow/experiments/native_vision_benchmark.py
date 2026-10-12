"""Measure the native growing-vision organism against the PyTorch reference.

Unlike ``native_cortex_benchmark.py``, this cannot report an exact-match
column: the vision port is disclosed as numerically-close, not bit-exact,
and per-step decisions can diverge later in a stream even when evidence
matches the reference to the float32 noise floor (see
``tests/test_native_vision_matches_python.py`` and ``cpp/README.md`` for
why). So this reports throughput plus final accuracy for each backend
side by side -- the honest way to judge "does it still work," not a
mismatch count that this port was never meant to satisfy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from manifoldflow.dendritic_basin.continual_vision import (
    ALM64MB4CGrowingVision,
    ContinualVisionConfig,
)
from manifoldflow.dendritic_basin.native_vision import (
    NativeGrowingVision,
    native_vision_available,
)


def _textured_images(
    count: int, *, classes: int, seed: int, size: int, noise_std: float = 30.0
) -> tuple[list[Image.Image], list[int]]:
    """Class-correlated low-frequency pattern plus per-pixel noise --
    synthetic stand-in for real photo statistics (coherent per-class
    structure plus meaningful high-frequency texture), used here because no
    CLEAR-10/Stream-51 data is available locally. See
    ``tests/test_native_vision_matches_python.py`` for why pure noise or a
    bare smooth pattern are each adversarial in opposite ways for a
    top-k-routed system."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, size)
    xv, yv = np.meshgrid(xs, xs)
    images, labels = [], []
    for _ in range(count):
        label = int(rng.integers(0, classes))
        pattern = (np.sin((label + 1) * 3.0 * xv) * np.cos((label + 1) * 3.0 * yv) + 1.0) / 2.0
        r = np.clip(pattern * 200 + (label * 23) % 255 + rng.normal(0, noise_std, size=(size, size)), 0, 255)
        g = np.clip((1 - pattern) * 200 + rng.normal(0, noise_std, size=(size, size)), 0, 255)
        b = np.clip(pattern * 128 + rng.normal(0, noise_std, size=(size, size)), 0, 255)
        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
        images.append(Image.fromarray(rgb, mode="RGB"))
        labels.append(label)
    return images, labels


def _resized_rgb(image: Image.Image, size: int) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return (
        np.asarray(image.convert("RGB").resize((size, size), resampling), dtype=np.float32)
        / 255.0
    )


def _run_python(
    config: ContinualVisionConfig, images: list[Image.Image], labels: list[int]
) -> dict[str, object]:
    learner = ALM64MB4CGrowingVision(config, device="cpu")
    correct = 0
    started = time.perf_counter()
    for image, label in zip(images, labels):
        prediction, _, evidence = learner.predict(image)
        correct += int(prediction == label)
        learner.observe_external(evidence, label)
    elapsed = time.perf_counter() - started
    return {
        "samples_per_second": len(images) / max(elapsed, 1e-12),
        "elapsed_seconds": elapsed,
        "prequential_accuracy": correct / len(images),
        "sleep_cycles": learner.sleep_cycles,
        "live_assemblies": max(c.live for c in learner.columns),
        "visual_births": learner.visual_births,
    }


def _run_native(
    config: ContinualVisionConfig, images: list[Image.Image], labels: list[int]
) -> dict[str, object]:
    learner = NativeGrowingVision(
        classes=config.classes,
        image_size=config.image_size,
        columns=config.columns,
        context_decay=config.context_decay,
        sleep_interval=config.sleep_interval,
        max_body_bytes=config.max_body_bytes,
        learning_rate=config.learning_rate,
        maximum_synapses=config.maximum_synapses,
        maximum_visual_assemblies=config.maximum_visual_assemblies,
        visual_birth_activation=config.visual_birth_activation,
        visual_routing_width=config.visual_routing_width,
        relation_sketch_words=config.relation_sketch_words,
        relation_turnover_fraction=config.relation_turnover_fraction,
    )
    correct = 0
    started = time.perf_counter()
    for image, label in zip(images, labels):
        rgb = _resized_rgb(image, config.image_size)
        evidence = learner.evidence(rgb)
        prediction, _ = learner.predict(evidence)
        correct += int(prediction == label)
        learner.observe_external(evidence, label)
    elapsed = time.perf_counter() - started
    return {
        "samples_per_second": len(images) / max(elapsed, 1e-12),
        "elapsed_seconds": elapsed,
        "prequential_accuracy": correct / len(images),
        "sleep_cycles": learner.sleep_cycles,
        "live_assemblies": learner.live_assemblies,
        "visual_births": int(learner.visual_births),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not native_vision_available():
        raise SystemExit("alm_vision is not built; see cpp/README.md")

    config = ContinualVisionConfig(
        classes=args.classes,
        image_size=args.image_size,
        columns=args.columns,
        sleep_interval=args.sleep_interval,
        maximum_visual_assemblies=args.maximum_visual_assemblies,
    )
    images, labels = _textured_images(
        args.samples, classes=args.classes, seed=args.seed, size=args.image_size
    )

    torch.set_num_threads(1)
    python_result = _run_python(config, images, labels)
    native_result = _run_native(config, images, labels)

    report = {
        "task": "native_vision_benchmark",
        "correctness_contract": (
            "numerically close, not bit-exact; aggregate quality only past "
            "a short horizon -- see tests/test_native_vision_matches_python.py"
        ),
        "samples": len(images),
        "config": {
            "classes": args.classes,
            "image_size": args.image_size,
            "columns": args.columns,
            "sleep_interval": args.sleep_interval,
            "maximum_visual_assemblies": args.maximum_visual_assemblies,
        },
        "python": python_result,
        "native": native_result,
        "speedup": python_result["elapsed_seconds"] / max(native_result["elapsed_seconds"], 1e-12),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=24)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--sleep-interval", type=int, default=256)
    parser.add_argument("--maximum-visual-assemblies", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
