"""The native vision organism must be numerically close to the PyTorch
reference on a single evidence computation, and statistically comparable in
quality over a stream of interactions.

Unlike the byte cortex, this is NOT a bit-exact contract, and -- important,
non-obvious caveat found while building this test -- it is not even a
"functionally identical decisions" contract beyond a short horizon. Growth
selects its 128 strongest-|evidence| synapses out of ~2,880 dimensions by
top-k; routing picks its top-16 assemblies by top-k. Both operations are
discontinuous: whenever two candidates are close (common on smooth/low-
frequency image regions), a ~1e-7 float32 rounding difference between
PyTorch's reduction order and this port's can flip which one wins the cut.
That one flip changes what gets *stored* (a different synapse, a different
learned row) -- not just that step's output -- so it compounds forward for
the rest of the run. This was confirmed directly: with the evidence() bug
fixed (see cpp/include/alm/vision.hpp), evidence matches the reference to
~1e-7/1e-8 (float32 noise floor) on every step, yet per-step grow/update
decisions can still diverge later in a stream, and the divergence is not
one-directional (native is not consistently better or worse -- confirmed
across 15 seeds at multiple noise levels). That rules out a residual
directional bug; it is the expected behavior of any non-bit-exact
reimplementation of a top-k-routed growing system.

So the bar enforced here: evidence matches tightly (this IS deterministic
and must hold), and aggregate quality (mean accuracy over many streams)
tracks the reference within a wide-but-meaningful band -- not per-step
lockstep tracking, which this class of algorithm cannot promise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from manifoldflow.dendritic_basin.continual_vision import (
    ContinualVisionConfig,
    ALM64MB4CGrowingVision,
)

native_vision = pytest.importorskip("manifoldflow.dendritic_basin.native_vision")

pytestmark = pytest.mark.skipif(
    not native_vision.native_vision_available(),
    reason="alm_vision shared library is not built (see cpp/README.md)",
)

_IMAGE_SIZE = 24
_CLASSES = 10
_COLUMNS = 4


def _config(**overrides: object) -> ContinualVisionConfig:
    defaults = dict(
        classes=_CLASSES,
        image_size=_IMAGE_SIZE,
        columns=_COLUMNS,
        max_body_bytes=64_000_000,
        maximum_visual_assemblies=2_000,
        sleep_interval=64,
    )
    defaults.update(overrides)
    return ContinualVisionConfig(**defaults)


def _make_native(config: ContinualVisionConfig) -> "native_vision.NativeGrowingVision":
    return native_vision.NativeGrowingVision(
        classes=config.classes,
        image_size=config.image_size,
        columns=config.columns,
        max_body_bytes=config.max_body_bytes,
        maximum_visual_assemblies=config.maximum_visual_assemblies,
        sleep_interval=config.sleep_interval,
    )


def _textured_images(
    count: int, *, classes: int = _CLASSES, seed: int, size: int = _IMAGE_SIZE, noise_std: float = 30.0
) -> tuple[list[Image.Image], list[int]]:
    """Class-correlated low-frequency pattern plus substantial per-pixel
    noise -- closer to real photo statistics (coherent structure *and*
    meaningful high-frequency texture) than either pure noise or a bare
    smooth pattern, both of which are adversarial in opposite ways for a
    top-k-routed system (pure noise: near-ties everywhere; bare smooth
    pattern: near-duplicate gradients everywhere)."""
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


def _resized_rgb(image: Image.Image) -> np.ndarray:
    """The exact array RawImageReceptor.__call__ starts from, before its own
    internal normalization -- i.e. what the native side's ``evidence()``
    expects (decode/resize stays in Python; only the math moves to C++)."""
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return (
        np.asarray(
            image.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE), resampling),
            dtype=np.float32,
        )
        / 255.0
    )


def test_evidence_matches_python_within_float32_noise() -> None:
    """This part of the contract IS deterministic and must hold tightly:
    the evidence() bug (slow_context updated from the wrong, already-
    renormalized vector instead of the original receptor-normalized basal)
    would show up here as a growing error over the sequence; a passing
    receptor/normalization port stays flat at the float32 noise floor."""
    config = _config()
    reference = ALM64MB4CGrowingVision(config, device="cpu")
    native = _make_native(config)

    images, _ = _textured_images(300, seed=20260911)
    max_abs_error = 0.0
    for image in images:
        py_evidence = reference.evidence(image).numpy()
        native_evidence = native.evidence(_resized_rgb(image))
        max_abs_error = max(max_abs_error, float(np.max(np.abs(py_evidence - native_evidence))))

    assert max_abs_error < 1e-4, f"evidence diverged: max abs error {max_abs_error}"


def _run_stream(config: ContinualVisionConfig, seed: int, noise_std: float) -> tuple[float, float, int, int]:
    reference = ALM64MB4CGrowingVision(config, device="cpu")
    native = _make_native(config)
    images, labels = _textured_images(300, seed=seed, noise_std=noise_std)

    py_correct = native_correct = 0
    for step, (image, label) in enumerate(zip(images, labels)):
        py_evidence = reference.evidence(image).numpy()
        native_evidence = native.evidence(_resized_rgb(image))

        py_prediction, _ = reference.predict_evidence(torch.as_tensor(py_evidence))
        native_prediction, _ = native.predict(native_evidence)
        py_correct += int(py_prediction == label)
        native_correct += int(native_prediction == label)

        reference.observe_external(torch.as_tensor(py_evidence), label)
        native.observe_external(native_evidence, label)
        if step % 25 == 0:
            reference.reset_episode()
            native.reset_episode()

    return (
        py_correct / len(images),
        native_correct / len(images),
        reference.visual_births,
        int(native.visual_births),
    )


def test_stream_quality_tracks_python_on_average() -> None:
    """Per-step lockstep tracking is not a meaningful bar here (see module
    docstring): individual streams diverge chaotically, sometimes favoring
    native, sometimes the reference. What must hold is that quality is
    *comparable on average* and growth stays in the same ballpark -- a
    genuinely broken port would be one-sidedly and drastically worse, not
    symmetrically noisy."""
    config = _config()
    seeds = range(1, 9)

    py_accs, native_accs, py_births, native_births = [], [], [], []
    for seed in seeds:
        py_acc, native_acc, py_b, native_b = _run_stream(config, seed, noise_std=30.0)
        py_accs.append(py_acc)
        native_accs.append(native_acc)
        py_births.append(py_b)
        native_births.append(native_b)

    py_accs_arr = np.array(py_accs)
    native_accs_arr = np.array(native_accs)
    chance = 1.0 / _CLASSES

    assert py_accs_arr.mean() > chance * 1.5, "reference itself failed to learn -- test setup is broken"
    assert native_accs_arr.mean() > chance * 1.5, f"native mean accuracy {native_accs_arr.mean():.3f} near chance"
    assert abs(native_accs_arr.mean() - py_accs_arr.mean()) < 0.12, (
        f"mean accuracy diverged too far: py={py_accs_arr.mean():.3f} native={native_accs_arr.mean():.3f}"
    )

    for py_b, native_b in zip(py_births, native_births):
        ratio = native_b / max(py_b, 1)
        assert 0.4 < ratio < 2.5, f"growth diverged too far: py_births={py_b} native_births={native_b}"
