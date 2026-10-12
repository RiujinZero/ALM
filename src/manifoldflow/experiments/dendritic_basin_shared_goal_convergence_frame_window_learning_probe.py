"""Test whether a larger context window lets a local cortex learn frame -> target.

Diagnosis under test
--------------------
The base's readable window is ``max_context_bytes`` (6). The prime frame is
2061 bytes. So the base can only condition on a 2-6 byte tail of the frame,
which is why the oracle ceiling is ~0. The intended design is for the base to
read the FULL frame and learn ``frame -> turn`` locally. This experiment
isolates that mechanism: does increasing ``max_context_bytes`` to span the
frame let a fresh local cortex memorize ``frame -> target`` (low held-out
teacher-forced bpb), where window=6 fails?

No backprop / no gradient descent: the cortex is a local byte learner
(observe + learn). We only vary the context-window hyperparameter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.native_cortex import (
    NativeGrowableCortex,
    native_available,
)
from manifoldflow.dendritic_basin.vector_conditioned_base import (
    fit_vector_conditioned_turn,
    score_vector_conditioned_turn,
    vector_input_frame,
)


def _make_cortex(window: int) -> NativeGrowableCortex:
    return NativeGrowableCortex(
        max_context_bytes=window,
        long_range_bank=False,
        long_range_context_bytes=24,
    )


def _make_pairs(rng: np.random.Generator, n: int, dims: int, target_len: int):
    pairs = []
    for i in range(n):
        vector = rng.standard_normal(dims)
        target = rng.integers(0, 256, size=target_len, dtype=np.uint8).tobytes()
        pairs.append((vector_input_frame(vector), target))
    return pairs


def _mean_bpb(diagnostics) -> float:
    return float(np.mean([d.mean_surprise_bits for d in diagnostics]))


def _evaluate(window: int, pairs, heldout: int, label: str) -> dict:
    cortex = _make_cortex(window)
    train_pairs = pairs[:-heldout] if heldout else pairs
    test_pairs = pairs[-heldout:] if heldout else []

    started = time.perf_counter()
    train_diag = []
    for frame, target in train_pairs:
        train_diag.append(
            fit_vector_conditioned_turn(cortex, _frame_to_vector(frame), target)
        )
    train_seconds = time.perf_counter() - started

    test_diag = []
    if test_pairs:
        for frame, target in test_pairs:
            test_diag.append(score_vector_conditioned_turn(cortex, _frame_to_vector(frame), target))

    return {
        "window": window,
        "label": label,
        "n_train": len(train_pairs),
        "n_test": len(test_pairs),
        "train_mean_bpb": _mean_bpb(train_diag) if train_diag else None,
        "test_mean_bpb": _mean_bpb(test_diag) if test_diag else None,
        "train_seconds": train_seconds,
    }


def _frame_to_vector(frame: bytes) -> np.ndarray:
    # frame = magic(9) + uint32(4) + payload(f4). Recover payload as float32.
    import struct

    magic_len = 9
    (dims,) = struct.unpack("<I", frame[magic_len:magic_len + 4])
    payload = frame[magic_len + 4:]
    return np.frombuffer(payload, dtype="<f4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=60)
    parser.add_argument("--dims", type=int, default=512)
    parser.add_argument("--target-len", type=int, default=24)
    parser.add_argument("--heldout", type=int, default=20)
    parser.add_argument("--windows", type=str, default="6,64,256,1024,2061")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--report", default="runs/shared_goal/frame_window_learning_probe.json"
    )
    args = parser.parse_args()

    if not native_available():
        raise SystemExit("native cortex not available")

    rng = np.random.default_rng(args.seed)
    pairs = _make_pairs(rng, args.n_pairs, args.dims, args.target_len)
    windows = [int(w) for w in args.windows.split(",")]

    results = []
    for window in windows:
        started = time.perf_counter()
        results.append(_evaluate(window, pairs, args.heldout, "random_target"))
        print(
            f"window={window:>5} train_bpb="
            f"{results[-1]['train_mean_bpb']:.3f} test_bpb="
            f"{results[-1]['test_mean_bpb']:.3f} "
            f"({time.perf_counter() - started:.1f}s)"
        )

    report = {
        "n_pairs": args.n_pairs,
        "dims": args.dims,
        "target_len": args.target_len,
        "heldout": args.heldout,
        "frame_len": len(pairs[0][0]),
        "results": results,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
