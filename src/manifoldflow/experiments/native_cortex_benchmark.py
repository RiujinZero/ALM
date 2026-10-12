"""Measure the native cortex against the Python learner on real bytes.

Reports development throughput, frozen scoring throughput, resident body
size, and -- the only number that licenses the others -- whether the two
implementations produced identical probability bits.
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
    native_version,
)
from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)

DEFAULT_TRAIN = Path(r"C:\path\to\gdrive\training_data\TinyStories-train.txt")
DEFAULT_VALID = Path(r"C:\path\to\gdrive\training_data\TinyStories-valid.txt")


def _mismatches(left: np.ndarray, right: np.ndarray) -> int:
    both_nan = np.isnan(left) & np.isnan(right)
    return int(np.sum(left[~both_nan] != right[~both_nan]))


def _bits_per_byte(probabilities: np.ndarray) -> float:
    finite = probabilities[np.isfinite(probabilities)]
    return float(np.mean(-np.log2(np.maximum(finite, 1e-300))))


def run(args: argparse.Namespace) -> dict[str, object]:
    if not native_available():
        raise SystemExit("alm_native is not built; see cpp/README.md")

    train = np.frombuffer(
        args.train.read_bytes()[: args.train_bytes], dtype=np.uint8
    )
    valid = np.frombuffer(
        args.valid.read_bytes()[: args.eval_bytes], dtype=np.uint8
    )

    started = time.perf_counter()
    reference = SequenceCortexOpponentLearner()
    reference_train = reference.observe_chunk_probabilities(train, learn=True)
    python_develop_seconds = time.perf_counter() - started

    started = time.perf_counter()
    native = NativeGrowableCortex()
    native_train = native.observe_chunk_probabilities(train, learn=True)
    native_develop_seconds = time.perf_counter() - started

    packed = native.pack()

    reference.reset_episode()
    native.reset_episode()
    packed.reset_episode()

    started = time.perf_counter()
    reference_eval = reference.observe_chunk_probabilities(valid, learn=False)
    python_score_seconds = time.perf_counter() - started

    started = time.perf_counter()
    native_eval = native.observe_chunk_probabilities(valid, learn=False)
    native_score_seconds = time.perf_counter() - started

    started = time.perf_counter()
    packed_eval = packed.observe_chunk_probabilities(valid)
    packed_score_seconds = time.perf_counter() - started

    report = {
        "task": "native_cortex_benchmark",
        "version": native_version(),
        "train_bytes": int(len(train)),
        "eval_bytes": int(len(valid)),
        "exact_match": {
            "development": _mismatches(reference_train, native_train) == 0,
            "frozen_growable": _mismatches(reference_eval, native_eval) == 0,
            "frozen_packed": _mismatches(reference_eval, packed_eval) == 0,
        },
        "bits_per_byte": {
            "python": _bits_per_byte(reference_eval),
            "native_growable": _bits_per_byte(native_eval),
            "native_packed": _bits_per_byte(packed_eval),
        },
        "development_bytes_per_second": {
            "python": len(train) / max(python_develop_seconds, 1e-12),
            "native": len(train) / max(native_develop_seconds, 1e-12),
            "speedup": python_develop_seconds / max(native_develop_seconds, 1e-12),
        },
        "frozen_bytes_per_second": {
            "python": len(valid) / max(python_score_seconds, 1e-12),
            "native_growable": len(valid) / max(native_score_seconds, 1e-12),
            "native_packed": len(valid) / max(packed_score_seconds, 1e-12),
            "packed_speedup": python_score_seconds / max(packed_score_seconds, 1e-12),
        },
        "resident_bytes": {
            "native_growable": native.memory_bytes,
            "native_packed": packed.memory_bytes,
            "packed_fraction": packed.memory_bytes / max(native.memory_bytes, 1),
        },
        "anatomy": {
            "live_sequences": native.live_sequence_count,
            "live_contexts": native.live_context_count,
            "sequence_births": native.sequence_births,
            "nursery_evictions": native.sequence_nursery_evictions,
            "matches_python": (
                native.live_sequence_count == reference.live_sequence_count
                and native.live_context_count == reference.live_context_count
                and native.sequence_births == reference.sequence_births
            ),
        },
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--train-bytes", type=int, default=5_000_000)
    parser.add_argument("--eval-bytes", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
