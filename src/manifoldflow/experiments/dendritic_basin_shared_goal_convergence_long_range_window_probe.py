"""Measure the native base's readable window with the long-range bank on/off.

The short context bank is hard-capped at ``max_context_bytes`` in [2, 8]. The
long-range bank (``long_range_bank=True``) can add a second, deeper context
(``long_range_context_bytes`` ~24, band depths up to 32). This probe trains a
fresh local cortex on a prose slice, with the long-range bank off and on, and
scans in-distribution tail offsets to measure the effective readable window in
each case.

If the long-range bank extends the readable window to ~24-32 bytes, that is a
tractable way to give the base a usable context window for short multi-turn
generation (without a new architecture). Local learning only.
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


def _train(cortex: NativeGrowableCortex, corpus: Path, slice_bytes: int, passes: int) -> None:
    for _ in range(passes):
        with corpus.open("rb") as handle:
            handle.seek(0)
            remaining = slice_bytes
            while remaining > 0:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    break
                cortex.observe_chunk_probabilities(
                    np.frombuffer(chunk, dtype=np.uint8), learn=True
                )
                remaining -= len(chunk)


def _law(cortex: NativeGrowableCortex, frame: bytes) -> np.ndarray:
    cortex.reset_episode()
    cortex.observe_chunk_probabilities(
        np.frombuffer(frame, dtype=np.uint8), learn=False
    )
    values = np.asarray(cortex.predict_next_distribution(), dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(256, 1.0 / 256.0)
    return values / total


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    tiny = np.finfo(np.float64).tiny
    return float(np.sum(np.clip(p, tiny, None) * np.log2(np.clip(p, tiny, None) / np.clip(q, tiny, None))))


def _scan(cortex: NativeGrowableCortex, frame: bytes, max_offset: int) -> list[int]:
    canonical = _law(cortex, frame)
    readable = []
    for offset in range(1, max_offset + 1):
        index = len(frame) - offset
        if index < 0:
            break
        variant = bytearray(frame)
        variant[index] = ord("q") if variant[index] != ord("q") else ord("z")
        law = _law(cortex, bytes(variant))
        if _kl(law, canonical) > 1e-9:
            readable.append(offset)
    return readable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="runs/coadapt_corpus_v1/train.txt")
    parser.add_argument("--slice-bytes", type=int, default=40_000_000)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--max-offset", type=int, default=48)
    parser.add_argument("--report", default="runs/shared_goal/long_range_window_probe.json")
    args = parser.parse_args()

    if not native_available():
        raise SystemExit("native cortex not available")

    corpus = Path(args.corpus)
    # In-distribution, highly common repeated phrase so contexts have mass.
    frame = (b"the cat sat on the mat " * 12).rstrip(b" ")

    configs = [
        ("short_only", dict(max_context_bytes=6, long_range_bank=False)),
        ("long_range_on", dict(max_context_bytes=6, long_range_bank=True)),
        ("long_range_bands", dict(
            max_context_bytes=8, long_range_bank=True,
            long_range_bands=True, long_range_band_count=4,
        )),
    ]

    results = []
    for name, kwargs in configs:
        started = time.perf_counter()
        cortex = NativeGrowableCortex(**kwargs)
        _train(cortex, corpus, args.slice_bytes, args.passes)
        readable = _scan(cortex, frame, args.max_offset)
        results.append({
            "config": name,
            "kwargs": kwargs,
            "readable_offsets": readable,
            "max_readable": max(readable) if readable else 0,
            "train_seconds": time.perf_counter() - started,
        })
        print(f"{name}: readable_offsets={readable} ({time.perf_counter() - started:.1f}s)")

    report = {
        "corpus": str(corpus),
        "slice_bytes": args.slice_bytes,
        "passes": args.passes,
        "frame_tail_hex": frame[-16:].hex(),
        "results": results,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
