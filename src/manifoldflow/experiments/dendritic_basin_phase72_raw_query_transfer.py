"""Transfer the fast raw-query binder into one frozen TinyStories cortex."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.experiments.dendritic_basin_generate import _read_checkpoint
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_VALID_PATH,
)


def _run_condition(
    checkpoint: str,
    eval_path: str,
    eval_bytes: int,
    mode: str,
) -> tuple[str, dict[str, float | int]]:
    learner = _read_checkpoint(Path(checkpoint))
    learner.reset_episode()
    if mode == "off":
        learner.binding_index = None
    elif mode.startswith("raw_peak_"):
        learner.binding_address_context = "raw_query"
        learner.binding_surprise_bits = float(mode.rsplit("_", 1)[1])
        learner.binding_index.reset_episode()
        learner.binding_raw_window = bytearray()
        learner.binding_recent_raw_queries = []
    formed_before = learner.binding_formation_events
    recalled_before = learner.binding_reinstatements
    stream = np.fromfile(eval_path, dtype=np.uint8, count=eval_bytes)
    bits = 0.0
    scored = 0
    started = time.perf_counter()
    for byte_id in stream:
        probability = learner._observe_probability_only(int(byte_id), learn=False)
        if probability is not None:
            bits += -log2(max(probability, 1e-300))
            scored += 1
    seconds = time.perf_counter() - started
    return mode, {
        "bpb": bits / max(1, scored),
        "scored_bytes": scored,
        "bytes_per_second": len(stream) / max(seconds, 1e-12),
        "binding_formations": learner.binding_formation_events - formed_before,
        "binding_reinstatements": learner.binding_reinstatements - recalled_before,
        "binding_live_cells": (
            0 if learner.binding_index is None else learner.binding_index.live_cells
        ),
        "embodied_bytes": learner.embodied_bytes,
    }


def run_transfer(
    checkpoint: Path,
    eval_path: Path,
    *,
    eval_bytes: int,
    workers: int,
) -> dict[str, object]:
    modes = ("off", "legacy", "raw_peak_2", "raw_peak_4")
    conditions: dict[str, object] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_condition,
                str(checkpoint),
                str(eval_path),
                eval_bytes,
                mode,
            ): mode
            for mode in modes
        }
        for future in as_completed(futures):
            mode, result = future.result()
            conditions[mode] = result
    return {
        "task": "phase72_frozen_tinystories_raw_query_transfer",
        "checkpoint": str(checkpoint),
        "eval_path": str(eval_path),
        "eval_bytes": eval_bytes,
        "conditions": {mode: conditions[mode] for mode in modes},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_transfer(
        args.checkpoint,
        args.eval_path,
        eval_bytes=args.eval_bytes,
        workers=args.workers,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
