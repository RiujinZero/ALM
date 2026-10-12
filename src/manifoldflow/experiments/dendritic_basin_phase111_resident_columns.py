"""Benchmark a resident ALM-1GB packed cortical population on CPU."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _migrate_sequence_checkpoint(learner: object) -> None:
    """Supply execution-era limits absent from early packed checkpoints."""
    if not hasattr(learner, "max_sequence_nursery_paths"):
        learner.max_sequence_nursery_paths = 50_000
    if not hasattr(learner, "sequence_nursery_evictions"):
        learner.sequence_nursery_evictions = 0


def _column(
    checkpoint_path: str,
    data_path: str,
    start: int,
    train_bytes: int,
    eval_start: int,
    eval_bytes: int,
) -> dict[str, float | int]:
    load_started = time.perf_counter()
    with open(checkpoint_path, "rb") as handle:
        payload = pickle.load(handle)
    learner = payload["learner"]
    _migrate_sequence_checkpoint(learner)
    load_seconds = time.perf_counter() - load_started

    with open(data_path, "rb") as handle:
        handle.seek(start)
        training = np.frombuffer(handle.read(train_bytes), dtype=np.uint8)
        handle.seek(eval_start)
        validation = np.frombuffer(handle.read(eval_bytes), dtype=np.uint8)

    learner.reset_episode()
    train_started = time.perf_counter()
    probability = learner.observe_chunk_probabilities(training, learn=True)
    train_seconds = time.perf_counter() - train_started
    finite = probability[np.isfinite(probability)]

    learner.reset_episode()
    eval_started = time.perf_counter()
    probability = learner.observe_chunk_probabilities(validation, learn=False)
    eval_seconds = time.perf_counter() - eval_started
    finite_eval = probability[np.isfinite(probability)]
    return {
        "train_bytes": len(training),
        "train_seconds": train_seconds,
        "train_bpb": float(
            np.mean(-np.log2(np.maximum(finite, 1e-300)))
        ),
        "eval_bytes": len(validation),
        "eval_seconds": eval_seconds,
        "eval_bpb": float(
            np.mean(-np.log2(np.maximum(finite_eval, 1e-300)))
        ),
        "load_seconds": load_seconds,
    }


def run(
    checkpoint: Path,
    data: Path,
    *,
    columns: int,
    train_bytes_per_column: int,
    eval_bytes: int,
) -> dict[str, object]:
    if columns < 1 or train_bytes_per_column < 1 or eval_bytes < 1:
        raise ValueError("column and byte counts must be positive")
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    learner = payload["learner"]
    _migrate_sequence_checkpoint(learner)
    body_bytes = int(learner.embodied_bytes)
    available = data.stat().st_size
    required = columns * train_bytes_per_column + eval_bytes
    if required > available:
        raise ValueError("dataset is shorter than the requested disjoint shards")

    eval_start = columns * train_bytes_per_column
    wall_started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=columns) as pool:
        futures = {
            pool.submit(
                _column,
                str(checkpoint),
                str(data),
                column * train_bytes_per_column,
                train_bytes_per_column,
                eval_start,
                eval_bytes,
            ): column
            for column in range(columns)
        }
        for future in as_completed(futures):
            row = future.result()
            row["column"] = futures[future]
            rows.append(row)
    wall_seconds = time.perf_counter() - wall_started
    rows.sort(key=lambda row: int(row["column"]))
    total_train = sum(int(row["train_bytes"]) for row in rows)
    total_eval = sum(int(row["eval_bytes"]) for row in rows)
    metabolic_train_seconds = max(float(row["train_seconds"]) for row in rows)
    metabolic_eval_seconds = max(float(row["eval_seconds"]) for row in rows)
    return {
        "task": "phase111_resident_packed_columns",
        "checkpoint": str(checkpoint),
        "data": str(data),
        "columns": columns,
        "body_bytes_per_column": body_bytes,
        "aggregate_body_bytes": columns * body_bytes,
        "aggregate_body_gb": columns * body_bytes / 1_000_000_000.0,
        "train_bytes": total_train,
        "train_bytes_per_column": train_bytes_per_column,
        "steady_train_bytes_per_second": total_train
        / max(metabolic_train_seconds, 1e-12),
        "wall_train_bytes_per_second_including_load": total_train
        / max(wall_seconds, 1e-12),
        "eval_bytes": total_eval,
        "steady_eval_bytes_per_second": total_eval
        / max(metabolic_eval_seconds, 1e-12),
        "mean_train_bpb": float(np.mean([row["train_bpb"] for row in rows])),
        "mean_eval_bpb": float(np.mean([row["eval_bpb"] for row in rows])),
        "maximum_column_train_seconds": metabolic_train_seconds,
        "maximum_column_eval_seconds": metabolic_eval_seconds,
        "wall_seconds_including_load": wall_seconds,
        "columns_detail": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--train-bytes-per-column", type=int, default=100_000)
    parser.add_argument("--eval-bytes", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.checkpoint,
        args.data,
        columns=args.columns,
        train_bytes_per_column=args.train_bytes_per_column,
        eval_bytes=args.eval_bytes,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
