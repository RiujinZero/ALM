"""Verify wake/sleep delta consolidation from one mature cortical ancestor."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)
from manifoldflow.experiments.dendritic_basin_phase111_resident_columns import (
    _migrate_sequence_checkpoint,
)


def _load(
    checkpoint: str | Path, *, shed_nursery: bool = False
) -> SequenceCortexOpponentLearner:
    with Path(checkpoint).open("rb") as handle:
        learner = pickle.load(handle)["learner"]
    _migrate_sequence_checkpoint(learner)
    if not isinstance(learner, SequenceCortexOpponentLearner):
        raise TypeError("checkpoint must contain a sequence cortex")
    if shed_nursery:
        learner.shed_sequence_nursery()
    learner.reset_episode()
    return learner


def _read(path: str | Path, offset: int, length: int) -> np.ndarray:
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError("dataset is shorter than the requested shard")
    return np.frombuffer(raw, dtype=np.uint8)


def _develop(
    arguments: tuple[str, str, int, int],
) -> tuple[SequenceCortexOpponentLearner, float]:
    checkpoint, data, offset, length = arguments
    learner = _load(checkpoint, shed_nursery=True)
    learner.begin_lineage_tracking()
    started = time.perf_counter()
    learner.observe_chunk_probabilities(_read(data, offset, length), learn=True)
    elapsed = time.perf_counter() - started
    learner.reset_episode()
    return learner, elapsed


def _score(
    learner: SequenceCortexOpponentLearner, values: np.ndarray
) -> tuple[float, float]:
    learner.reset_episode()
    started = time.perf_counter()
    probability = learner.observe_chunk_probabilities(values, learn=False)
    elapsed = time.perf_counter() - started
    finite = probability[np.isfinite(probability)]
    learner.reset_episode()
    return (
        float(np.mean(-np.log2(np.maximum(finite, 1e-300)))),
        len(values) / max(elapsed, 1e-12),
    )


def run(
    checkpoint: Path,
    data: Path,
    valid: Path,
    *,
    columns: int,
    train_bytes_per_column: int,
    eval_bytes: int,
    workers: int,
) -> dict[str, object]:
    if min(columns, train_bytes_per_column, eval_bytes, workers) < 1:
        raise ValueError("column, worker, and byte counts must be positive")
    raw_ancestor = _load(checkpoint)
    shed_ancestor_nursery_paths = raw_ancestor.sequence_nursery_count
    ancestor = _load(checkpoint, shed_nursery=True)
    arguments = [
        (
            str(checkpoint),
            str(data),
            column * train_bytes_per_column,
            train_bytes_per_column,
        )
        for column in range(columns)
    ]
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(columns, workers)) as pool:
        rows = list(pool.map(_develop, arguments))
    wake_wall_seconds = time.perf_counter() - wall_started

    master = copy.deepcopy(ancestor)
    merge_started = time.perf_counter()
    for descendant, _ in rows:
        master.consolidate_delta(descendant, ancestor)
    merge_seconds = time.perf_counter() - merge_started

    replay = copy.deepcopy(ancestor)
    replay_started = time.perf_counter()
    for _, data_path, offset, length in arguments:
        replay.reset_episode()
        replay.observe_chunk_probabilities(
            _read(data_path, offset, length), learn=True
        )
    replay_seconds = time.perf_counter() - replay_started
    replay.reset_episode()

    validation = _read(valid, 0, eval_bytes)
    master_bpb, master_bps = _score(master, validation)
    replay_bpb, replay_bps = _score(replay, validation)
    return {
        "task": "phase120_delta_consolidation",
        "checkpoint": str(checkpoint),
        "columns": columns,
        "train_bytes": columns * train_bytes_per_column,
        "eval_bytes": eval_bytes,
        "ancestor_body_bytes": ancestor.embodied_bytes,
        "shed_ancestor_nursery_paths": shed_ancestor_nursery_paths,
        "awake_parallel_body_bytes": sum(row[0].embodied_bytes for row in rows),
        "consolidated_body_bytes": master.embodied_bytes,
        "replay_body_bytes": replay.embodied_bytes,
        "wake_wall_seconds_including_workers": wake_wall_seconds,
        "maximum_column_development_seconds": max(row[1] for row in rows),
        "maximum_touched_contexts": max(
            len(row[0]._lineage_context_keys) for row in rows
        ),
        "maximum_touched_sequence_paths": max(
            sum(len(bank) for bank in row[0]._lineage_sequence_keys)
            for row in rows
        ),
        "delta_consolidation_seconds": merge_seconds,
        "sequential_replay_seconds": replay_seconds,
        "consolidated_bpb": master_bpb,
        "sequential_replay_bpb": replay_bpb,
        "bpb_difference": master_bpb - replay_bpb,
        "consolidated_inference_bytes_per_second": master_bps,
        "replay_inference_bytes_per_second": replay_bps,
        "consolidated_live_sequence_paths": master.live_sequence_count,
        "replay_live_sequence_paths": replay.live_sequence_count,
        "consolidated_nursery_paths": master.sequence_nursery_count,
        "replay_nursery_paths": replay.sequence_nursery_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("valid", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--train-bytes-per-column", type=int, default=250_000)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.checkpoint,
        args.data,
        args.valid,
        columns=args.columns,
        train_bytes_per_column=args.train_bytes_per_column,
        eval_bytes=args.eval_bytes,
        workers=args.workers,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
