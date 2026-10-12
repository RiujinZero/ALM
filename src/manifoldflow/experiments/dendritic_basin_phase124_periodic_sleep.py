"""Compare one long wake with periodic cortical sleep on the same byte stream."""

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


def _load(path: str | Path) -> SequenceCortexOpponentLearner:
    with Path(path).open("rb") as handle:
        learner = pickle.load(handle)["learner"]
    if not isinstance(learner, SequenceCortexOpponentLearner):
        raise TypeError("checkpoint must contain a sequence cortex")
    learner.reset_episode()
    return learner


def _save(path: Path, learner: SequenceCortexOpponentLearner) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"learner": learner}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _read(path: str | Path, offset: int, length: int) -> np.ndarray:
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError("dataset is shorter than the requested territory")
    return np.frombuffer(raw, dtype=np.uint8)


def _develop(
    arguments: tuple[str, str, int, int, int, bool],
) -> tuple[SequenceCortexOpponentLearner, dict[str, float | int]]:
    checkpoint, data, offset, length, chunk_bytes, lineage_journal = arguments
    learner = _load(checkpoint)
    if lineage_journal:
        learner.begin_lineage_tracking()
    births_before = learner.sequence_births
    evictions_before = learner.sequence_nursery_evictions
    started = time.perf_counter()
    scored_bits = 0.0
    scored_bytes = 0
    for chunk_offset in range(0, length, chunk_bytes):
        values = _read(
            data,
            offset + chunk_offset,
            min(chunk_bytes, length - chunk_offset),
        )
        probability = learner.observe_chunk_probabilities(values, learn=True)
        finite = probability[np.isfinite(probability)]
        scored_bits += float(np.sum(-np.log2(np.maximum(finite, 1e-300))))
        scored_bytes += len(finite)
    elapsed = time.perf_counter() - started
    learner.reset_episode()
    return learner, {
        "bytes": length,
        "seconds": elapsed,
        "bytes_per_second": length / max(elapsed, 1e-12),
        "bpb": scored_bits / max(scored_bytes, 1),
        "sequence_births": learner.sequence_births - births_before,
        "nursery_evictions": learner.sequence_nursery_evictions - evictions_before,
    }


def _score(
    learner: SequenceCortexOpponentLearner, path: Path, length: int
) -> float:
    learner.reset_episode()
    probability = learner.observe_chunk_probabilities(
        _read(path, 0, length), learn=False
    )
    finite = probability[np.isfinite(probability)]
    learner.reset_episode()
    return float(np.mean(-np.log2(np.maximum(finite, 1e-300))))


def _schedule(
    name: str,
    starting_checkpoint: Path,
    data: Path,
    valid: Path,
    work_dir: Path,
    *,
    total_bytes: int,
    rounds: int,
    columns: int,
    workers: int,
    chunk_bytes: int,
    eval_bytes: int,
    max_body_bytes: int,
    lineage_journal: bool,
) -> tuple[dict[str, object], SequenceCortexOpponentLearner]:
    if total_bytes % rounds:
        raise ValueError("total bytes must divide evenly across sleep rounds")
    round_bytes = total_bytes // rounds
    if round_bytes % columns:
        raise ValueError("round bytes must divide evenly across columns")
    territory_bytes = round_bytes // columns
    ancestor = _load(starting_checkpoint)
    generations: list[dict[str, object]] = []
    schedule_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(columns, workers)) as pool:
        for generation in range(rounds):
            # No longer shedding the nursery here -- see the docstring on
            # SequenceCortexOpponentLearner.consolidate_to_byte_budget:
            # consolidate_delta already merges nursery observations
            # correctly across columns, so wiping it every round only
            # threw away real cross-round promotion opportunities.
            ancestor.reset_episode()
            ancestor_path = work_dir / f"{name}_ancestor_{generation}.pkl"
            _save(ancestor_path, ancestor)
            base_offset = generation * round_bytes
            arguments = [
                (
                    str(ancestor_path),
                    str(data),
                    base_offset + column * territory_bytes,
                    territory_bytes,
                    chunk_bytes,
                    lineage_journal,
                )
                for column in range(columns)
            ]
            wake_started = time.perf_counter()
            descendants = list(pool.map(_develop, arguments))
            wake_seconds = time.perf_counter() - wake_started
            master = copy.deepcopy(ancestor)
            merge_started = time.perf_counter()
            for descendant, _ in descendants:
                master.consolidate_delta(descendant, ancestor)
            merge_seconds = time.perf_counter() - merge_started
            body_before_sleep = master.embodied_bytes
            sleep_started = time.perf_counter()
            sleep = master.consolidate_to_byte_budget(max_body_bytes)
            sleep_seconds = time.perf_counter() - sleep_started
            generations.append(
                {
                    "generation": generation,
                    "offset": base_offset,
                    "bytes": round_bytes,
                    "wake_seconds": wake_seconds,
                    "merge_seconds": merge_seconds,
                    "sleep_seconds": sleep_seconds,
                    "body_bytes_before_sleep": body_before_sleep,
                    "body_bytes_after_sleep": master.embodied_bytes,
                    "column_metrics": [row[1] for row in descendants],
                    "sleep": sleep,
                }
            )
            # Descendants are large unpacked developmental bodies. Release the
            # completed generation before workers return the next one so wake
            # generations do not accumulate in the parent's resident memory.
            del descendants
            ancestor = master
    wall_seconds = time.perf_counter() - schedule_started
    result = {
        "name": name,
        "rounds": rounds,
        "lineage_journal": lineage_journal,
        "columns": columns,
        "total_bytes": total_bytes,
        "wall_seconds": wall_seconds,
        "aggregate_bytes_per_second": total_bytes / max(wall_seconds, 1e-12),
        "final_body_bytes": ancestor.embodied_bytes,
        "final_live_sequence_paths": ancestor.live_sequence_count,
        "final_nursery_paths": ancestor.sequence_nursery_count,
        "validation_bpb": _score(ancestor, valid, eval_bytes),
        "generations": generations,
    }
    return result, ancestor


def run(
    checkpoint: Path,
    data: Path,
    valid: Path,
    work_dir: Path,
    *,
    total_bytes: int,
    periodic_rounds: int,
    columns: int,
    workers: int,
    chunk_bytes: int,
    eval_bytes: int,
    max_body_bytes: int,
    lineage_journal: bool = False,
) -> dict[str, object]:
    work_dir.mkdir(parents=True, exist_ok=True)
    control, control_learner = _schedule(
        "end_only",
        checkpoint,
        data,
        valid,
        work_dir,
        total_bytes=total_bytes,
        rounds=1,
        columns=columns,
        workers=workers,
        chunk_bytes=chunk_bytes,
        eval_bytes=eval_bytes,
        max_body_bytes=max_body_bytes,
        lineage_journal=lineage_journal,
    )
    periodic, periodic_learner = _schedule(
        "periodic",
        checkpoint,
        data,
        valid,
        work_dir,
        total_bytes=total_bytes,
        rounds=periodic_rounds,
        columns=columns,
        workers=workers,
        chunk_bytes=chunk_bytes,
        eval_bytes=eval_bytes,
        max_body_bytes=max_body_bytes,
        lineage_journal=lineage_journal,
    )
    _save(work_dir / "end_only_checkpoint.pkl", control_learner)
    _save(work_dir / "periodic_checkpoint.pkl", periodic_learner)
    return {
        "task": "phase124_periodic_sleep",
        "starting_checkpoint": str(checkpoint),
        "data": str(data),
        "valid": str(valid),
        "total_bytes_per_arm": total_bytes,
        "periodic_rounds": periodic_rounds,
        "max_body_bytes": max_body_bytes,
        "lineage_journal": lineage_journal,
        "control": control,
        "periodic": periodic,
        "periodic_minus_control_bpb": (
            periodic["validation_bpb"] - control["validation_bpb"]
        ),
        "periodic_speed_ratio": (
            periodic["aggregate_bytes_per_second"]
            / control["aggregate_bytes_per_second"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("valid", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--total-bytes", type=int, default=64_000_000)
    parser.add_argument("--periodic-rounds", type=int, default=4)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-bytes", type=int, default=1_000_000)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--max-body-bytes", type=int, default=64_000_000)
    parser.add_argument(
        "--lineage-journal",
        action="store_true",
        help="journal every changed path during wake instead of scanning at sleep",
    )
    args = parser.parse_args()
    report = run(
        args.checkpoint,
        args.data,
        args.valid,
        args.work_dir,
        total_bytes=args.total_bytes,
        periodic_rounds=args.periodic_rounds,
        columns=args.columns,
        workers=args.workers,
        chunk_bytes=args.chunk_bytes,
        eval_bytes=args.eval_bytes,
        max_body_bytes=args.max_body_bytes,
        lineage_journal=args.lineage_journal,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    (args.work_dir / "report.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
