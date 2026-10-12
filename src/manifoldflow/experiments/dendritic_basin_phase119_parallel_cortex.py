"""Compare consolidated cortex with permanent parallel cortical columns."""

from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)


@dataclass(frozen=True)
class Territory:
    label: str
    path: str
    offset: int
    length: int


def _new_cortex() -> SequenceCortexOpponentLearner:
    return SequenceCortexOpponentLearner(
        max_context_bytes=6,
        max_sequence_nursery_paths=50_000,
        packed_paths=True,
    )


def _read(path: str | Path, offset: int, length: int) -> np.ndarray:
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"{path} has only {len(raw):,} bytes at offset {offset:,}")
    return np.frombuffer(raw, dtype=np.uint8)


def _train_territory(
    territory: Territory,
) -> tuple[Territory, SequenceCortexOpponentLearner, float]:
    cortex = _new_cortex()
    values = _read(territory.path, territory.offset, territory.length)
    started = time.perf_counter()
    cortex.observe_chunk_probabilities(values, learn=True)
    elapsed = time.perf_counter() - started
    cortex.reset_episode()
    return territory, cortex, elapsed


def _score(
    cortex: SequenceCortexOpponentLearner, values: np.ndarray
) -> tuple[np.ndarray, float]:
    cortex.reset_episode()
    started = time.perf_counter()
    probability = cortex.observe_chunk_probabilities(values, learn=False)
    elapsed = time.perf_counter() - started
    cortex.reset_episode()
    return probability, elapsed


def _bpb(probability: np.ndarray) -> float:
    finite = probability[np.isfinite(probability)]
    return float(np.mean(-np.log2(np.maximum(finite, 1e-300))))


def pool_probabilities(
    probabilities: np.ndarray,
    *,
    mode: str,
    learning_rate: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Causally pool target probabilities from parallel cortical columns."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("probabilities must have shape (columns, bytes)")
    if mode not in {"uniform", "competitive"}:
        raise ValueError("mode must be uniform or competitive")
    weights = np.full(values.shape[0], 1.0 / values.shape[0], dtype=np.float64)
    pooled = np.full(values.shape[1], np.nan, dtype=np.float64)
    for index in range(values.shape[1]):
        current = values[:, index]
        finite = np.isfinite(current)
        if not np.any(finite):
            continue
        active_weights = weights[finite]
        active_weights /= active_weights.sum()
        bounded = np.maximum(current[finite], 1e-300)
        pooled[index] = float(np.dot(active_weights, bounded))
        if mode == "competitive":
            # The observed byte supplies local predictive fitness only after
            # every column has emitted its probability for this causal step.
            weights[finite] *= np.exp(
                learning_rate * (np.log(bounded) - np.max(np.log(bounded)))
            )
            total = weights.sum()
            if total <= 1e-300:
                weights.fill(1.0 / len(weights))
            else:
                weights /= total
    return pooled, weights


def _evaluate_population(
    columns: list[SequenceCortexOpponentLearner],
    values: np.ndarray,
) -> dict[str, object]:
    rows = [_score(column, values) for column in columns]
    matrix = np.stack([probability for probability, _ in rows])
    uniform, _ = pool_probabilities(matrix, mode="uniform")
    competitive, final_weights = pool_probabilities(
        matrix, mode="competitive"
    )
    individual = [_bpb(row) for row in matrix]
    return {
        "individual_bpb": individual,
        "best_individual_bpb": min(individual),
        "mean_individual_bpb": float(np.mean(individual)),
        "uniform_parallel_bpb": _bpb(uniform),
        "competitive_parallel_bpb": _bpb(competitive),
        "competitive_final_weights": final_weights.tolist(),
        "ideal_parallel_bytes_per_second": len(values)
        / max(max(elapsed for _, elapsed in rows), 1e-12),
        "summed_column_cpu_seconds": sum(elapsed for _, elapsed in rows),
    }


def run(
    territories: list[Territory],
    evaluations: dict[str, tuple[Path, int, int]],
    *,
    workers: int,
) -> dict[str, object]:
    if not territories or workers < 1:
        raise ValueError("at least one territory and worker are required")
    train_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, len(territories))) as pool:
        trained = list(pool.map(_train_territory, territories))
    parallel_train_seconds = time.perf_counter() - train_started
    columns = [row[1] for row in trained]

    merge_started = time.perf_counter()
    consolidated = copy.deepcopy(columns[0])
    for column in columns[1:]:
        consolidated.consolidate(column)
    consolidation_seconds = time.perf_counter() - merge_started

    sequential = _new_cortex()
    sequential_started = time.perf_counter()
    for territory in territories:
        sequential.reset_episode()
        sequential.observe_chunk_probabilities(
            _read(territory.path, territory.offset, territory.length), learn=True
        )
    sequential_seconds = time.perf_counter() - sequential_started
    sequential.reset_episode()

    report: dict[str, object] = {
        "task": "phase119_parallel_cortex",
        "columns": len(columns),
        "territories": [territory.__dict__ for territory in territories],
        "train_bytes": sum(territory.length for territory in territories),
        "parallel_train_seconds_including_workers": parallel_train_seconds,
        "maximum_column_train_seconds": max(row[2] for row in trained),
        "consolidation_seconds": consolidation_seconds,
        "sequential_train_seconds": sequential_seconds,
        "parallel_body_bytes": sum(column.embodied_bytes for column in columns),
        "consolidated_body_bytes": consolidated.embodied_bytes,
        "sequential_body_bytes": sequential.embodied_bytes,
        "column_body_bytes": [column.embodied_bytes for column in columns],
        "evaluations": {},
    }
    for label, (path, offset, length) in evaluations.items():
        values = _read(path, offset, length)
        population = _evaluate_population(columns, values)
        consolidated_probability, consolidated_seconds = _score(
            consolidated, values
        )
        sequential_probability, sequential_eval_seconds = _score(
            sequential, values
        )
        population.update(
            {
                "bytes": len(values),
                "consolidated_bpb": _bpb(consolidated_probability),
                "consolidated_bytes_per_second": len(values)
                / max(consolidated_seconds, 1e-12),
                "sequential_bpb": _bpb(sequential_probability),
                "sequential_bytes_per_second": len(values)
                / max(sequential_eval_seconds, 1e-12),
            }
        )
        report["evaluations"][label] = population
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_train", type=Path)
    parser.add_argument("primary_valid", type=Path)
    parser.add_argument("--secondary-train", type=Path)
    parser.add_argument("--secondary-valid", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--train-bytes-per-column", type=int, default=50_000)
    parser.add_argument("--eval-bytes", type=int, default=50_000)
    parser.add_argument("--secondary-valid-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.columns < 1 or args.train_bytes_per_column < 1 or args.eval_bytes < 1:
        raise ValueError("column and byte counts must be positive")
    if (args.secondary_train is None) != (args.secondary_valid is None):
        raise ValueError("secondary train and validation paths must be paired")

    territories = []
    primary_offset = 0
    secondary_offset = 0
    for column in range(args.columns):
        secondary = args.secondary_train is not None and column % 2 == 1
        if secondary:
            path = args.secondary_train
            offset = secondary_offset
            secondary_offset += args.train_bytes_per_column
            label = "secondary"
        else:
            path = args.primary_train
            offset = primary_offset
            primary_offset += args.train_bytes_per_column
            label = "primary"
        territories.append(
            Territory(label, str(path), offset, args.train_bytes_per_column)
        )
    evaluations = {
        "primary": (args.primary_valid, 0, args.eval_bytes),
    }
    if args.secondary_valid is not None:
        evaluations["secondary"] = (
            args.secondary_valid,
            args.secondary_valid_offset,
            args.eval_bytes,
        )
    report = run(territories, evaluations, workers=args.workers)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
