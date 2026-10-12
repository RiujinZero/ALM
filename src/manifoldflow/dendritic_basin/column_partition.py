"""Cache-local cortical-column territories for sparse CPU execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from manifoldflow.dendritic_basin.online_stream import OnlineDendriticStreamLearner


@dataclass(frozen=True)
class CorticalColumnPartition:
    """Assignment of anatomical columns to persistent CPU workers."""

    column_to_worker: np.ndarray
    column_loads: np.ndarray
    worker_loads: np.ndarray
    recursive_edges: int
    cross_worker_edges: int

    @property
    def worker_count(self) -> int:
        return len(self.worker_loads)

    @property
    def edge_locality(self) -> float:
        if self.recursive_edges == 0:
            return 1.0
        return 1.0 - self.cross_worker_edges / self.recursive_edges

    @property
    def load_imbalance(self) -> float:
        mean = float(np.mean(self.worker_loads))
        return 0.0 if mean == 0.0 else float(np.max(self.worker_loads) / mean - 1.0)


def _column_graph(
    current_channels: np.ndarray,
    trace_channels: np.ndarray,
    live: np.ndarray,
    *,
    vocab_size: int,
    column_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    relation_count = len(live)
    column_count = math.ceil(relation_count / column_size)
    live_ids = np.flatnonzero(live)
    column_loads = np.bincount(
        live_ids // column_size, minlength=column_count
    ).astype(np.int64)
    adjacency = np.zeros((column_count, column_count), dtype=np.int64)
    recursive_edges = 0
    for channels in (current_channels, trace_channels):
        recursive = channels[live_ids] - vocab_size
        valid = recursive >= 0
        parent_columns = recursive[valid] // column_size
        child_columns = live_ids[valid] // column_size
        np.add.at(adjacency, (parent_columns, child_columns), 1)
        np.add.at(adjacency, (child_columns, parent_columns), 1)
        recursive_edges += int(np.count_nonzero(valid))
    return column_loads, adjacency, recursive_edges


def _partition_metrics(
    assignment: np.ndarray,
    column_loads: np.ndarray,
    adjacency: np.ndarray,
    recursive_edges: int,
    worker_count: int,
) -> CorticalColumnPartition:
    worker_loads = np.bincount(
        assignment, weights=column_loads, minlength=worker_count
    ).astype(np.int64)
    cross_worker_edges = 0
    rows, columns = np.nonzero(np.triu(adjacency, k=1))
    for left, right in zip(rows, columns, strict=True):
        if assignment[left] != assignment[right]:
            cross_worker_edges += int(adjacency[left, right])
    return CorticalColumnPartition(
        column_to_worker=assignment.astype(np.int32),
        column_loads=column_loads,
        worker_loads=worker_loads,
        recursive_edges=recursive_edges,
        cross_worker_edges=cross_worker_edges,
    )


def partition_cortical_columns(
    current_channels: np.ndarray,
    trace_channels: np.ndarray,
    live: np.ndarray,
    *,
    vocab_size: int,
    column_size: int,
    worker_count: int,
    strategy: str = "contiguous",
) -> CorticalColumnPartition:
    """Group columns while balancing live tissue and recursive affinity."""
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    column_loads, adjacency, recursive_edges = _column_graph(
        np.asarray(current_channels),
        np.asarray(trace_channels),
        np.asarray(live, dtype=np.bool_),
        vocab_size=vocab_size,
        column_size=column_size,
    )
    column_count = len(column_loads)
    worker_count = min(worker_count, column_count)
    if strategy == "contiguous":
        # Territory is anatomical, not a transient consequence of occupancy.
        # Stable geometric ownership gives every worker future nursery space
        # even when the organism starts empty, and never migrates a column's
        # local tissue merely because development changed its live count.
        assignment = np.arange(column_count) * worker_count // column_count
    elif strategy == "affinity":
        assignment = np.full(column_count, -1, dtype=np.int64)
        worker_loads = np.zeros(worker_count, dtype=np.int64)
        target = math.ceil(max(int(column_loads.sum()), 1) / worker_count)
        capacity = target + int(column_loads.max(initial=0))
        degree = adjacency.sum(axis=1)
        order = np.lexsort((np.arange(column_count), -degree, -column_loads))
        for column in order:
            eligible = np.flatnonzero(
                worker_loads + column_loads[column] <= capacity
            )
            if not len(eligible):
                eligible = np.arange(worker_count)
            affinity = np.zeros(worker_count, dtype=np.int64)
            placed = np.flatnonzero(assignment >= 0)
            if len(placed):
                np.add.at(
                    affinity,
                    assignment[placed],
                    adjacency[column, placed],
                )
            best_affinity = affinity[eligible].max(initial=0)
            candidates = eligible[affinity[eligible] == best_affinity]
            worker = int(candidates[np.argmin(worker_loads[candidates])])
            assignment[column] = worker
            worker_loads[worker] += column_loads[column]
    else:
        raise ValueError("strategy must be 'contiguous' or 'affinity'")
    return _partition_metrics(
        assignment, column_loads, adjacency, recursive_edges, worker_count
    )


def partition_learner_columns(
    learner: OnlineDendriticStreamLearner,
    worker_count: int,
    *,
    strategy: str = "contiguous",
) -> CorticalColumnPartition:
    return partition_cortical_columns(
        learner._current_channels,
        learner._trace_channels,
        learner._live,
        vocab_size=learner.config.vocab_size,
        column_size=learner.config.column_size,
        worker_count=worker_count,
        strategy=strategy,
    )
