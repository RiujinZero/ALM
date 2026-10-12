from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.column_partition import (
    partition_cortical_columns,
)


def test_affinity_partition_keeps_recursive_column_pairs_together() -> None:
    current = np.asarray((0, 0, 0, 0, 256, 256, 258, 258), dtype=np.int64)
    traces = np.asarray((1, 1, 1, 1, 1, 1, 1, 1), dtype=np.int64)
    live = np.ones(8, dtype=np.bool_)
    partition = partition_cortical_columns(
        current,
        traces,
        live,
        vocab_size=256,
        column_size=2,
        worker_count=2,
        strategy="affinity",
    )
    assert partition.column_to_worker[0] == partition.column_to_worker[2]
    assert partition.column_to_worker[1] == partition.column_to_worker[3]
    assert partition.edge_locality == 1.0
    assert partition.worker_loads.tolist() == [4, 4]


def test_contiguous_partition_balances_live_tissue() -> None:
    current = np.zeros(10, dtype=np.int64)
    traces = np.ones(10, dtype=np.int64)
    live = np.ones(10, dtype=np.bool_)
    partition = partition_cortical_columns(
        current,
        traces,
        live,
        vocab_size=256,
        column_size=2,
        worker_count=2,
        strategy="contiguous",
    )
    assert partition.worker_loads.tolist() == [6, 4]
    assert partition.recursive_edges == 0
    assert partition.edge_locality == 1.0


def test_contiguous_partition_assigns_empty_developmental_territories() -> None:
    partition = partition_cortical_columns(
        np.zeros(16, dtype=np.int64),
        np.ones(16, dtype=np.int64),
        np.zeros(16, dtype=np.bool_),
        vocab_size=256,
        column_size=2,
        worker_count=4,
        strategy="contiguous",
    )
    assert partition.column_to_worker.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert partition.worker_loads.tolist() == [0, 0, 0, 0]
