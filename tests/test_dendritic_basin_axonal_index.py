from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.axonal_index import AxonalConsumerIndex
from manifoldflow.dendritic_basin.gpu_axonal_scheduler import (
    CudaAxonalDepthScheduler,
)


def _index() -> AxonalConsumerIndex:
    return AxonalConsumerIndex.from_arrays(
        np.asarray((1, 1, 4, 5, 4), dtype=np.int64),
        np.asarray((1, 2, 2, 3, 4), dtype=np.int64),
        np.asarray((True, True, True, True, False)),
        vocab_size=4,
        max_depth=4,
    )


def test_axonal_index_routes_and_deduplicates_by_causal_depth() -> None:
    index = _index()
    np.testing.assert_array_equal(index.consumers(1), (0, 1))
    np.testing.assert_array_equal(index.consumers(4), (2,))
    queues = index.depth_queues(
        np.asarray((1, 4), dtype=np.int32),
        persistent_ids=np.asarray((1, 3), dtype=np.int32),
    )
    np.testing.assert_array_equal(queues[0], (0,))
    np.testing.assert_array_equal(queues[1], (1, 2))
    np.testing.assert_array_equal(queues[2], (3,))
    assert len(queues[3]) == 0
    assert index.storage_bytes == (
        index.offsets.nbytes + index.slot_ids.nbytes + index.depths.nbytes
    )


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA runtime is unavailable")
def test_cuda_axonal_scheduler_matches_cpu_depth_queues() -> None:
    index = _index()
    expected = index.depth_queues(
        np.asarray((1, 4), dtype=np.int32),
        persistent_ids=np.asarray((1, 3), dtype=np.int32),
    )
    actual = CudaAxonalDepthScheduler(index).schedule(
        np.asarray((1, 4), dtype=np.int32),
        persistent_ids=np.asarray((1, 3), dtype=np.int32),
    )
    for actual_depth, expected_depth in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_depth, expected_depth)
