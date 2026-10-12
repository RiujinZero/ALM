"""CUDA depth-queue construction from a compact axonal consumer index."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.axonal_index import AxonalConsumerIndex
from manifoldflow.dendritic_basin.gpu_online_stream import _cupy

_SOURCE = r"""
extern "C" __global__ void schedule_afferents(
    const int* active_channels, int active_count,
    const long long* offsets, const int* consumers, const int* depths,
    int generation, int* stamps, int* queues, int* counts,
    int max_relations, int max_depth
) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int active = thread; active < active_count; active += stride) {
        int channel = active_channels[active];
        long long start = offsets[channel];
        long long stop = offsets[channel + 1];
        for (long long edge = start; edge < stop; ++edge) {
            int slot = consumers[edge];
            if (atomicExch(stamps + slot, generation) == generation) continue;
            int depth = depths[edge] - 1;
            if (depth < 0 || depth >= max_depth) continue;
            int position = atomicAdd(counts + depth, 1);
            queues[depth * max_relations + position] = slot;
        }
    }
}

extern "C" __global__ void schedule_persistence(
    const int* persistent_ids, int persistent_count,
    const int* depth_by_slot, int generation, int* stamps,
    int* queues, int* counts, int max_relations, int max_depth
) {
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= persistent_count) return;
    int slot = persistent_ids[position];
    if (atomicExch(stamps + slot, generation) == generation) return;
    int depth = depth_by_slot[slot] - 1;
    if (depth < 0 || depth >= max_depth) return;
    int target = atomicAdd(counts + depth, 1);
    queues[depth * max_relations + target] = slot;
}
"""


class CudaAxonalDepthScheduler:
    """Build compact GPU work queues without scanning relation slots."""

    def __init__(self, index: AxonalConsumerIndex) -> None:
        cp = _cupy()
        self.cp = cp
        self.index = index
        module = cp.RawModule(
            code=_SOURCE,
            options=("-std=c++17",),
            name_expressions=("schedule_afferents", "schedule_persistence"),
        )
        self._afferents = module.get_function("schedule_afferents")
        self._persistence = module.get_function("schedule_persistence")
        self.offsets = cp.asarray(index.offsets)
        self.consumers = cp.asarray(index.slot_ids)
        self.consumer_depths = cp.asarray(index.depths)
        depth_by_slot = np.zeros(index.max_relations, dtype=np.int32)
        depth_by_slot[index.slot_ids] = index.depths
        self.depth_by_slot = cp.asarray(depth_by_slot)
        self.stamps = cp.zeros(index.max_relations, dtype=cp.int32)
        self.queues = cp.empty(
            (index.max_depth, index.max_relations), dtype=cp.int32
        )
        self.counts = cp.zeros(index.max_depth, dtype=cp.int32)
        self.generation = 0

    def schedule(
        self,
        active_channels: np.ndarray,
        persistent_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        cp = self.cp
        self.generation += 1
        if self.generation >= np.iinfo(np.int32).max:
            self.stamps.fill(0)
            self.generation = 1
        self.counts.fill(0)
        active = cp.asarray(
            np.ascontiguousarray(active_channels, dtype=np.int32).reshape(-1)
        )
        if len(active):
            blocks = min(1024, (len(active) + 255) // 256)
            self._afferents(
                (blocks,),
                (256,),
                (
                    active,
                    np.int32(len(active)),
                    self.offsets,
                    self.consumers,
                    self.consumer_depths,
                    np.int32(self.generation),
                    self.stamps,
                    self.queues,
                    self.counts,
                    np.int32(self.index.max_relations),
                    np.int32(self.index.max_depth),
                ),
            )
        if persistent_ids is not None and len(persistent_ids):
            persistent = cp.asarray(
                np.ascontiguousarray(persistent_ids, dtype=np.int32).reshape(-1)
            )
            self._persistence(
                ((len(persistent) + 255) // 256,),
                (256,),
                (
                    persistent,
                    np.int32(len(persistent)),
                    self.depth_by_slot,
                    np.int32(self.generation),
                    self.stamps,
                    self.queues,
                    self.counts,
                    np.int32(self.index.max_relations),
                    np.int32(self.index.max_depth),
                ),
            )
        counts = self.counts.get()
        return tuple(
            np.sort(self.queues[depth, :count].get())
            for depth, count in enumerate(counts)
        )
