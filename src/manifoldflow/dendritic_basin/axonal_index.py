"""Compact afferent-to-dendrite routing for event-sparse cortical execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldflow.dendritic_basin.online_stream import OnlineDendriticStreamLearner


@dataclass(frozen=True)
class AxonalConsumerIndex:
    """CSR fan-out from sensory/relation channels to live relation slots."""

    offsets: np.ndarray
    slot_ids: np.ndarray
    depths: np.ndarray
    max_relations: int
    max_depth: int

    @classmethod
    def from_arrays(
        cls,
        current_channels: np.ndarray,
        depths: np.ndarray,
        live: np.ndarray,
        *,
        vocab_size: int,
        max_depth: int,
    ) -> AxonalConsumerIndex:
        live_ids = np.flatnonzero(live).astype(np.int32)
        channels = np.asarray(current_channels[live_ids], dtype=np.int64)
        channel_count = vocab_size + len(live)
        if np.any((channels < 0) | (channels >= channel_count)):
            raise ValueError("live branch has an invalid current afferent")
        order = np.argsort(channels, kind="stable")
        ordered_channels = channels[order]
        ordered_ids = live_ids[order]
        counts = np.bincount(ordered_channels, minlength=channel_count)
        offsets = np.empty(channel_count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return cls(
            offsets=offsets,
            slot_ids=ordered_ids,
            depths=np.asarray(depths[ordered_ids], dtype=np.int32),
            max_relations=len(live),
            max_depth=max_depth,
        )

    @classmethod
    def from_learner(
        cls, learner: OnlineDendriticStreamLearner
    ) -> AxonalConsumerIndex:
        return cls.from_arrays(
            learner._current_channels,
            learner._depths,
            learner._live,
            vocab_size=learner.config.vocab_size,
            max_depth=learner.config.max_depth,
        )

    @property
    def storage_bytes(self) -> int:
        return self.offsets.nbytes + self.slot_ids.nbytes + self.depths.nbytes

    def consumers(self, channel: int) -> np.ndarray:
        if not 0 <= channel < len(self.offsets) - 1:
            raise ValueError("channel outside axonal index")
        start, stop = self.offsets[channel : channel + 2]
        return self.slot_ids[start:stop]

    def depth_queues(
        self,
        active_channels: np.ndarray,
        persistent_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Return stable, deduplicated causal-depth queues for afferent events."""
        gathered = []
        for channel in np.asarray(active_channels, dtype=np.int64).reshape(-1):
            gathered.append(self.consumers(int(channel)))
        if persistent_ids is not None:
            gathered.append(np.asarray(persistent_ids, dtype=np.int32).reshape(-1))
        if not gathered:
            return tuple(np.empty(0, dtype=np.int32) for _ in range(self.max_depth))
        nonempty = [values for values in gathered if len(values)]
        if not nonempty:
            return tuple(np.empty(0, dtype=np.int32) for _ in range(self.max_depth))
        selected = np.unique(np.concatenate(nonempty)).astype(np.int32)
        depth_by_slot = np.empty(self.max_relations, dtype=np.int32)
        depth_by_slot[self.slot_ids] = self.depths
        return tuple(
            selected[depth_by_slot[selected] == depth]
            for depth in range(1, self.max_depth + 1)
        )
