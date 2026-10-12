"""Sparse content-addressable resonance over event capsules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldflow.dendritic_basin.event_capsule import (
    BoundaryEventCapsuleMemory,
    CapsuleRead,
)


@dataclass(frozen=True)
class ResonantCompletion:
    read: CapsuleRead
    maximum_overlap: float
    posterior_peak: float
    posterior_entropy: float
    confidence: float
    active_capsules: int


class SparseResonantCapsuleIndex:
    """Use the distributed cortical state itself as a capsule address."""

    def __init__(
        self,
        *,
        dimensions: int = 512,
        hashes_per_item: int = 4,
        inverse_temperature: float = 8.0,
        completion_threshold: float = 0.02,
        content_threshold: float = 0.25,
        maximum_items: int = 8,
        capture_recency_gain: float = 0.0,
        recall_recency_gain: float = 0.0,
        capture_half_life: float = 1_024.0,
        recall_half_life: float = 1_024.0,
        include_pair_hashes: bool = True,
    ) -> None:
        if dimensions < 8 or hashes_per_item < 1:
            raise ValueError("the sparse expansion dimensions are invalid")
        if inverse_temperature <= 0.0 or completion_threshold < 0.0:
            raise ValueError("the attractor temperature/threshold is invalid")
        if not 0.0 < content_threshold <= 1.0 or maximum_items < 1:
            raise ValueError("the content completion limits are invalid")
        if capture_recency_gain < 0.0 or recall_recency_gain < 0.0:
            raise ValueError("recency gains must be non-negative")
        if capture_half_life <= 0.0 or recall_half_life <= 0.0:
            raise ValueError("recency half-lives must be positive")
        self.dimensions = dimensions
        self.hashes_per_item = hashes_per_item
        self.inverse_temperature = inverse_temperature
        self.completion_threshold = completion_threshold
        self.content_threshold = content_threshold
        self.maximum_items = maximum_items
        self.capture_recency_gain = capture_recency_gain
        self.recall_recency_gain = recall_recency_gain
        self.capture_half_life = capture_half_life
        self.recall_half_life = recall_half_life
        self.include_pair_hashes = include_pair_hashes
        self._address_signature: tuple[object, ...] = ()
        self._address_matrix = np.empty((0, dimensions), dtype=np.float64)
        self._row_capsules: tuple[int, ...] = ()
        self._row_contents: tuple[tuple[tuple[int, int], ...], ...] = ()

    def sparse_code(self, items: tuple[tuple[int, int], ...]) -> np.ndarray:
        """Expand assembly/receptor activity into a deterministic DG-like code."""
        code = np.zeros(self.dimensions, dtype=np.float64)
        base_seeds = [
            (
                (int(assembly) + 1) * 0x9E3779B185EBCA87
                ^ (int(receptor) + 1) * 0xC2B2AE3D27D4EB4F
            )
            & 0xFFFFFFFFFFFFFFFF
            for assembly, receptor in sorted(set(items))
        ]
        pair_seeds = [
            (left * 0xD6E8FEB86659FD93 ^ right * 0xA5A3564E27F8862D)
            & 0xFFFFFFFFFFFFFFFF
            for index, left in enumerate(base_seeds)
            for right in base_seeds[index + 1 :]
        ]
        seeds = base_seeds + pair_seeds if self.include_pair_hashes else base_seeds
        for original_seed in seeds:
            seed = original_seed
            for index in range(self.hashes_per_item):
                seed ^= seed >> 30
                seed = (seed * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
                seed ^= seed >> 27
                seed = (seed * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
                seed ^= seed >> 31
                code[seed % self.dimensions] = 1.0
                seed = (seed + index + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        norm = float(np.linalg.norm(code))
        if norm > 0.0:
            code /= norm
        return code

    @staticmethod
    def _trajectory_rows(
        capsule: object,
    ) -> tuple[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]], ...]:
        frames = tuple(getattr(capsule, "frames", ()))
        if not frames:
            items = tuple(capsule.items)
            return ((items, items),)
        rows = []
        for index, (_query, address) in enumerate(frames):
            content = (
                tuple(frames[index + 1][1])
                if index + 1 < len(frames)
                else tuple(capsule.items)
            )
            if address and content:
                rows.append((tuple(address), content))
        return tuple(rows)

    def resonate(
        self,
        memory: BoundaryEventCapsuleMemory,
        context_items: tuple[tuple[int, int], ...],
        *,
        content_gain: dict[tuple[int, int], float] | None = None,
    ) -> ResonantCompletion | None:
        """Complete a cortical assembly through overlap and normalization."""
        if not context_items or not memory.capsules:
            return None
        query = self.sparse_code(context_items)
        if not np.any(query):
            return None
        signature = tuple(
            (tuple(capsule.items), tuple(getattr(capsule, "frames", ())))
            for capsule in memory.capsules
        )
        if signature != self._address_signature:
            self._address_signature = signature
            rows = [
                (capsule_index, address, content)
                for capsule_index, capsule in enumerate(memory.capsules)
                for address, content in self._trajectory_rows(capsule)
            ]
            self._address_matrix = np.stack(
                [self.sparse_code(address) for _, address, _ in rows]
            )
            self._row_capsules = tuple(row[0] for row in rows)
            self._row_contents = tuple(row[2] for row in rows)
        overlap = self._address_matrix @ query
        logits = self.inverse_temperature * overlap
        for index, capsule_index in enumerate(self._row_capsules):
            capsule = memory.capsules[capsule_index]
            created_at = capsule.__dict__.get("created_at", capsule.last_used)
            capture_age = max(0.0, float(memory.clock - created_at))
            logits[index] += self.capture_recency_gain * 2.0 ** (
                -capture_age / self.capture_half_life
            )
            last_recalled = capsule.__dict__.get("last_recalled")
            if last_recalled is not None:
                recall_age = max(0.0, float(memory.clock - last_recalled))
                logits[index] += self.recall_recency_gain * 2.0 ** (
                    -recall_age / self.recall_half_life
                )
        logits -= float(np.max(logits))
        posterior = np.exp(logits)
        posterior /= float(np.sum(posterior))
        peak = float(np.max(posterior))
        maximum_overlap = float(np.max(overlap))
        confidence = peak * maximum_overlap
        if confidence < self.completion_threshold:
            return None

        content: dict[tuple[int, int], float] = {}
        for weight, items in zip(posterior, self._row_contents, strict=True):
            for item in items:
                gain = 1.0 if content_gain is None else content_gain.get(item, 0.0)
                content[item] = content.get(item, 0.0) + float(weight) * gain
        maximum_content = max(content.values(), default=0.0)
        if maximum_content <= 0.0:
            return None
        completed = [
            (item, weight)
            for item, weight in content.items()
            if weight >= self.content_threshold * maximum_content
        ]
        completed.sort(key=lambda pair: (-pair[1], pair[0]))
        items = tuple(item for item, _ in completed[: self.maximum_items])
        if not items:
            return None
        winner = self._row_capsules[int(np.argmax(posterior))]
        entropy = -float(np.sum(posterior * np.log2(np.maximum(posterior, 1e-300))))
        return ResonantCompletion(
            CapsuleRead(
                winner,
                0,
                items[0][1],
                items,
                confidence,
                len(memory.capsules),
            ),
            maximum_overlap,
            peak,
            entropy,
            confidence,
            len(self._row_capsules),
        )
