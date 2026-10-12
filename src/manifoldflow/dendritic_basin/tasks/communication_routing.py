"""Modality-neutral hidden routing task with irrelevant-source perturbations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommunicationBatch:
    source_states: np.ndarray
    context_receptors: np.ndarray
    context_ids: np.ndarray
    addressed_sources: np.ndarray
    outcomes: np.ndarray


class HiddenCommunicationTask:
    """Each distributed context selects one of many independent sources."""

    def __init__(
        self,
        source_count: int = 16,
        *,
        context_receptor_count: int = 8,
        active_context_receptors: int = 4,
        seed: int = 0,
    ) -> None:
        if source_count < 2:
            raise ValueError("source_count must be at least two")
        if not 1 <= active_context_receptors < context_receptor_count:
            raise ValueError("active context receptors must fit the organ")
        self.source_count = int(source_count)
        self.context_receptor_count = int(context_receptor_count)
        self._rng = np.random.default_rng(seed)

        patterns: set[bytes] = set()
        codebook = []
        while len(codebook) < source_count:
            active = self._rng.choice(
                context_receptor_count,
                size=active_context_receptors,
                replace=False,
            )
            vector = np.zeros(context_receptor_count, dtype=np.uint8)
            vector[active] = 1
            key = vector.tobytes()
            if key not in patterns:
                patterns.add(key)
                codebook.append(vector)
        self.context_codebook = np.asarray(codebook, dtype=np.uint8)
        self.context_to_source = self._rng.permutation(source_count)

    def sample(
        self, trial_count: int, *, adversarial_irrelevant: bool = False
    ) -> CommunicationBatch:
        trial_count = int(trial_count)
        contexts = self._rng.integers(0, self.source_count, size=trial_count)
        sources = self._rng.integers(
            0, 2, size=(trial_count, self.source_count), dtype=np.uint8
        )
        addressed = self.context_to_source[contexts]
        outcomes = sources[np.arange(trial_count), addressed].copy()
        if adversarial_irrelevant:
            sources[...] = (1 - outcomes)[:, None]
            sources[np.arange(trial_count), addressed] = outcomes
        return CommunicationBatch(
            source_states=sources,
            context_receptors=self.context_codebook[contexts],
            context_ids=contexts,
            addressed_sources=addressed,
            outcomes=outcomes,
        )
