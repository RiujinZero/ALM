"""Language-free compositional binding and anti-binding stream."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoleBindingBatch:
    actor_receptors: np.ndarray
    patient_receptors: np.ndarray
    actor_ids: np.ndarray
    patient_ids: np.ndarray
    outcomes: np.ndarray


class CompositionalRoleTask:
    """Pair reusable distributed entities in directed, unseen relations."""

    def __init__(
        self,
        entity_count: int = 16,
        *,
        receptor_count: int = 16,
        active_receptors: int = 8,
        held_out_pair_fraction: float = 0.25,
        seed: int = 0,
    ) -> None:
        if entity_count < 4:
            raise ValueError("entity_count must be at least four")
        if not 1 <= active_receptors < receptor_count:
            raise ValueError("active_receptors must be inside receptor_count")
        if not 0.0 < held_out_pair_fraction < 1.0:
            raise ValueError("held_out_pair_fraction must be in (0, 1)")
        self.entity_count = int(entity_count)
        self.receptor_count = int(receptor_count)
        self.active_receptors = int(active_receptors)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

        patterns: set[bytes] = set()
        codebook = []
        while len(codebook) < self.entity_count:
            active = self._rng.choice(
                self.receptor_count, size=self.active_receptors, replace=False
            )
            vector = np.zeros(self.receptor_count, dtype=np.uint8)
            vector[active] = 1
            key = vector.tobytes()
            if key not in patterns:
                patterns.add(key)
                codebook.append(vector)
        self.codebook = np.asarray(codebook, dtype=np.uint8)

        properties = np.arange(self.entity_count, dtype=np.uint8) & 1
        self._rng.shuffle(properties)
        self.hidden_properties = properties

        unordered = [
            (left, right)
            for left in range(self.entity_count)
            for right in range(left + 1, self.entity_count)
        ]
        self._rng.shuffle(unordered)
        held_out_count = max(1, round(len(unordered) * held_out_pair_fraction))
        held_out = set(unordered[:held_out_count])
        self.train_pairs = np.asarray(
            [
                (actor, patient)
                for actor in range(self.entity_count)
                for patient in range(self.entity_count)
                if actor != patient and tuple(sorted((actor, patient))) not in held_out
            ],
            dtype=np.int64,
        )
        self.eval_pairs = np.asarray(
            [
                (actor, patient)
                for left, right in sorted(held_out)
                for actor, patient in ((left, right), (right, left))
            ],
            dtype=np.int64,
        )
        if len(set(self.train_pairs[:, 0])) != self.entity_count:
            raise RuntimeError("pair split left an entity without an actor example")

    def _sample_pairs(self, pairs: np.ndarray, trial_count: int) -> RoleBindingBatch:
        indices = self._rng.integers(0, len(pairs), size=int(trial_count))
        selected = pairs[indices]
        actors = selected[:, 0]
        patients = selected[:, 1]
        return RoleBindingBatch(
            actor_receptors=self.codebook[actors],
            patient_receptors=self.codebook[patients],
            actor_ids=actors,
            patient_ids=patients,
            outcomes=self.hidden_properties[actors],
        )

    def sample_train(self, trial_count: int) -> RoleBindingBatch:
        return self._sample_pairs(self.train_pairs, trial_count)

    def sample_unseen(self, trial_count: int) -> RoleBindingBatch:
        return self._sample_pairs(self.eval_pairs, trial_count)
