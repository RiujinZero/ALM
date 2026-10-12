"""Reusable content assemblies coordinated by local role receptors."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum

import numpy as np


class BindingMode(str, Enum):
    """Ablations for content coactivity, conjunction, and role binding."""

    BAG = "bag"
    CONJUNCTIVE = "conjunctive"
    ROLE_GATED = "role_gated"


@dataclass(frozen=True)
class RoleBindingStep:
    observed_probability: float
    predicted_outcome: int
    update_count: int
    off_role_updates: int


def _assembly_key(receptors: np.ndarray) -> bytes:
    vector = np.asarray(receptors, dtype=np.uint8).reshape(-1)
    if np.any(vector > 1):
        raise ValueError("content receptors must be binary")
    return len(vector).to_bytes(2, "little") + np.packbits(vector).tobytes()


class RoleBindingLearner:
    """Local outcome counts over distributed content assemblies.

    The role-gated condition updates the actor assembly only.  The bag
    condition treats simultaneous content as an unordered association.  The
    conjunctive condition preserves the whole ordered episode but cannot reuse
    content knowledge in a novel pairing.
    """

    def __init__(
        self,
        *,
        binding_mode: BindingMode | str,
        prior: float = 0.5,
    ) -> None:
        if prior <= 0.0:
            raise ValueError("prior must be positive")
        self.binding_mode = BindingMode(binding_mode)
        self.prior = float(prior)
        self.outcome_counts: dict[bytes | tuple[bytes, bytes], np.ndarray] = {}
        self.steps = 0
        self.local_updates = 0
        self.off_role_updates = 0

    @property
    def embodied_bytes(self) -> int:
        body = (self.binding_mode.value, self.prior, self.outcome_counts)
        return len(pickle.dumps(body, protocol=5))

    def _row(self, key: bytes | tuple[bytes, bytes]) -> np.ndarray:
        row = self.outcome_counts.get(key)
        if row is None:
            return np.zeros(2, dtype=np.float64)
        return row

    def _probability_one(self, actor: bytes, patient: bytes) -> float:
        if self.binding_mode is BindingMode.CONJUNCTIVE:
            counts = self._row((actor, patient)) + self.prior
        elif self.binding_mode is BindingMode.ROLE_GATED:
            counts = self._row(actor) + self.prior
        else:
            counts = self._row(actor) + self._row(patient) + self.prior
        return float(counts[1] / counts.sum())

    def probability(
        self, actor_receptors: np.ndarray, patient_receptors: np.ndarray
    ) -> float:
        actor = _assembly_key(actor_receptors)
        patient = _assembly_key(patient_receptors)
        if len(actor) != len(patient):
            raise ValueError("actor and patient receptor organs must match")
        return self._probability_one(actor, patient)

    def observe(
        self,
        actor_receptors: np.ndarray,
        patient_receptors: np.ndarray,
        outcome: int,
        *,
        learn: bool = True,
    ) -> RoleBindingStep:
        outcome = int(outcome)
        if outcome not in (0, 1):
            raise ValueError("outcome must be binary")
        actor = _assembly_key(actor_receptors)
        patient = _assembly_key(patient_receptors)
        if len(actor) != len(patient):
            raise ValueError("actor and patient receptor organs must match")
        probability_one = self._probability_one(actor, patient)
        observed = probability_one if outcome else 1.0 - probability_one
        update_count = 0
        off_role_updates = 0

        if learn:
            if self.binding_mode is BindingMode.CONJUNCTIVE:
                targets: tuple[bytes | tuple[bytes, bytes], ...] = ((actor, patient),)
            elif self.binding_mode is BindingMode.ROLE_GATED:
                targets = (actor,)
            else:
                targets = (actor, patient)
            for index, key in enumerate(targets):
                row = self.outcome_counts.setdefault(
                    key, np.zeros(2, dtype=np.float64)
                )
                row[outcome] += 1.0
                update_count += 1
                if self.binding_mode is BindingMode.BAG and index == 1:
                    off_role_updates += 1
            self.local_updates += update_count
            self.off_role_updates += off_role_updates
            self.steps += 1

        return RoleBindingStep(
            observed_probability=observed,
            predicted_outcome=int(probability_one >= 0.5),
            update_count=update_count,
            off_role_updates=off_role_updates,
        )
