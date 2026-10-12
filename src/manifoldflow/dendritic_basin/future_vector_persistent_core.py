"""Persistent, context-keyed core estimator for `z_m = core + edge_m`.

The cheapest possible core -- a per-episode mass-weighted mean, see
future_vector_core_edge.py -- passed its cross-pair context-specificity gate,
but is refit from scratch every episode; it cannot generalize across
mutually compatible futures. The native500 decoder-exposure curve then
showed that more receiver exposure does *not* reliably close a train/predict
distribution gap -- the fix for that kind of gap is making predicted-time
input look more like training-time input, which a per-episode-only estimator
structurally cannot do.

This estimator keeps a small set of context-affinity-gated slots (the same
Gaussian-kernel affinity gate and Robbins-Monro decaying EMA rate already
validated in SurpriseGrownFutureTransition's context assemblies and
vision.hpp's update_winner), each holding ONE persistent core rather than a
population of competing residual modes -- there is only one quantity to
track per slot here, not several alternative futures. A context with no
sufficiently close slot yet falls back to a slowly-updated global core, so
`predict` always returns something rather than requiring a cold-start
special case downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PersistentCoreEstimatorConfig:
    context_dim: int
    core_dim: int
    max_slots: int = 16
    context_bandwidth: float = 1.0
    min_context_affinity: float = 0.50
    slot_learning_rate: float = 0.5
    global_learning_rate: float = 0.5

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not finite:
                raise ValueError(f"{name} must be finite")
        for name in ("context_dim", "core_dim", "max_slots"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("context_bandwidth", "slot_learning_rate", "global_learning_rate"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.min_context_affinity <= 1.0:
            raise ValueError("min_context_affinity must be in [0, 1]")


@dataclass(frozen=True)
class CoreSlotState:
    """Read-only public view of one persistent context slot."""

    slot_id: int
    context_prototype: np.ndarray
    persistent_core: np.ndarray
    observations: float


@dataclass
class _CoreSlot:
    slot_id: int
    context_prototype: np.ndarray
    persistent_core: np.ndarray
    observations: float = 0.0


class PersistentCoreEstimator:
    """Bounded online context-keyed core with a global EMA fallback.

    ``predict`` never learns or mutates state. ``update`` is the only
    learning step: it always refines the global fallback, and separately
    finds or creates the best-matching context slot and refines that too --
    both via a decaying-rate EMA, not a one-shot overwrite, so a slot's
    persistent core is a running estimate across every episode that landed
    in its affinity radius, not just the most recent one.
    """

    def __init__(self, config: PersistentCoreEstimatorConfig):
        self.config = config
        self._slots: list[_CoreSlot | None] = [None] * config.max_slots
        self._global_core = np.zeros(config.core_dim, dtype=np.float64)
        self._global_observations = 0.0
        self.updates = 0
        self.births = 0
        self.recycles = 0

    @property
    def slot_count(self) -> int:
        return sum(slot is not None for slot in self._slots)

    @property
    def slot_states(self) -> tuple[CoreSlotState, ...]:
        return tuple(
            CoreSlotState(
                slot_id=slot.slot_id,
                context_prototype=slot.context_prototype.copy(),
                persistent_core=slot.persistent_core.copy(),
                observations=slot.observations,
            )
            for slot in self._slots
            if slot is not None
        )

    @property
    def global_core(self) -> np.ndarray:
        return self._global_core.copy()

    def _vector(self, value: np.ndarray, dim: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (dim,):
            raise ValueError(f"{name} must have shape ({dim},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector.copy()

    def _affinity(self, context: np.ndarray, prototype: np.ndarray) -> float:
        z = float(
            np.sqrt(np.mean(np.square(context - prototype)))
            / self.config.context_bandwidth
        )
        return math.exp(-0.5 * z * z)

    def _best_slot(self, context: np.ndarray) -> tuple[int, float] | None:
        best: tuple[int, float] | None = None
        for index, slot in enumerate(self._slots):
            if slot is None:
                continue
            affinity = self._affinity(context, slot.context_prototype)
            if affinity < self.config.min_context_affinity:
                continue
            if best is None or affinity > best[1]:
                best = (index, affinity)
        return best

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Affinity-weighted blend of every eligible slot's persistent core.

        Falls back to the global running core if no slot's affinity clears
        `min_context_affinity` -- including before any slot exists.
        """

        context_vector = self._vector(context, self.config.context_dim, "context")
        eligible: list[tuple[_CoreSlot, float]] = []
        for slot in self._slots:
            if slot is None:
                continue
            affinity = self._affinity(context_vector, slot.context_prototype)
            if affinity >= self.config.min_context_affinity:
                eligible.append((slot, affinity))
        if not eligible:
            return self._global_core.copy()
        weights = np.asarray([affinity for _, affinity in eligible], dtype=np.float64)
        weights /= weights.sum()
        blended = sum(
            (weight * slot.persistent_core for weight, (slot, _) in zip(weights, eligible)),
            np.zeros(self.config.core_dim, dtype=np.float64),
        )
        return blended

    def update(self, context: np.ndarray, episode_core: np.ndarray) -> int:
        """Refine the global fallback and the best-matching (or new) slot.

        Returns the updated slot's id. Rate decays as
        `learning_rate / (1 + sqrt(observations))`, the same schedule
        vision.hpp's update_winner uses -- large steps early, smaller ones
        as a slot accumulates evidence, rather than a fixed-window average.
        """

        context_vector = self._vector(context, self.config.context_dim, "context")
        core_vector = self._vector(episode_core, self.config.core_dim, "episode_core")

        global_rate = self.config.global_learning_rate / (
            1.0 + math.sqrt(self._global_observations)
        )
        self._global_core += global_rate * (core_vector - self._global_core)
        self._global_observations += 1.0
        self.updates += 1

        match = self._best_slot(context_vector)
        if match is not None:
            index, _affinity = match
            slot = self._slots[index]
            assert slot is not None
            rate = self.config.slot_learning_rate / (1.0 + math.sqrt(slot.observations))
            slot.persistent_core = slot.persistent_core + rate * (
                core_vector - slot.persistent_core
            )
            slot.observations += 1.0
            return index

        for index, slot in enumerate(self._slots):
            if slot is None:
                self._slots[index] = _CoreSlot(
                    slot_id=index,
                    context_prototype=context_vector.copy(),
                    persistent_core=core_vector.copy(),
                    observations=1.0,
                )
                self.births += 1
                return index

        recycle_index = min(
            range(len(self._slots)),
            key=lambda index: self._slots[index].observations,  # type: ignore[union-attr]
        )
        self._slots[recycle_index] = _CoreSlot(
            slot_id=recycle_index,
            context_prototype=context_vector.copy(),
            persistent_core=core_vector.copy(),
            observations=1.0,
        )
        self.recycles += 1
        return recycle_index
