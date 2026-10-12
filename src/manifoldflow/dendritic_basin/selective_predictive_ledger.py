"""Surprise-created persistent assemblies with local predictive utility.

The prototype bank grows from observed event vectors only.  A conversation
activates matching prototypes and keeps those activations until reset; unrelated
events do not decay or overwrite earlier entries.  A separate categorical
readout adds one logit vector per active assembly to an externally supplied
causal prior.  Once the successor arrives, local delta updates train the active
assembly weights and leave-one-assembly-out log gain updates their utility.

There is no autograd, backpropagation, or BPTT.  Prototype growth, categorical
updates, and predictive-utility estimates are explicit local NumPy rules.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurprisePrototypeBankConfig:
    input_dim: int
    capacity: int = 64
    birth_distance: float = 1.0
    match_distance: float = 2.5
    prototype_learning_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.capacity <= 0:
            raise ValueError("input_dim and capacity must be positive")
        for name in (
            "birth_distance",
            "match_distance",
            "prototype_learning_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.prototype_learning_rate > 1.0:
            raise ValueError("prototype_learning_rate must be at most one")
        if self.match_distance < self.birth_distance:
            raise ValueError("match_distance cannot be below birth_distance")


class SurprisePrototypeBank:
    """Bounded online prototypes created only by event-space surprise."""

    def __init__(self, config: SurprisePrototypeBankConfig):
        self.config = config
        self._prototypes = np.zeros(
            (config.capacity, config.input_dim), dtype=np.float64
        )
        self._observations = np.zeros(config.capacity, dtype=np.int64)
        self.count = 0
        self.births = 0
        self.unmatched_at_capacity = 0

    def _event(self, event: np.ndarray) -> np.ndarray:
        value = np.asarray(event, dtype=np.float64)
        if value.shape != (self.config.input_dim,):
            raise ValueError(f"event must have shape ({self.config.input_dim},)")
        if not np.all(np.isfinite(value)):
            raise ValueError("event must be finite")
        return value

    def _distances(self, event: np.ndarray) -> np.ndarray:
        return np.sqrt(
            np.mean(np.square(self._prototypes[: self.count] - event), axis=1)
        )

    def observe(self, event: np.ndarray, *, learn: bool) -> int | None:
        """Match causally; optionally grow/update prototypes from this event."""

        value = self._event(event)
        if self.count == 0:
            if not learn:
                return None
            self._prototypes[0] = value
            self._observations[0] = 1
            self.count = 1
            self.births = 1
            return 0
        distances = self._distances(value)
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        if learn and distance >= self.config.birth_distance:
            if self.count < self.config.capacity:
                index = self.count
                self._prototypes[index] = value
                self._observations[index] = 1
                self.count += 1
                self.births += 1
                return index
            self.unmatched_at_capacity += 1
        if distance > self.config.match_distance:
            return None
        if learn:
            rate = self.config.prototype_learning_rate
            self._prototypes[nearest] += rate * (value - self._prototypes[nearest])
            self._observations[nearest] += 1
        return nearest

    @property
    def prototypes(self) -> np.ndarray:
        values = self._prototypes[: self.count].copy()
        values.setflags(write=False)
        return values

    @property
    def observations(self) -> np.ndarray:
        values = self._observations[: self.count].copy()
        values.setflags(write=False)
        return values

    def fingerprint(self) -> str:
        digest = hashlib.sha256(b"surprise-prototype-bank-v1\0")
        for name in self.config.__dataclass_fields__:
            digest.update(name.encode() + b"\0")
            digest.update(repr(getattr(self.config, name)).encode() + b"\0")
        for array in (self._prototypes, self._observations):
            values = np.ascontiguousarray(array)
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(values.tobytes())
        digest.update(self.count.to_bytes(8, "little"))
        digest.update(self.births.to_bytes(8, "little"))
        digest.update(self.unmatched_at_capacity.to_bytes(8, "little"))
        return digest.hexdigest()


@dataclass(frozen=True)
class PredictiveLedgerReadoutConfig:
    assembly_capacity: int
    mode_capacity: int
    learning_rate: float = 0.08
    utility_decay: float = 0.99
    weight_decay: float = 1e-5
    probability_floor: float = 1e-9

    def __post_init__(self) -> None:
        if self.assembly_capacity <= 0 or self.mode_capacity <= 0:
            raise ValueError("assembly and mode capacities must be positive")
        for name in ("learning_rate", "probability_floor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.utility_decay <= 1.0:
            raise ValueError("utility_decay must be in [0, 1]")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")


class PredictiveLedgerReadout:
    """Per-assembly residual mode logits and prequential marginal utility."""

    def __init__(self, config: PredictiveLedgerReadoutConfig):
        self.config = config
        self._weights = np.zeros(
            (config.assembly_capacity, config.mode_capacity + 1),
            dtype=np.float64,
        )
        self._utility = np.zeros(config.assembly_capacity, dtype=np.float64)
        self._fires = np.zeros(config.assembly_capacity, dtype=np.int64)
        self.updates = 0

    def _assemblies(self, active: tuple[int, ...]) -> np.ndarray:
        if len(set(active)) != len(active):
            raise ValueError("active assembly IDs must be unique")
        if any(not 0 <= item < self.config.assembly_capacity for item in active):
            raise ValueError("active assembly outside capacity")
        return np.asarray(active, dtype=np.int64)

    def _classes(self, mode_ids: tuple[int, ...]) -> np.ndarray:
        if len(set(mode_ids)) != len(mode_ids):
            raise ValueError("mode IDs must be unique")
        if any(not 0 <= item < self.config.mode_capacity for item in mode_ids):
            raise ValueError("mode ID outside capacity")
        return np.asarray((0, *(item + 1 for item in mode_ids)), dtype=np.int64)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - float(np.max(logits))
        values = np.exp(shifted)
        return values / values.sum()

    def predict(
        self,
        active: tuple[int, ...],
        mode_ids: tuple[int, ...],
        prior_probabilities: np.ndarray,
    ) -> np.ndarray:
        assemblies = self._assemblies(active)
        classes = self._classes(mode_ids)
        prior = np.asarray(prior_probabilities, dtype=np.float64)
        if prior.shape != (len(classes),):
            raise ValueError("prior must align to basal plus mode IDs")
        if not np.all(np.isfinite(prior)) or np.any(prior <= 0.0):
            raise ValueError("prior must be finite and positive")
        prior = prior / prior.sum()
        residual = (
            np.sum(self._weights[np.ix_(assemblies, classes)], axis=0)
            / math.sqrt(len(assemblies))
            if len(assemblies)
            else np.zeros(len(classes), dtype=np.float64)
        )
        probabilities = self._softmax(np.log(prior) + residual)
        probabilities = np.maximum(probabilities, self.config.probability_floor)
        return probabilities / probabilities.sum()

    def update(
        self,
        active: tuple[int, ...],
        mode_ids: tuple[int, ...],
        prior_probabilities: np.ndarray,
        soft_target: np.ndarray,
    ) -> dict[str, float]:
        assemblies = self._assemblies(active)
        classes = self._classes(mode_ids)
        target = np.asarray(soft_target, dtype=np.float64)
        if target.shape != (len(classes),) or np.any(target < 0.0):
            raise ValueError("soft target must align and be nonnegative")
        if not np.all(np.isfinite(target)) or float(target.sum()) <= 0.0:
            raise ValueError("soft target must be finite with positive mass")
        target = target / target.sum()
        prediction = self.predict(active, mode_ids, prior_probabilities)
        full_loss = -float(np.sum(target * np.log2(np.maximum(prediction, 1e-12))))
        if len(assemblies):
            scale = math.sqrt(len(assemblies))
            decay = self.config.utility_decay
            for assembly in assemblies:
                without = tuple(item for item in active if item != int(assembly))
                leave_prediction = self.predict(without, mode_ids, prior_probabilities)
                leave_loss = -float(
                    np.sum(target * np.log2(np.maximum(leave_prediction, 1e-12)))
                )
                gain = leave_loss - full_loss
                self._utility[assembly] = (
                    decay * self._utility[assembly] + (1.0 - decay) * gain
                )
                self._fires[assembly] += 1
            if self.config.weight_decay:
                self._weights[np.ix_(assemblies, classes)] *= (
                    1.0 - self.config.weight_decay
                )
            self._weights[np.ix_(assemblies, classes)] += (
                self.config.learning_rate * (target - prediction)[None, :] / scale
            )
        self.updates += 1
        return {
            "log_loss_bits": full_loss,
            "brier": float(np.sum(np.square(target - prediction))),
        }

    def retained(
        self, *, minimum_fires: int, minimum_utility_bits: float
    ) -> tuple[int, ...]:
        if minimum_fires < 1 or not math.isfinite(minimum_utility_bits):
            raise ValueError("retention thresholds are invalid")
        return tuple(
            int(index)
            for index in range(self.config.assembly_capacity)
            if self._fires[index] >= minimum_fires
            and self._utility[index] >= minimum_utility_bits
        )

    @property
    def utility(self) -> np.ndarray:
        values = self._utility.copy()
        values.setflags(write=False)
        return values

    @property
    def fires(self) -> np.ndarray:
        values = self._fires.copy()
        values.setflags(write=False)
        return values

    @property
    def trainable_bytes(self) -> int:
        return self._weights.nbytes

    def fingerprint(self) -> str:
        digest = hashlib.sha256(b"predictive-ledger-readout-v1\0")
        for name in self.config.__dataclass_fields__:
            digest.update(name.encode() + b"\0")
            digest.update(repr(getattr(self.config, name)).encode() + b"\0")
        for array in (self._weights, self._utility, self._fires):
            values = np.ascontiguousarray(array)
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(values.tobytes())
        digest.update(self.updates.to_bytes(8, "little"))
        return digest.hexdigest()
