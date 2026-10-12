"""Local categorical selection over a bounded population of future modes.

The selector consumes an already-compressed causal context and a set of mode
identifiers supplied by another mechanism.  A fixed random feature expansion
feeds one categorical readout.  When the future observation later supplies a
soft responsibility vector, the readout applies the local delta rule

``weight[class, synapse] += rate * (target[class] - probability[class]) * feature``.

There is no differentiated hidden stack, autograd, backpropagation, or BPTT.
The mode population and its predictions are owned by the caller and never
changed here.  Class zero is reserved for the basal fallback; mode ``m`` maps
to class ``m + 1``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CategoricalFutureModeSelectorConfig:
    """Capacity and local update constants for the categorical readout."""

    context_dim: int
    mode_capacity: int
    random_features: int = 96
    include_direct_context: bool = True
    feature_scale: float = 0.75
    learning_rate: float = 0.08
    weight_decay: float = 1e-5
    initial_readout_scale: float = 0.01
    probability_floor: float = 1e-9
    seed: int = 261_101

    def __post_init__(self) -> None:
        for name in (
            "context_dim",
            "mode_capacity",
            "random_features",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "feature_scale",
            "learning_rate",
            "probability_floor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if (
            not math.isfinite(self.initial_readout_scale)
            or self.initial_readout_scale < 0.0
        ):
            raise ValueError("initial_readout_scale must be finite and nonnegative")
        if self.probability_floor >= 1.0:
            raise ValueError("probability_floor must be below one")


@dataclass(frozen=True)
class CategoricalSelection:
    """Probability distribution aligned to basal plus available mode IDs."""

    mode_ids: tuple[int, ...]
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities, dtype=np.float64).copy()
        if values.shape != (len(self.mode_ids) + 1,):
            raise ValueError("probabilities must align to basal plus mode_ids")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("probabilities must be finite and positive")
        if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-12):
            raise ValueError("probabilities must sum to one")
        values.setflags(write=False)
        object.__setattr__(self, "probabilities", values)


class CategoricalFutureModeSelector:
    """Fixed-feature, locally trained categorical selector."""

    def __init__(self, config: CategoricalFutureModeSelectorConfig):
        self.config = config
        generator = np.random.default_rng(config.seed)
        self._projection = generator.normal(
            0.0,
            config.feature_scale / math.sqrt(config.context_dim),
            size=(config.context_dim, config.random_features),
        )
        self._phase = generator.uniform(-math.pi, math.pi, size=config.random_features)
        self._readout = generator.normal(
            0.0,
            config.initial_readout_scale,
            size=(config.mode_capacity + 1, self.feature_dimensions),
        )
        self.updates = 0

    @property
    def feature_dimensions(self) -> int:
        direct = self.config.context_dim if self.config.include_direct_context else 0
        return 1 + direct + self.config.random_features

    def _context(self, context: np.ndarray) -> np.ndarray:
        values = np.asarray(context, dtype=np.float64)
        if values.shape != (self.config.context_dim,):
            raise ValueError(f"context must have shape ({self.config.context_dim},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("context must be finite")
        return values

    def features(self, context: np.ndarray) -> np.ndarray:
        """Return fixed bounded features without changing learned state."""

        values = self._context(context)
        random = math.sqrt(2.0 / self.config.random_features) * np.cos(
            values @ self._projection + self._phase
        )
        pieces = [np.asarray([1.0])]
        if self.config.include_direct_context:
            pieces.append(values / math.sqrt(self.config.context_dim))
        pieces.append(random)
        return np.concatenate(pieces)

    def _classes(self, mode_ids: tuple[int, ...]) -> np.ndarray:
        if len(set(mode_ids)) != len(mode_ids):
            raise ValueError("mode_ids must be unique")
        if any(not 0 <= item < self.config.mode_capacity for item in mode_ids):
            raise ValueError("mode_id outside configured capacity")
        return np.asarray((0, *(item + 1 for item in mode_ids)), dtype=np.int64)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - float(np.max(logits))
        values = np.exp(shifted)
        return values / values.sum()

    def predict(
        self, context: np.ndarray, mode_ids: tuple[int, ...]
    ) -> CategoricalSelection:
        """Predict before observing the future target."""

        features = self.features(context)
        classes = self._classes(mode_ids)
        probabilities = self._softmax(self._readout[classes] @ features)
        probabilities = np.maximum(probabilities, self.config.probability_floor)
        probabilities /= probabilities.sum()
        return CategoricalSelection(mode_ids, probabilities)

    def predict_with_prior(
        self,
        context: np.ndarray,
        mode_ids: tuple[int, ...],
        prior_probabilities: np.ndarray,
    ) -> CategoricalSelection:
        """Add this expert's logits to a frozen causal prior distribution."""

        prior = np.asarray(prior_probabilities, dtype=np.float64)
        if prior.shape != (len(mode_ids) + 1,):
            raise ValueError("prior_probabilities must align to basal plus mode_ids")
        if not np.all(np.isfinite(prior)) or np.any(prior <= 0.0):
            raise ValueError("prior_probabilities must be finite and positive")
        prior = prior / prior.sum()
        features = self.features(context)
        classes = self._classes(mode_ids)
        logits = np.log(prior) + self._readout[classes] @ features
        probabilities = self._softmax(logits)
        probabilities = np.maximum(probabilities, self.config.probability_floor)
        probabilities /= probabilities.sum()
        return CategoricalSelection(mode_ids, probabilities)

    def update(
        self,
        context: np.ndarray,
        mode_ids: tuple[int, ...],
        soft_target: np.ndarray,
    ) -> dict[str, float]:
        """Score the causal prediction, then apply one local synaptic update."""

        target = np.asarray(soft_target, dtype=np.float64)
        if target.shape != (len(mode_ids) + 1,):
            raise ValueError("soft_target must align to basal plus mode_ids")
        if not np.all(np.isfinite(target)) or np.any(target < 0.0):
            raise ValueError("soft_target must be finite and nonnegative")
        total = float(target.sum())
        if total <= 0.0:
            raise ValueError("soft_target must have positive mass")
        target = target / total
        features = self.features(context)
        classes = self._classes(mode_ids)
        prediction = self.predict(context, mode_ids).probabilities
        floor = self.config.probability_floor
        log_loss_bits = -float(np.sum(target * np.log2(np.maximum(prediction, floor))))
        brier = float(np.sum(np.square(target - prediction)))
        if self.config.weight_decay:
            self._readout[classes] *= 1.0 - self.config.weight_decay
        self._readout[classes] += self.config.learning_rate * (
            (target - prediction)[:, None] * features[None, :]
        )
        self.updates += 1
        return {"log_loss_bits": log_loss_bits, "brier": brier}

    def update_with_prior(
        self,
        context: np.ndarray,
        mode_ids: tuple[int, ...],
        prior_probabilities: np.ndarray,
        soft_target: np.ndarray,
    ) -> dict[str, float]:
        """Score a frozen-prior prediction, then learn only residual logits."""

        target = np.asarray(soft_target, dtype=np.float64)
        if target.shape != (len(mode_ids) + 1,):
            raise ValueError("soft_target must align to basal plus mode_ids")
        if not np.all(np.isfinite(target)) or np.any(target < 0.0):
            raise ValueError("soft_target must be finite and nonnegative")
        total = float(target.sum())
        if total <= 0.0:
            raise ValueError("soft_target must have positive mass")
        target = target / total
        features = self.features(context)
        classes = self._classes(mode_ids)
        prediction = self.predict_with_prior(
            context, mode_ids, prior_probabilities
        ).probabilities
        floor = self.config.probability_floor
        log_loss_bits = -float(np.sum(target * np.log2(np.maximum(prediction, floor))))
        brier = float(np.sum(np.square(target - prediction)))
        if self.config.weight_decay:
            self._readout[classes] *= 1.0 - self.config.weight_decay
        self._readout[classes] += self.config.learning_rate * (
            (target - prediction)[:, None] * features[None, :]
        )
        self.updates += 1
        return {"log_loss_bits": log_loss_bits, "brier": brier}

    def fingerprint(self) -> str:
        """Hash immutable features, learned weights, configuration, and count."""

        digest = hashlib.sha256()
        digest.update(b"categorical-future-mode-selector-v1\0")
        for name in self.config.__dataclass_fields__:
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(repr(getattr(self.config, name)).encode("ascii") + b"\0")
        for array in (self._projection, self._phase, self._readout):
            values = np.ascontiguousarray(array, dtype="<f8")
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(values.tobytes())
        digest.update(int(self.updates).to_bytes(8, "little", signed=False))
        return digest.hexdigest()

    @property
    def allocated_bytes(self) -> int:
        return self._projection.nbytes + self._phase.nbytes + self._readout.nbytes

    @property
    def trainable_bytes(self) -> int:
        """Bytes changed by learning; fixed feature geometry is excluded."""

        return self._readout.nbytes
