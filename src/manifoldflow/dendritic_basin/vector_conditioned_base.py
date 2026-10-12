"""Drive the byte cortex directly from a dense predicted turn vector.

This is the inverse of the old receiver arrangement. The dense vector is an
input frame, not a residual correction to byte logits. The cortex is primed
with that frame and the observed target turn is then learned as its byte
continuation. Consequently all response surprise and plasticity live in the
base cortex itself.

The current cortex alphabet is byte-valued, so a raw floating-point vector is
crossed into that alphabet with a versioned, canonical IEEE-754 frame. There
is deliberately no PCA top-k selection, magnitude binning, or sparse receiver
feature hashing in this boundary encoding.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

_FRAME_MAGIC = b"\x00ALM-VEC\x01"


class VectorConditionedCortex(Protocol):
    """Small common surface implemented by Python and native byte cortices."""

    def reset_episode(self) -> None: ...

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> Any: ...

    def predict_next_distribution(self) -> np.ndarray: ...

    def advance_observation(self, byte_id: int, *, learn: bool = False) -> Any: ...


@dataclass(frozen=True)
class VectorConditionedFitDiagnostics:
    """Response-only surprise from one vector -> target-turn training pair."""

    vector_dimensions: int
    conditioning_bytes: int
    target_bytes: int
    total_surprise_bits: float
    mean_surprise_bits: float
    minimum_observed_probability: float


@dataclass(frozen=True)
class StatefulVectorConditionedConfig:
    """Bounded local state for a vector-conditioned byte transducer.

    The adapter keeps a small associative state in the base wrapper. State
    updates use only the current surprise, the active state, and the observed
    byte; there is no backward pass or gradient estimate. Weights and state are
    explicitly bounded so free generation cannot run away.
    """

    state_dimensions: int = 32
    learning_rate: float = 0.03
    state_decay: float = 0.98
    surprise_gate_threshold: float = 2.0
    surprise_gate_scale: float = 1.0
    weight_decay: float = 0.001
    weight_clip: float = 4.0
    modulation_strength: float = 1.0
    seed: int = 0
    update_on_generated: bool = True

    def __post_init__(self) -> None:
        if self.state_dimensions < 1:
            raise ValueError("state_dimensions must be positive")
        if not math.isfinite(self.learning_rate) or not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be finite and in (0, 1]")
        if (
            not math.isfinite(self.state_decay)
            or not 0.0 <= self.state_decay <= 1.0
        ):
            raise ValueError("state_decay must be finite and in [0, 1]")
        if not math.isfinite(self.surprise_gate_threshold):
            raise ValueError("surprise_gate_threshold must be finite")
        if (
            not math.isfinite(self.surprise_gate_scale)
            or self.surprise_gate_scale <= 0.0
        ):
            raise ValueError("surprise_gate_scale must be finite and positive")
        if not math.isfinite(self.weight_decay) or not 0.0 <= self.weight_decay < 1.0:
            raise ValueError("weight_decay must be finite and in [0, 1)")
        if (
            not math.isfinite(self.weight_clip)
            or self.weight_clip <= 0.0
            or not math.isfinite(self.modulation_strength)
            or self.modulation_strength < 0.0
        ):
            raise ValueError(
                "weight_clip must be finite and positive; modulation nonnegative"
            )
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


class StatefulVectorConditionedAdapter:
    """Local, bounded stateful wrapper around a vector-conditioned cortex.

    The underlying cortex remains the byte predictor. This wrapper adds a
    low-dimensional state and a state-to-byte associative memory. The state is
    initialized from the predictor vector, then updated from each generated or
    observed byte using a surprise-gated Hebbian rule. Context resets preserve
    this memory; ``clear_memory`` provides the one-shot control.
    """

    def __init__(
        self,
        cortex: VectorConditionedCortex,
        config: StatefulVectorConditionedConfig | None = None,
    ) -> None:
        self.cortex = cortex
        self.config = config or StatefulVectorConditionedConfig()
        self._projection: np.ndarray | None = None
        self._byte_features: np.ndarray | None = None
        self._vector_dimensions: int | None = None
        self.state = np.zeros(self.config.state_dimensions, dtype=np.float64)
        self.retained_state = np.zeros(
            self.config.state_dimensions, dtype=np.float64
        )
        self.weights = np.zeros(
            (self.config.state_dimensions, 256), dtype=np.float64
        )
        self._generation_updates = False

    def _ensure_projection(self, vector_dimensions: int) -> None:
        if (
            self._projection is not None
            and self._vector_dimensions == vector_dimensions
        ):
            return
        rng = np.random.default_rng(self.config.seed + vector_dimensions)
        self._projection = rng.normal(
            0.0,
            1.0 / math.sqrt(vector_dimensions),
            size=(self.config.state_dimensions, vector_dimensions),
        )
        frequencies = rng.uniform(-math.pi, math.pi, size=self.config.state_dimensions)
        phases = rng.uniform(-math.pi, math.pi, size=self.config.state_dimensions)
        symbols = np.arange(256, dtype=np.float64)[:, None]
        self._byte_features = np.sin(symbols * frequencies[None, :] + phases[None, :])
        self._vector_dimensions = vector_dimensions

    def _state_from_vector(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(vector)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("vector must be a nonempty one-dimensional array")
        if np.iscomplexobj(values) or not np.issubdtype(values.dtype, np.number):
            raise TypeError("vector must contain only real numeric values")
        if not np.all(np.isfinite(values)):
            raise ValueError("vector must contain only finite values")
        # Parse the same little-endian float32 payload that is sent to the cortex.
        frame = vector_input_frame(values)
        dense = np.frombuffer(frame[len(_FRAME_MAGIC) + 4 :], dtype="<f4")
        self._ensure_projection(int(dense.size))
        assert self._projection is not None
        projected = self._projection @ np.tanh(dense.astype(np.float64) / 3.0)
        norm = float(np.linalg.norm(projected))
        if norm > 1.0:
            projected = projected / norm
        return np.clip(projected, -1.0, 1.0)

    def _surprise_gate(self, surprise_bits: float) -> float:
        value = (
            surprise_bits - self.config.surprise_gate_threshold
        ) / self.config.surprise_gate_scale
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def reset_context(self) -> None:
        """Reset only the underlying byte episode, retaining adapter memory."""

        self.cortex.reset_episode()

    def clear_memory(self) -> None:
        """Clear persistent state and associative weights, not byte context."""

        self.state.fill(0.0)
        self.retained_state.fill(0.0)
        self.weights.fill(0.0)

    def set_generation_updates(self, enabled: bool) -> None:
        """Enable or disable local updates while generating."""

        self._generation_updates = bool(enabled)

    def reset_episode(self) -> None:
        """Reset both episode context and adapter memory for compatibility."""

        self.reset_context()
        self.clear_memory()

    def prime(
        self,
        vector: Sequence[float] | np.ndarray,
        *,
        clear_memory: bool = False,
    ) -> bytes:
        """Install a predictor vector while optionally clearing retained memory.

        A repeated ``prime`` on the same adapter starts a new byte episode but
        retains the associative memory by default. This is the multi-turn path;
        ``clear_memory=True`` is the one-shot control.
        """

        frame = vector_input_frame(vector)
        self.reset_context()
        if clear_memory:
            self.clear_memory()
        vector_state = self._state_from_vector(vector)
        if np.any(self.retained_state):
            vector_state = 0.5 * (vector_state + self.retained_state)
        self.state = np.clip(vector_state, -1.0, 1.0)
        self.cortex.observe_chunk_probabilities(
            np.frombuffer(frame, dtype=np.uint8), learn=False
        )
        return frame

    def predict_next_distribution(self) -> np.ndarray:
        base = _normalized_distribution(self.cortex.predict_next_distribution())
        if not np.any(self.weights):
            return base
        logits = self.config.modulation_strength * (self.state @ self.weights)
        logits = np.clip(logits, -12.0, 12.0)
        log_modulated = np.log(base + np.finfo(np.float64).tiny) + logits
        return _normalized_distribution(np.exp(log_modulated - np.max(log_modulated)))

    def _update_for_observation(self, byte_id: int, probability: float) -> None:
        assert self._byte_features is not None
        probability = max(float(probability), np.finfo(np.float64).tiny)
        surprise = -math.log2(probability)
        gate = self._surprise_gate(surprise)
        target = self._byte_features[byte_id]
        self.state *= self.config.state_decay
        self.state += self.config.learning_rate * gate * (target - self.state)
        self.state = np.clip(self.state, -1.0, 1.0)
        self.retained_state *= self.config.state_decay
        self.retained_state += self.config.learning_rate * gate * (
            target - self.retained_state
        )
        self.retained_state = np.clip(self.retained_state, -1.0, 1.0)
        self.weights *= 1.0 - self.config.weight_decay
        self.weights[:, byte_id] += self.config.learning_rate * gate * self.state
        self.weights = np.clip(
            self.weights, -self.config.weight_clip, self.config.weight_clip
        )

    def advance_observation(
        self,
        byte_id: int,
        *,
        learn: bool = False,
        adapt: bool | None = None,
    ) -> float:
        byte_id = int(byte_id)
        if not 0 <= byte_id < 256:
            raise ValueError("byte_id must be in [0, 255]")
        law = self.predict_next_distribution()
        probability = max(float(law[byte_id]), np.finfo(np.float64).tiny)
        if adapt is None:
            adapt = learn or (
                self._generation_updates and self.config.update_on_generated
            )
        if adapt:
            self._update_for_observation(byte_id, probability)
        self.cortex.advance_observation(byte_id, learn=learn)
        return probability

    def observe_chunk_probabilities(
        self,
        byte_ids: np.ndarray,
        *,
        learn: bool,
        adapt: bool | None = None,
    ) -> np.ndarray:
        values = np.asarray(byte_ids, dtype=np.uint8).reshape(-1)
        return np.asarray(
            [
                self.advance_observation(int(value), learn=learn, adapt=adapt)
                for value in values
            ],
            dtype=np.float64,
        )

    def observe_context(
        self, context: bytes | bytearray | memoryview, *, adapt: bool = True
    ) -> np.ndarray:
        """Observe a prior turn without enabling base-cortex learning."""

        return self.observe_chunk_probabilities(
            np.frombuffer(bytes(context), dtype=np.uint8), learn=False, adapt=adapt
        )

    def observe_turn(
        self,
        turn: bytes | bytearray | memoryview,
        *,
        adapt: bool = True,
        learn: bool = False,
    ) -> np.ndarray:
        """Observe an actual or generated turn through the local transducer."""

        return self.observe_chunk_probabilities(
            np.frombuffer(bytes(turn), dtype=np.uint8), learn=learn, adapt=adapt
        )

    def generate(
        self,
        *,
        horizon: int,
        uniform_draws: Sequence[float] | np.ndarray | None = None,
        update_generated: bool | None = None,
        reset_after: bool = False,
    ) -> VectorConditionedGeneration:
        """Generate one turn while optionally retaining memory for the next turn."""

        if horizon < 1:
            raise ValueError("horizon must be positive")
        draws: np.ndarray | None = None
        if uniform_draws is not None:
            draws = np.asarray(uniform_draws, dtype=np.float64)
            if (
                draws.shape != (horizon,)
                or not np.all(np.isfinite(draws))
                or np.any((draws < 0.0) | (draws >= 1.0))
            ):
                raise ValueError("uniform_draws must have shape (horizon,) in [0, 1)")
        previous_generation_updates = self._generation_updates
        if update_generated is not None:
            self._generation_updates = bool(update_generated)
        generated = bytearray()
        digest = hashlib.sha256()
        try:
            for position in range(horizon):
                law = _normalized_distribution(self.predict_next_distribution())
                digest.update(law.tobytes())
                if draws is None:
                    symbol = int(np.argmax(law))
                else:
                    symbol = min(
                        int(
                            np.searchsorted(
                                np.cumsum(law), draws[position], side="right"
                            )
                        ),
                        255,
                    )
                generated.append(symbol)
                self.advance_observation(symbol, learn=False)
        finally:
            self._generation_updates = previous_generation_updates
            if reset_after:
                self.reset_context()
        return VectorConditionedGeneration(bytes(generated), digest.hexdigest())

    def end_turn(self) -> None:
        """Reset the byte episode while retaining persistent adapter memory."""

        self.reset_context()


@dataclass(frozen=True)
class StatefulVectorConditionedTurn:
    """One causal turn in a multi-turn stateful generation session.

    ``context`` is observed before the turn vector is installed. ``observed``
    is optional bytes from the completed turn; when present it is observed after
    generation and can seed the next turn's persistent state.
    """

    vector: Sequence[float] | np.ndarray
    context: bytes | bytearray | memoryview = b""
    observed: bytes | bytearray | memoryview | None = None
    horizon: int = 1
    clear_memory: bool = False
    adapt_context: bool = True
    adapt_observed: bool = True
    update_generated: bool = True


@dataclass(frozen=True)
class VectorConditionedGeneration:
    data: bytes
    probability_sha256: str


def vector_input_frame(vector: Sequence[float] | np.ndarray) -> bytes:
    """Return the canonical dense float32 input frame for one turn vector.

    Values are copied to little-endian float32 because that is the predictor
    interface used at deployment. The typed marker and dimension make frame
    boundaries unambiguous without hashing or sparsifying any coordinates.
    """

    values = np.asarray(vector)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("vector must be a nonempty one-dimensional array")
    if values.size > 0xFFFFFFFF:
        raise ValueError("vector has too many dimensions")
    if np.iscomplexobj(values) or not np.issubdtype(values.dtype, np.number):
        raise TypeError("vector must contain only real numeric values")
    dense = np.asarray(values, dtype="<f4")
    if not np.all(np.isfinite(dense)):
        raise ValueError("vector must contain only finite values")
    payload = dense.tobytes(order="C")
    return _FRAME_MAGIC + struct.pack("<I", dense.size) + payload


def _as_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    data = bytes(payload)
    if not data:
        raise ValueError("target turn must be nonempty")
    return data


def _normalized_distribution(values: Any) -> np.ndarray:
    distribution = np.asarray(values, dtype=np.float64)
    if (
        distribution.shape != (256,)
        or not np.all(np.isfinite(distribution))
        or np.any(distribution < 0.0)
    ):
        raise ValueError("cortex distribution must be finite and 256-way")
    total = float(distribution.sum())
    if total <= 0.0:
        raise ValueError("cortex distribution has no probability mass")
    return distribution / total


def _observe_vector_frame(
    cortex: VectorConditionedCortex, vector: Sequence[float] | np.ndarray
) -> bytes:
    frame = vector_input_frame(vector)
    cortex.observe_chunk_probabilities(
        np.frombuffer(frame, dtype=np.uint8), learn=False
    )
    return frame


def prime_with_vector(
    cortex: VectorConditionedCortex,
    vector: Sequence[float] | np.ndarray,
) -> bytes:
    """Reset and install a raw predictor vector as read-only episode context."""

    cortex.reset_episode()
    return _observe_vector_frame(cortex, vector)


def fit_vector_conditioned_turn(
    cortex: VectorConditionedCortex,
    vector: Sequence[float] | np.ndarray,
    target: bytes | bytearray | memoryview,
) -> VectorConditionedFitDiagnostics:
    """Teach ``target`` as the continuation of ``vector`` in the base cortex.

    The vector is causal context only (``learn=False``). Each actual target
    byte is scored and learned by the cortex (``learn=True``), so surprise is
    computed from the logged example exactly where base plasticity occurs.
    """

    data = _as_bytes(target)
    frame = prime_with_vector(cortex, vector)
    observed = np.asarray(
        cortex.observe_chunk_probabilities(
            np.frombuffer(data, dtype=np.uint8), learn=True
        ),
        dtype=np.float64,
    )
    cortex.reset_episode()
    if observed.shape != (len(data),):
        raise ValueError("cortex returned one probability per target byte")
    # A newborn cortex can report NaN when no predictive law existed yet.
    probabilities = np.where(np.isfinite(observed), observed, 1.0 / 256.0)
    probabilities = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    surprise = -np.log2(probabilities)
    return VectorConditionedFitDiagnostics(
        vector_dimensions=int(np.asarray(vector).size),
        conditioning_bytes=len(frame),
        target_bytes=len(data),
        total_surprise_bits=float(surprise.sum()),
        mean_surprise_bits=float(surprise.mean()),
        minimum_observed_probability=float(probabilities.min()),
    )


def score_vector_conditioned_turn(
    cortex: VectorConditionedCortex,
    vector: Sequence[float] | np.ndarray,
    target: bytes | bytearray | memoryview,
) -> VectorConditionedFitDiagnostics:
    """Teacher-force a vector/turn pair without changing base anatomy."""

    data = _as_bytes(target)
    frame = prime_with_vector(cortex, vector)
    probabilities: list[float] = []
    for symbol in data:
        law = _normalized_distribution(cortex.predict_next_distribution())
        probabilities.append(max(float(law[symbol]), np.finfo(np.float64).tiny))
        cortex.advance_observation(symbol, learn=False)
    cortex.reset_episode()
    surprise = -np.log2(np.asarray(probabilities, dtype=np.float64))
    return VectorConditionedFitDiagnostics(
        vector_dimensions=int(np.asarray(vector).size),
        conditioning_bytes=len(frame),
        target_bytes=len(data),
        total_surprise_bits=float(surprise.sum()),
        mean_surprise_bits=float(surprise.mean()),
        minimum_observed_probability=float(min(probabilities)),
    )


def generate_from_vector(
    cortex: VectorConditionedCortex,
    vector: Sequence[float] | np.ndarray,
    *,
    horizon: int,
    uniform_draws: Sequence[float] | np.ndarray | None = None,
) -> VectorConditionedGeneration:
    """Generate bytes from the base with the predictor vector as its only input."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    draws: np.ndarray | None = None
    if uniform_draws is not None:
        draws = np.asarray(uniform_draws, dtype=np.float64)
        if (
            draws.shape != (horizon,)
            or not np.all(np.isfinite(draws))
            or np.any((draws < 0.0) | (draws >= 1.0))
        ):
            raise ValueError("uniform_draws must have shape (horizon,) in [0, 1)")
    prime_with_vector(cortex, vector)
    generated = bytearray()
    digest = hashlib.sha256()
    for position in range(horizon):
        law = _normalized_distribution(cortex.predict_next_distribution())
        digest.update(law.tobytes())
        if draws is None:
            symbol = int(np.argmax(law))
        else:
            symbol = min(
                int(np.searchsorted(np.cumsum(law), draws[position], side="right")),
                255,
            )
        generated.append(symbol)
        cortex.advance_observation(symbol, learn=False)
    cortex.reset_episode()
    return VectorConditionedGeneration(bytes(generated), digest.hexdigest())


def generate_from_vector_after_context(
    cortex: VectorConditionedCortex,
    vector: Sequence[float] | np.ndarray,
    context: bytes | bytearray | memoryview,
    *,
    horizon: int,
    uniform_draws: Sequence[float] | np.ndarray | None = None,
) -> VectorConditionedGeneration:
    """Generate from a vector after a read-only causal context episode.

    This is the context-matched control for the stateful transducer: the base
    sees the same context and vector frames, but receives no adapter memory.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")
    draws: np.ndarray | None = None
    if uniform_draws is not None:
        draws = np.asarray(uniform_draws, dtype=np.float64)
        if (
            draws.shape != (horizon,)
            or not np.all(np.isfinite(draws))
            or np.any((draws < 0.0) | (draws >= 1.0))
        ):
            raise ValueError("uniform_draws must have shape (horizon,) in [0, 1)")
    cortex.reset_episode()
    context_bytes = bytes(context)
    if context_bytes:
        cortex.observe_chunk_probabilities(
            np.frombuffer(context_bytes, dtype=np.uint8), learn=False
        )
    _observe_vector_frame(cortex, vector)
    generated = bytearray()
    digest = hashlib.sha256()
    try:
        for position in range(horizon):
            law = _normalized_distribution(cortex.predict_next_distribution())
            digest.update(law.tobytes())
            if draws is None:
                symbol = int(np.argmax(law))
            else:
                symbol = min(
                    int(
                        np.searchsorted(
                            np.cumsum(law), draws[position], side="right"
                        )
                    ),
                    255,
                )
            generated.append(symbol)
            cortex.advance_observation(symbol, learn=False)
    finally:
        cortex.reset_episode()
    return VectorConditionedGeneration(bytes(generated), digest.hexdigest())


def fit_stateful_vector_conditioned_turn(
    cortex: VectorConditionedCortex,
    vector: Sequence[float] | np.ndarray,
    target: bytes | bytearray | memoryview,
    *,
    config: StatefulVectorConditionedConfig | None = None,
) -> VectorConditionedFitDiagnostics:
    """Fit a target with bounded local state updates in the transducer."""

    data = _as_bytes(target)
    adapter = StatefulVectorConditionedAdapter(cortex, config)
    frame = adapter.prime(vector)
    observed = adapter.observe_chunk_probabilities(
        np.frombuffer(data, dtype=np.uint8), learn=True
    )
    adapter.reset_episode()
    probabilities = np.where(np.isfinite(observed), observed, 1.0 / 256.0)
    probabilities = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    surprise = -np.log2(probabilities)
    return VectorConditionedFitDiagnostics(
        vector_dimensions=int(np.asarray(vector).size),
        conditioning_bytes=len(frame),
        target_bytes=len(data),
        total_surprise_bits=float(surprise.sum()),
        mean_surprise_bits=float(surprise.mean()),
        minimum_observed_probability=float(probabilities.min()),
    )


def score_stateful_vector_conditioned_turn(
    cortex: VectorConditionedCortex,
    vector: Sequence[float] | np.ndarray,
    target: bytes | bytearray | memoryview,
    *,
    config: StatefulVectorConditionedConfig | None = None,
) -> VectorConditionedFitDiagnostics:
    """Score a vector/turn pair through the stateful transducer without fitting."""

    data = _as_bytes(target)
    adapter = StatefulVectorConditionedAdapter(cortex, config)
    frame = adapter.prime(vector)
    probabilities: list[float] = []
    for symbol in data:
        law = _normalized_distribution(adapter.predict_next_distribution())
        probabilities.append(max(float(law[symbol]), np.finfo(np.float64).tiny))
        adapter.advance_observation(symbol, learn=False, adapt=False)
    adapter.reset_episode()
    surprise = -np.log2(np.asarray(probabilities, dtype=np.float64))
    return VectorConditionedFitDiagnostics(
        vector_dimensions=int(np.asarray(vector).size),
        conditioning_bytes=len(frame),
        target_bytes=len(data),
        total_surprise_bits=float(surprise.sum()),
        mean_surprise_bits=float(surprise.mean()),
        minimum_observed_probability=float(min(probabilities)),
    )


def generate_stateful_from_vector(
    cortex: VectorConditionedCortex,
    vector: Sequence[float] | np.ndarray,
    *,
    horizon: int,
    config: StatefulVectorConditionedConfig | None = None,
    uniform_draws: Sequence[float] | np.ndarray | None = None,
) -> VectorConditionedGeneration:
    """Generate through a bounded stateful vector-conditioned transducer."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    draws: np.ndarray | None = None
    if uniform_draws is not None:
        draws = np.asarray(uniform_draws, dtype=np.float64)
        if (
            draws.shape != (horizon,)
            or not np.all(np.isfinite(draws))
            or np.any((draws < 0.0) | (draws >= 1.0))
        ):
            raise ValueError("uniform_draws must have shape (horizon,) in [0, 1)")
    adapter = StatefulVectorConditionedAdapter(cortex, config)
    adapter.prime(vector)
    return adapter.generate(
        horizon=horizon,
        uniform_draws=uniform_draws,
        update_generated=adapter.config.update_on_generated,
        reset_after=True,
    )


def generate_stateful_after_context(
    cortex: VectorConditionedCortex,
    vector: Sequence[float] | np.ndarray,
    context: bytes | bytearray | memoryview,
    *,
    horizon: int,
    config: StatefulVectorConditionedConfig | None = None,
    uniform_draws: Sequence[float] | np.ndarray | None = None,
    adapt_context: bool = True,
    clear_memory_after_context: bool = False,
    update_generated: bool | None = None,
) -> VectorConditionedGeneration:
    """Generate after a causal context episode with optional retained memory.

    The context is observed with cortex learning disabled. When
    ``adapt_context`` is true, only the local adapter updates; the byte cortex
    therefore cannot explain a retained-memory effect through episodic writes.
    The adapter context is reset before the predictor vector is primed, while
    associative memory is retained unless ``clear_memory_after_context`` is set.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")
    draws: np.ndarray | None = None
    if uniform_draws is not None:
        draws = np.asarray(uniform_draws, dtype=np.float64)
        if (
            draws.shape != (horizon,)
            or not np.all(np.isfinite(draws))
            or np.any((draws < 0.0) | (draws >= 1.0))
        ):
            raise ValueError("uniform_draws must have shape (horizon,) in [0, 1)")
    adapter = StatefulVectorConditionedAdapter(cortex, config)
    neutral = np.zeros_like(np.asarray(vector, dtype=np.float64))
    adapter.prime(neutral, clear_memory=True)
    adapter.observe_context(bytes(context), adapt=adapt_context)
    if clear_memory_after_context:
        adapter.clear_memory()
    adapter.prime(vector)
    return adapter.generate(
        horizon=horizon,
        uniform_draws=uniform_draws,
        update_generated=(
            adapter.config.update_on_generated
            if update_generated is None
            else update_generated
        ),
        reset_after=True,
    )


def generate_stateful_multi_turn(
    cortex: VectorConditionedCortex,
    turns: Sequence[StatefulVectorConditionedTurn],
    *,
    config: StatefulVectorConditionedConfig | None = None,
    uniform_draws: Sequence[
        Sequence[float] | np.ndarray | None
    ]
    | None = None,
) -> tuple[VectorConditionedGeneration, ...]:
    """Generate a causal sequence while retaining local memory across turns.

    Each turn observes its context, installs its predictor vector, generates,
    and then optionally observes its completed bytes. The byte episode resets
    between turns, but persistent state and associative weights remain unless a
    turn explicitly requests ``clear_memory``.
    """

    if not turns:
        raise ValueError("at least one turn is required")
    if uniform_draws is not None and len(uniform_draws) != len(turns):
        raise ValueError("uniform_draws must contain one entry per turn")
    adapter = StatefulVectorConditionedAdapter(cortex, config)
    outputs: list[VectorConditionedGeneration] = []
    for index, turn in enumerate(turns):
        draws = None if uniform_draws is None else uniform_draws[index]
        adapter.prime(turn.vector, clear_memory=turn.clear_memory)
        adapter.observe_context(turn.context, adapt=turn.adapt_context)
        generated = adapter.generate(
            horizon=turn.horizon,
            uniform_draws=draws,
            update_generated=turn.update_generated,
            reset_after=False,
        )
        outputs.append(generated)
        if turn.observed is not None:
            adapter.observe_turn(
                turn.observed,
                adapt=turn.adapt_observed,
                learn=False,
            )
        adapter.end_turn()
    return tuple(outputs)
