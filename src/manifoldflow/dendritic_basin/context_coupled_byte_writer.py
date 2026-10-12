"""Bounded local writer with persistent-state / production-phase conjunctions.

The writer is intentionally ignorant of conversations, roles, and even bytes.
It consumes integer symbols from an alphabet of at most 256 members and sparse,
caller-supplied integer IDs for a persistent state.  A text adapter can use byte
values; an audio, vision, or control adapter can use quantized event IDs.

This is the reusable mechanism isolated by the REPORT2 sandbox.  Learning is a
normalized local delta rule.  Production phase is represented only by suffixes
of symbols already observed or emitted.  The ``coupled`` condition adds sparse
conjunctions between that evolving local context and a state held fixed for the
episode.  Its controls are:

``local``
    Local suffix features only.
``additive``
    Local and persistent-state features, without conjunctions.
``additive_wide``
    The additive model plus pure-local and pure-state replicas.  It has exactly
    the coupled model's raw active-feature count and the same weight allocation,
    but still has no state / local cross-term.
``coupled``
    Additive features plus every local / persistent-state conjunction.

All feature addresses are projected into one fixed-size table.  Repeated raw
features and hash collisions retain their multiplicity in both prediction and
learning.  Each ``fit_episode`` call starts from its explicit prefix, preventing
an accidental transition between adjacent training responses.

``fit_base_anchored_episode`` is the direct receiver-training path: a caller
provides one causal base probability vector per observed symbol, and this table
learns the normalized softmax residual by cross-entropy. The matching inference
API applies that residual without expanding the caller's output support.
"""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

WriterKind = Literal["local", "additive", "additive_wide", "coupled"]
SymbolSequence = bytes | bytearray | memoryview | Sequence[int] | np.ndarray
FeatureSequence = Sequence[int] | np.ndarray

WRITER_KINDS: tuple[WriterKind, ...] = (
    "local",
    "additive",
    "additive_wide",
    "coupled",
)

_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class ContextCoupledEpisode:
    """One independently bounded teacher-forced training episode."""

    target: SymbolSequence
    persistent_feature_ids: FeatureSequence
    prefix: SymbolSequence = b""


@dataclass(frozen=True)
class ContextCoupledWriterSnapshot:
    """Complete detached state for :class:`ContextCoupledByteWriter`."""

    version: int
    kind: WriterKind
    alphabet_size: int
    table_bits: int
    learning_rate: float
    local_orders: tuple[int, ...]
    max_persistent_features: int
    episodes_seen: int
    symbols_seen: int
    weights: np.ndarray


@dataclass(frozen=True)
class BaseAnchoredFitDiagnostics:
    """Loss and update accounting for one base-anchored episode."""

    symbols: int
    symbol_updates: int
    raw_feature_activations: int
    unique_feature_row_updates: int
    scalar_weight_slots_touched: int
    base_nll_nats: float
    anchored_nll_nats: float
    residual_improvement_nats: float
    mean_base_nll_nats: float
    mean_anchored_nll_nats: float
    mean_residual_improvement_nats: float
    maximum_absolute_update: float
    minimum_target_base_probability: float | None


def _integer_tuple(values: object, *, name: str) -> tuple[int, ...]:
    """Copy a one-dimensional integer iterable without lossy coercion."""

    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        items = values.tolist()
    else:
        try:
            items = list(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"{name} must be an iterable of integers") from exc
    result: list[int] = []
    for item in items:
        try:
            result.append(operator.index(item))
        except TypeError as exc:
            raise TypeError(f"{name} must contain only integers") from exc
    return tuple(result)


def likelihood_ratio_correction(
    base_distribution: Sequence[float] | np.ndarray,
    local_distribution: Sequence[float] | np.ndarray,
    coupled_distribution: Sequence[float] | np.ndarray,
    *,
    beta: float,
    log_ratio_clip: float = 2.0,
) -> np.ndarray:
    """Apply ``base * exp(beta * clip(log(qc / q0)))`` and normalize.

    ``q0`` and ``qc`` normally come from allocation-matched ``local`` and
    ``coupled`` writers trained on the same episodes.  At ``beta == 0`` this is
    an exact bypass: a value-for-value copy with the base input's dtype is
    returned before either route distribution is inspected.
    """

    base_input = np.asarray(base_distribution)
    if base_input.ndim != 1 or not 1 <= len(base_input) <= 256:
        raise ValueError("base_distribution must have between 1 and 256 entries")
    base = np.asarray(base_input, dtype=np.float64)
    if not np.all(np.isfinite(base)) or np.any(base < 0.0):
        raise ValueError("base_distribution must contain finite probabilities")
    if not np.isclose(float(base.sum()), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("base_distribution must sum to one")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    if not np.isfinite(log_ratio_clip) or log_ratio_clip <= 0.0:
        raise ValueError("log_ratio_clip must be finite and positive")
    if beta == 0.0:
        return np.array(base_input, copy=True)

    alphabet_size = len(base)
    local = _normalized_distribution(
        local_distribution,
        alphabet_size=alphabet_size,
        name="local_distribution",
    )
    coupled = _normalized_distribution(
        coupled_distribution,
        alphabet_size=alphabet_size,
        name="coupled_distribution",
    )
    tiny = np.finfo(np.float64).tiny
    log_ratio = np.clip(
        np.log(np.maximum(coupled, tiny)) - np.log(np.maximum(local, tiny)),
        -log_ratio_clip,
        log_ratio_clip,
    )
    support = base > 0.0
    logits = np.full(alphabet_size, -np.inf, dtype=np.float64)
    logits[support] = np.log(base[support]) + beta * log_ratio[support]
    pivot = float(np.max(logits[support]))
    corrected = np.zeros(alphabet_size, dtype=np.float64)
    corrected[support] = np.exp(logits[support] - pivot)
    corrected /= corrected.sum()
    return corrected


def _normalized_distribution(
    values: Sequence[float] | np.ndarray,
    *,
    alphabet_size: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (alphabet_size,):
        raise ValueError(f"{name} must have shape ({alphabet_size},)")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return np.array(vector / total, dtype=np.float64, copy=True)


def _strict_probability_distribution(
    values: Sequence[float] | np.ndarray,
    *,
    alphabet_size: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a probability vector while retaining its original representation."""

    original = np.asarray(values)
    if original.shape != (alphabet_size,):
        raise ValueError(f"{name} must have shape ({alphabet_size},)")
    distribution = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(distribution)) or np.any(distribution < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative probabilities")
    if not np.isclose(
        float(distribution.sum()),
        1.0,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"{name} must sum to one")
    return original, np.array(distribution, dtype=np.float64, copy=True)


def _base_anchored_softmax(
    base: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """Normalize ``base * exp(residual)`` without expanding base support."""

    if base.shape != residual.shape:
        raise ValueError("base and residual distributions must have matching shapes")
    if not np.all(np.isfinite(residual)):
        raise ValueError("writer residual contains non-finite values")
    support = base > 0.0
    if not np.any(support):
        raise ValueError("base distribution must have positive support")
    logits = np.full(base.shape, -np.inf, dtype=np.float64)
    logits[support] = np.log(base[support]) + residual[support]
    pivot = float(np.max(logits[support]))
    probabilities = np.zeros(base.shape, dtype=np.float64)
    probabilities[support] = np.maximum(
        np.exp(logits[support] - pivot),
        np.finfo(np.float64).tiny,
    )
    probabilities /= float(probabilities.sum())
    return probabilities


def _strict_probability_matrix(
    values: Sequence[Sequence[float] | np.ndarray] | np.ndarray,
    *,
    rows: int,
    alphabet_size: int,
    name: str,
) -> np.ndarray:
    """Copy and validate one normalized probability vector per target symbol."""

    try:
        supplied = list(values)
    except TypeError as exc:
        raise TypeError(
            f"{name} must contain one distribution per target symbol"
        ) from exc
    if len(supplied) != rows:
        raise ValueError(f"{name} must contain exactly {rows} rows")
    matrix = np.empty((rows, alphabet_size), dtype=np.float64)
    for index, row in enumerate(supplied):
        _, matrix[index] = _strict_probability_distribution(
            row,
            alphabet_size=alphabet_size,
            name=f"{name}[{index}]",
        )
    return matrix


class ContextCoupledByteWriter:
    """Sparse normalized-delta writer with deterministic bounded storage."""

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        kind: WriterKind,
        *,
        alphabet_size: int = 256,
        table_bits: int = 14,
        learning_rate: float = 0.55,
        local_orders: Sequence[int] = (1, 2, 4, 8),
        max_persistent_features: int = 128,
    ) -> None:
        if kind not in WRITER_KINDS:
            raise ValueError(f"unknown writer kind: {kind!r}")
        if not 2 <= alphabet_size <= 256:
            raise ValueError("alphabet_size must be between 2 and 256")
        if not 1 <= table_bits <= 20:
            raise ValueError("table_bits must be between 1 and 20")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        orders = _integer_tuple(local_orders, name="local_orders")
        if not orders or any(order < 1 or order > 64 for order in orders):
            raise ValueError("local_orders must contain values between 1 and 64")
        if len(set(orders)) != len(orders):
            raise ValueError("local_orders must not contain duplicates")
        if max_persistent_features < 1:
            raise ValueError("max_persistent_features must be positive")

        self.kind = kind
        self.alphabet_size = int(alphabet_size)
        self.table_bits = int(table_bits)
        self.table_size = 1 << self.table_bits
        self._table_mask = self.table_size - 1
        self.learning_rate = float(learning_rate)
        self.local_orders = tuple(orders)
        self.max_persistent_features = int(max_persistent_features)
        self.episodes_seen = 0
        self.symbols_seen = 0
        self._weights = np.zeros(
            (self.table_size, self.alphabet_size), dtype=np.float32
        )

    @property
    def parameter_count(self) -> int:
        """Allocated scalar weights; identical across all writer conditions."""

        return int(self._weights.size)

    @property
    def allocated_bytes(self) -> int:
        """Bytes occupied by the bounded weight table."""

        return int(self._weights.nbytes)

    @property
    def maximum_local_order(self) -> int:
        return max(self.local_orders)

    def raw_active_feature_count(self, persistent_feature_count: int) -> int:
        """Raw active count before collisions for a given state width."""

        count = operator.index(persistent_feature_count)
        if count < 0 or count > self.max_persistent_features:
            raise ValueError("persistent_feature_count is outside configured bounds")
        local_count = 1 + len(self.local_orders)
        if self.kind == "local":
            return local_count
        if count == 0:
            raise ValueError("contextual writer kinds require persistent features")
        if self.kind == "additive":
            return local_count + count
        return local_count + count + local_count * count

    def local_feature_ids(self, prefix: SymbolSequence) -> np.ndarray:
        """Hash multiscale suffixes of observed/emitted symbols only."""

        symbols = self._symbols(prefix, name="prefix")
        return self._local_feature_ids_from_tuple(symbols)

    def raw_feature_ids(
        self,
        prefix: SymbolSequence,
        persistent_feature_ids: FeatureSequence,
        *,
        freeze_joint: bool = False,
    ) -> np.ndarray:
        """Return projected raw addresses, retaining duplicates/collisions."""

        symbols = self._symbols(prefix, name="prefix")
        state = self._persistent_features(persistent_feature_ids)
        return self._raw_feature_ids_from_tuples(symbols, state, freeze_joint)

    def features(
        self,
        prefix: SymbolSequence,
        persistent_feature_ids: FeatureSequence,
        *,
        freeze_joint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return unique row IDs and collision multiplicities."""

        raw = self.raw_feature_ids(
            prefix,
            persistent_feature_ids,
            freeze_joint=freeze_joint,
        )
        indexes, counts = np.unique(raw, return_counts=True)
        return indexes, counts.astype(np.float32)

    def scores(
        self,
        prefix: SymbolSequence,
        persistent_feature_ids: FeatureSequence,
        *,
        freeze_joint: bool = False,
    ) -> np.ndarray:
        """Return next-symbol scores without modifying learned state."""

        indexes, multiplicities = self.features(
            prefix,
            persistent_feature_ids,
            freeze_joint=freeze_joint,
        )
        return np.array(multiplicities @ self._weights[indexes], copy=True)

    def base_anchored_distribution(
        self,
        base_distribution: Sequence[float] | np.ndarray,
        prefix: SymbolSequence,
        persistent_feature_ids: FeatureSequence,
        *,
        gain: float = 1.0,
        residual_clip: float | None = None,
        freeze_joint: bool = False,
    ) -> np.ndarray:
        """Apply this writer as a support-preserving residual to a caller base.

        The returned law is proportional to ``base * exp(residual)``. ``gain``
        scales the writer score, and ``residual_clip`` optionally clips that
        scaled logit correction to a symmetric finite bound. At exactly zero
        gain, this returns a value-for-value copy with the caller's dtype before
        inspecting prefix, persistent IDs, or writer scores.
        """

        base_input, base = _strict_probability_distribution(
            base_distribution,
            alphabet_size=self.alphabet_size,
            name="base_distribution",
        )
        if not np.isfinite(gain) or gain < 0.0:
            raise ValueError("gain must be finite and non-negative")
        if residual_clip is not None and (
            not np.isfinite(residual_clip) or residual_clip <= 0.0
        ):
            raise ValueError("residual_clip must be finite and positive")
        if gain == 0.0:
            return np.array(base_input, copy=True)

        residual = self.scores(
            prefix,
            persistent_feature_ids,
            freeze_joint=freeze_joint,
        ).astype(np.float64)
        scaled = float(gain) * residual
        if residual_clip is not None:
            scaled = np.clip(scaled, -residual_clip, residual_clip)
        if not np.all(np.isfinite(scaled)):
            raise ValueError(
                "scaled writer residual is non-finite; configure residual_clip"
            )
        return _base_anchored_softmax(base, scaled)

    def distribution(
        self,
        prefix: SymbolSequence,
        persistent_feature_ids: FeatureSequence,
        *,
        freeze_joint: bool = False,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Softmax the local scores for fusion with another predictor."""

        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        scores = self.scores(
            prefix,
            persistent_feature_ids,
            freeze_joint=freeze_joint,
        ).astype(np.float64)
        scores = (scores - float(np.max(scores))) / float(temperature)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()
        return probabilities

    def fit_episode(
        self,
        target: SymbolSequence,
        persistent_feature_ids: FeatureSequence,
        *,
        prefix: SymbolSequence = b"",
    ) -> None:
        """Teacher-force one episode, resetting production history at its prefix.

        Inputs are copied and validated before the first update.  The target is
        never used as a feature ahead of its observation; each emitted/observed
        symbol becomes local context only for subsequent positions.
        """

        target_symbols = self._symbols(target, name="target")
        prefix_symbols = self._symbols(prefix, name="prefix")
        state = self._persistent_features(persistent_feature_ids)
        history = list(prefix_symbols[-self.maximum_local_order :])
        for target_symbol in target_symbols:
            indexes, multiplicities = self._features_from_tuples(tuple(history), state)
            prediction = multiplicities @ self._weights[indexes]
            error = -prediction
            error[target_symbol] += 1.0
            denominator = float(multiplicities @ multiplicities)
            update = (
                (self.learning_rate / denominator)
                * multiplicities[:, None]
                * error[None, :]
            )
            self._weights[indexes] += update
            history.append(target_symbol)
            del history[: -self.maximum_local_order]
        self.episodes_seen += 1
        self.symbols_seen += len(target_symbols)

    def fit_base_anchored_episode(
        self,
        target: SymbolSequence,
        base_distributions: Sequence[Sequence[float] | np.ndarray] | np.ndarray,
        persistent_feature_ids: FeatureSequence,
        *,
        prefix: SymbolSequence = b"",
        freeze_joint: bool = False,
    ) -> BaseAnchoredFitDiagnostics:
        """Fit a softmax residual against one causal base law per target symbol.

        Row ``j`` of ``base_distributions`` must have been computed without
        observing target symbol ``j``. This method cannot certify an external
        caller's causal provenance, but it enforces alignment and advances its
        local suffix only after the corresponding observed symbol is updated.
        Every input row and every observed symbol's base support are validated
        before the first weight update.
        """

        target_symbols = self._symbols(target, name="target")
        prefix_symbols = self._symbols(prefix, name="prefix")
        state = self._persistent_features(persistent_feature_ids)
        bases = _strict_probability_matrix(
            base_distributions,
            rows=len(target_symbols),
            alphabet_size=self.alphabet_size,
            name="base_distributions",
        )
        unsupported = [
            index
            for index, symbol in enumerate(target_symbols)
            if bases[index, symbol] <= 0.0
        ]
        if unsupported:
            raise ValueError(
                "observed target symbol must have positive base support at every row"
            )

        history = list(prefix_symbols[-self.maximum_local_order :])
        base_nll = 0.0
        anchored_nll = 0.0
        raw_feature_activations = 0
        unique_feature_row_updates = 0
        scalar_weight_slots_touched = 0
        maximum_absolute_update = 0.0
        minimum_target_base_probability: float | None = None
        rollback: list[tuple[np.ndarray, np.ndarray]] = []
        try:
            for position, target_symbol in enumerate(target_symbols):
                prefix_tuple = tuple(history)
                raw = self._raw_feature_ids_from_tuples(
                    prefix_tuple,
                    state,
                    freeze_joint=freeze_joint,
                )
                raw_feature_activations += len(raw)
                indexes, counts = np.unique(raw, return_counts=True)
                multiplicities = counts.astype(np.float32)
                residual = np.asarray(
                    multiplicities @ self._weights[indexes],
                    dtype=np.float64,
                )
                prediction = _base_anchored_softmax(bases[position], residual)
                target_base_probability = float(bases[position, target_symbol])
                target_probability = float(prediction[target_symbol])
                base_nll -= float(np.log(target_base_probability))
                anchored_nll -= float(np.log(target_probability))
                minimum_target_base_probability = (
                    target_base_probability
                    if minimum_target_base_probability is None
                    else min(minimum_target_base_probability, target_base_probability)
                )

                error = -prediction
                error[target_symbol] += 1.0
                denominator = float(multiplicities @ multiplicities)
                update = (
                    (self.learning_rate / denominator)
                    * multiplicities[:, None].astype(np.float64)
                    * error[None, :]
                )
                before = np.array(self._weights[indexes], copy=True)
                candidate = before.astype(np.float64) + update
                if not np.all(np.isfinite(candidate)) or np.any(
                    np.abs(candidate) > np.finfo(np.float32).max
                ):
                    raise FloatingPointError(
                        "base-anchored writer update is non-finite"
                    )
                rollback.append((np.array(indexes, copy=True), before))
                self._weights[indexes] = candidate.astype(np.float32)
                applied = self._weights[indexes].astype(np.float64) - before.astype(
                    np.float64
                )
                maximum_absolute_update = max(
                    maximum_absolute_update,
                    float(np.max(np.abs(applied))),
                )
                unique_feature_row_updates += len(indexes)
                scalar_weight_slots_touched += len(indexes) * self.alphabet_size
                history.append(target_symbol)
                del history[: -self.maximum_local_order]
        except BaseException:
            for indexes, weights in reversed(rollback):
                self._weights[indexes] = weights
            raise

        self.episodes_seen += 1
        self.symbols_seen += len(target_symbols)
        symbols = len(target_symbols)
        denominator = float(symbols) if symbols else 1.0
        improvement = base_nll - anchored_nll
        return BaseAnchoredFitDiagnostics(
            symbols=symbols,
            symbol_updates=symbols,
            raw_feature_activations=raw_feature_activations,
            unique_feature_row_updates=unique_feature_row_updates,
            scalar_weight_slots_touched=scalar_weight_slots_touched,
            base_nll_nats=base_nll,
            anchored_nll_nats=anchored_nll,
            residual_improvement_nats=improvement,
            mean_base_nll_nats=base_nll / denominator,
            mean_anchored_nll_nats=anchored_nll / denominator,
            mean_residual_improvement_nats=improvement / denominator,
            maximum_absolute_update=maximum_absolute_update,
            minimum_target_base_probability=minimum_target_base_probability,
        )

    def fit_episodes(self, episodes: Iterable[ContextCoupledEpisode]) -> None:
        """Fit episodes without ever carrying a suffix across their boundaries."""

        for episode in episodes:
            self.fit_episode(
                episode.target,
                episode.persistent_feature_ids,
                prefix=episode.prefix,
            )

    def generate(
        self,
        persistent_feature_ids: FeatureSequence,
        *,
        max_symbols: int,
        prefix: SymbolSequence = b"",
        stop_symbol: int | None = None,
        freeze_joint: bool = False,
    ) -> tuple[int, ...]:
        """Greedily generate from a clamped state and the writer's own output."""

        maximum = operator.index(max_symbols)
        if maximum < 0:
            raise ValueError("max_symbols must be non-negative")
        prefix_symbols = self._symbols(prefix, name="prefix")
        state = self._persistent_features(persistent_feature_ids)
        if stop_symbol is not None:
            stop = operator.index(stop_symbol)
            if not 0 <= stop < self.alphabet_size:
                raise ValueError("stop_symbol is outside the output alphabet")
        else:
            stop = None

        history = list(prefix_symbols[-self.maximum_local_order :])
        output: list[int] = []
        for _ in range(maximum):
            indexes, multiplicities = self._features_from_tuples(
                tuple(history), state, freeze_joint=freeze_joint
            )
            symbol = int(np.argmax(multiplicities @ self._weights[indexes]))
            output.append(symbol)
            history.append(symbol)
            del history[: -self.maximum_local_order]
            if symbol == stop:
                break
        return tuple(output)

    def generate_bytes(
        self,
        persistent_feature_ids: FeatureSequence,
        *,
        max_bytes: int,
        prefix: SymbolSequence = b"",
        stop_byte: int | None = None,
        freeze_joint: bool = False,
    ) -> bytes:
        """Byte convenience wrapper; symbol generation remains the core API."""

        return bytes(
            self.generate(
                persistent_feature_ids,
                max_symbols=max_bytes,
                prefix=prefix,
                stop_symbol=stop_byte,
                freeze_joint=freeze_joint,
            )
        )

    def fingerprint(self) -> str:
        """Digest learned weights for inference-purity checks."""

        return hashlib.sha256(self._weights.view(np.uint8)).hexdigest()

    def snapshot(self) -> ContextCoupledWriterSnapshot:
        """Return a detached, read-only copy suitable for exact restoration."""

        weights = np.array(self._weights, copy=True)
        weights.flags.writeable = False
        return ContextCoupledWriterSnapshot(
            version=self.SNAPSHOT_VERSION,
            kind=self.kind,
            alphabet_size=self.alphabet_size,
            table_bits=self.table_bits,
            learning_rate=self.learning_rate,
            local_orders=self.local_orders,
            max_persistent_features=self.max_persistent_features,
            episodes_seen=self.episodes_seen,
            symbols_seen=self.symbols_seen,
            weights=weights,
        )

    @classmethod
    def from_snapshot(
        cls, snapshot: ContextCoupledWriterSnapshot
    ) -> ContextCoupledByteWriter:
        """Restore a writer from a validated detached snapshot."""

        if snapshot.version != cls.SNAPSHOT_VERSION:
            raise ValueError("unsupported context-coupled writer snapshot version")
        writer = cls(
            snapshot.kind,
            alphabet_size=snapshot.alphabet_size,
            table_bits=snapshot.table_bits,
            learning_rate=snapshot.learning_rate,
            local_orders=snapshot.local_orders,
            max_persistent_features=snapshot.max_persistent_features,
        )
        weights = np.asarray(snapshot.weights)
        if weights.shape != writer._weights.shape or weights.dtype != np.float32:
            raise ValueError("snapshot weight table has incompatible shape or dtype")
        if not np.all(np.isfinite(weights)):
            raise ValueError("snapshot weight table contains non-finite values")
        if snapshot.episodes_seen < 0 or snapshot.symbols_seen < 0:
            raise ValueError("snapshot counters must be non-negative")
        writer._weights[...] = weights
        writer.episodes_seen = int(snapshot.episodes_seen)
        writer.symbols_seen = int(snapshot.symbols_seen)
        return writer

    def _symbols(self, values: object, *, name: str) -> tuple[int, ...]:
        symbols = _integer_tuple(values, name=name)
        if any(symbol < 0 or symbol >= self.alphabet_size for symbol in symbols):
            raise ValueError(f"{name} contains a symbol outside the output alphabet")
        return symbols

    def _persistent_features(self, values: object) -> tuple[int, ...]:
        features = _integer_tuple(values, name="persistent_feature_ids")
        if len(features) > self.max_persistent_features:
            raise ValueError("persistent feature count exceeds configured maximum")
        if self.kind != "local" and not features:
            raise ValueError("contextual writer kinds require persistent features")
        return tuple(feature & _UINT64_MASK for feature in features)

    def _mix(self, value: int) -> int:
        # Same process-independent integer mixer used by the source sandbox.
        mixed = value & _UINT64_MASK
        mixed = (mixed ^ (mixed >> 16)) * 0x45D9F3B
        mixed = (mixed ^ (mixed >> 16)) * 0x45D9F3B
        return (mixed ^ (mixed >> 16)) & self._table_mask

    def _local_feature_ids_from_tuple(self, prefix: tuple[int, ...]) -> np.ndarray:
        result = [0]
        for order in self.local_orders:
            rolling = order * 1_000_003
            for symbol in prefix[-order:]:
                rolling = ((rolling * 257) ^ symbol) & _UINT64_MASK
            result.append(self._mix(rolling))
        return np.asarray(result, dtype=np.int64)

    def _state_feature_ids(self, state: tuple[int, ...]) -> np.ndarray:
        return np.asarray(
            [self._mix(1_234_567 + feature) for feature in state], dtype=np.int64
        )

    def _raw_feature_ids_from_tuples(
        self,
        prefix: tuple[int, ...],
        state: tuple[int, ...],
        freeze_joint: bool = False,
    ) -> np.ndarray:
        local = self._local_feature_ids_from_tuple(prefix)
        if self.kind == "local":
            return local

        persistent = self._state_feature_ids(state)
        shared_state = (persistent * 7_919 + 23_497) & self._table_mask
        shared = np.concatenate((local, shared_state))
        if self.kind == "additive":
            return shared

        joint_count = len(local) * len(persistent)
        if self.kind == "additive_wide":
            # Every extra address depends on a local OR a state feature, never both.
            local_extra_count = (joint_count + 1) // 2
            state_extra_count = joint_count - local_extra_count
            more_local = [
                (
                    (int(local[index % len(local)]) * 1_000_033)
                    ^ ((index // len(local) + 1) * 15_485_863 + 321_743)
                )
                & self._table_mask
                for index in range(local_extra_count)
            ]
            more_state = [
                (
                    (int(persistent[index % len(persistent)]) * 1_000_037)
                    ^ ((index // len(persistent) + 1) * 32_452_843 + 678_679)
                )
                & self._table_mask
                for index in range(state_extra_count)
            ]
            return np.concatenate(
                (
                    shared,
                    np.asarray(more_local, dtype=np.int64),
                    np.asarray(more_state, dtype=np.int64),
                )
            )

        joint_local = self._local_feature_ids_from_tuple(()) if freeze_joint else local
        joint = (
            (joint_local[:, None] * 1_000_033)
            ^ (persistent[None, :] * 1_000_037)
            ^ 972_663_749
        ) & self._table_mask
        return np.concatenate((shared, joint.ravel()))

    def _features_from_tuples(
        self,
        prefix: tuple[int, ...],
        state: tuple[int, ...],
        *,
        freeze_joint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = self._raw_feature_ids_from_tuples(prefix, state, freeze_joint)
        indexes, counts = np.unique(raw, return_counts=True)
        return indexes, counts.astype(np.float32)
