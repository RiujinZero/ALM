"""Exact finite completion potentials and their Doob transform.

The core is deliberately ignorant of conversations and model internals.  It
accepts a finite, prefix-free set of byte sequences and a frozen base next-byte
distribution for every internal trie prefix.  Backward dynamic programming
then computes

``h(s) = P_base(eventually reaching an accepted leaf | prefix=s)``.

At gain one, :meth:`ExactCompletionPotential.distribution` is the exact Doob
transform ``p*(b|s) = p0(b|s) h(sb) / h(s)``.  Thus it conditions the base
process on the finite accepted set without changing relative probabilities of
accepted complete paths.  Gain zero is an exact base-distribution bypass.

``false_positive_floor`` is an explicit estimator-error stress.  It replaces
each successor potential by ``max(h, floor)``.  This can leak into an off-trie
branch, but it can never give mass to a symbol that has zero base support.  Once
such a leak occurs, the caller may supply the current base distribution for an
off-trie prefix; all successor potentials are then the same floor, so the
correct stressed route is exactly the supplied base distribution.
"""

from __future__ import annotations

import hashlib
import math
import operator
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

ByteSequence = bytes | bytearray | memoryview | Sequence[int]
Distribution = Sequence[float] | np.ndarray
DistributionCallback = Callable[[bytes], Distribution]


class PrefixStatus(str, Enum):
    """Relationship between a prefix and the accepted-sequence trie."""

    INTERNAL = "internal"
    LEAF = "leaf"
    OFF_TRIE = "off_trie"


class CompletionPotentialError(RuntimeError):
    """Base class for completion-potential routing failures."""


class CompletionLeafError(CompletionPotentialError):
    """Raised when a next-symbol distribution is requested at a leaf."""


class OffTriePrefixError(CompletionPotentialError):
    """Raised when an operation requires a prefix in the compiled trie."""


class ZeroAcceptedMassError(CompletionPotentialError):
    """Raised when an exact guided distribution has no supported successor."""


@dataclass(frozen=True)
class CompletionPotentialDiagnostics:
    """Small immutable summary of the compiled finite potential."""

    alphabet_size: int
    accepted_sequence_count: int
    internal_prefix_count: int
    leaf_count: int
    maximum_depth: int
    finite_mass_internal_prefix_count: int
    zero_mass_internal_prefix_count: int
    root_log_mass: float
    root_mass: float


def _byte_string(value: object, *, name: str) -> bytes:
    """Copy bytes or a one-dimensional sequence of integer byte values."""

    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        raise TypeError(f"{name} must be a byte sequence, not text")
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a byte sequence") from exc
    result = bytearray()
    for item in items:
        try:
            byte = operator.index(item)
        except TypeError as exc:
            raise TypeError(f"{name} must contain only integers") from exc
        if not 0 <= byte <= 255:
            raise ValueError(f"{name} contains a value outside the byte range")
        result.append(byte)
    return bytes(result)


def _normalized_distribution(
    values: Distribution,
    *,
    alphabet_size: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (alphabet_size,):
        raise ValueError(f"{name} must have shape ({alphabet_size},)")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    total = float(vector.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} must sum to one")
    normalized = np.array(vector / total, dtype=np.float64, copy=True)
    normalized.flags.writeable = False
    return normalized


def _logsumexp(terms: Iterable[float]) -> float:
    finite = [term for term in terms if math.isfinite(term)]
    if not finite:
        return -math.inf
    pivot = max(finite)
    return pivot + math.log(sum(math.exp(term - pivot) for term in finite))


class ExactCompletionPotential:
    """Compiled exact potential for a finite prefix-free completion set."""

    FINGERPRINT_VERSION = 1

    def __init__(
        self,
        accepted_sequences: tuple[bytes, ...],
        *,
        alphabet_size: int,
        children: dict[bytes, tuple[int, ...]],
        base_distributions: dict[bytes, np.ndarray],
        log_masses: dict[bytes, float],
    ) -> None:
        self.alphabet_size = alphabet_size
        self._accepted_sequences = accepted_sequences
        self._leaf_set = frozenset(accepted_sequences)
        self._children = children
        self._base_distributions = base_distributions
        self._log_masses = log_masses
        self._internal_prefixes = tuple(sorted(base_distributions))

    @classmethod
    def compile(
        cls,
        accepted_sequences: Iterable[ByteSequence],
        base_distribution_callback: DistributionCallback,
        *,
        alphabet_size: int = 256,
    ) -> ExactCompletionPotential:
        """Build a trie and query a frozen base distribution per internal prefix.

        Prefixes are queried exactly once each, in canonical byte-string order,
        so compilation is independent of accepted-sequence input order.
        """

        if not callable(base_distribution_callback):
            raise TypeError("base_distribution_callback must be callable")
        sequences, children = cls._validated_trie(
            accepted_sequences, alphabet_size=alphabet_size
        )
        leaves = frozenset(sequences)
        internal = tuple(sorted(prefix for prefix in children if prefix not in leaves))
        distributions = {
            prefix: _normalized_distribution(
                base_distribution_callback(prefix),
                alphabet_size=alphabet_size,
                name=f"base distribution at prefix {prefix!r}",
            )
            for prefix in internal
        }
        return cls._from_validated_parts(
            sequences,
            alphabet_size=alphabet_size,
            children=children,
            base_distributions=distributions,
        )

    @classmethod
    def from_prefix_distributions(
        cls,
        accepted_sequences: Iterable[ByteSequence],
        prefix_distributions: Mapping[ByteSequence, Distribution],
        *,
        alphabet_size: int = 256,
    ) -> ExactCompletionPotential:
        """Build from pre-recorded distributions for exactly the internal nodes."""

        if not isinstance(prefix_distributions, Mapping):
            raise TypeError("prefix_distributions must be a mapping")
        sequences, children = cls._validated_trie(
            accepted_sequences, alphabet_size=alphabet_size
        )
        leaves = frozenset(sequences)
        expected = frozenset(prefix for prefix in children if prefix not in leaves)
        imported: dict[bytes, Distribution] = {}
        for raw_prefix, distribution in prefix_distributions.items():
            prefix = _byte_string(raw_prefix, name="prefix-distribution key")
            if prefix in imported:
                raise ValueError("prefix_distributions contains duplicate byte keys")
            imported[prefix] = distribution
        supplied = frozenset(imported)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            raise ValueError(
                "prefix_distributions keys must exactly match internal prefixes; "
                f"missing={missing!r}, extra={extra!r}"
            )
        distributions = {
            prefix: _normalized_distribution(
                imported[prefix],
                alphabet_size=alphabet_size,
                name=f"base distribution at prefix {prefix!r}",
            )
            for prefix in sorted(expected)
        }
        return cls._from_validated_parts(
            sequences,
            alphabet_size=alphabet_size,
            children=children,
            base_distributions=distributions,
        )

    @classmethod
    def _validated_trie(
        cls,
        raw_sequences: Iterable[ByteSequence],
        *,
        alphabet_size: int,
    ) -> tuple[tuple[bytes, ...], dict[bytes, tuple[int, ...]]]:
        try:
            size = operator.index(alphabet_size)
        except TypeError as exc:
            raise TypeError("alphabet_size must be an integer") from exc
        if not 1 <= size <= 256:
            raise ValueError("alphabet_size must be between 1 and 256")
        if isinstance(raw_sequences, (bytes, bytearray, memoryview)):
            raise TypeError("accepted_sequences must be an iterable of byte sequences")
        try:
            sequences = tuple(
                _byte_string(sequence, name="accepted sequence")
                for sequence in raw_sequences
            )
        except TypeError as exc:
            if "accepted sequence" in str(exc):
                raise
            raise TypeError(
                "accepted_sequences must be an iterable of byte sequences"
            ) from exc
        if not sequences:
            raise ValueError("accepted_sequences must be non-empty")
        if any(not sequence for sequence in sequences):
            raise ValueError("accepted sequences must be non-empty")
        outside = next(
            (
                byte
                for sequence in sequences
                for byte in sequence
                if byte >= size
            ),
            None,
        )
        if outside is not None:
            raise ValueError("accepted sequence contains a byte outside the alphabet")

        ordered = tuple(sorted(sequences))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous == current:
                raise ValueError("accepted_sequences contains a duplicate")
            if current.startswith(previous):
                raise ValueError(
                    "accepted_sequences must be prefix-free; "
                    f"{previous!r} prefixes {current!r}"
                )

        mutable_children: dict[bytes, set[int]] = {b"": set()}
        for sequence in ordered:
            for index, byte in enumerate(sequence):
                prefix = sequence[:index]
                child = sequence[: index + 1]
                mutable_children.setdefault(prefix, set()).add(byte)
                mutable_children.setdefault(child, set())
        children = {
            prefix: tuple(sorted(symbols))
            for prefix, symbols in mutable_children.items()
        }
        return ordered, children

    @classmethod
    def _from_validated_parts(
        cls,
        sequences: tuple[bytes, ...],
        *,
        alphabet_size: int,
        children: dict[bytes, tuple[int, ...]],
        base_distributions: dict[bytes, np.ndarray],
    ) -> ExactCompletionPotential:
        log_masses: dict[bytes, float] = {sequence: 0.0 for sequence in sequences}
        for prefix in sorted(base_distributions, key=lambda item: (-len(item), item)):
            base = base_distributions[prefix]
            terms: list[float] = []
            for byte in children[prefix]:
                probability = float(base[byte])
                child_log_mass = log_masses[prefix + bytes((byte,))]
                if probability > 0.0 and math.isfinite(child_log_mass):
                    terms.append(math.log(probability) + child_log_mass)
            log_masses[prefix] = _logsumexp(terms)
        return cls(
            sequences,
            alphabet_size=alphabet_size,
            children=children,
            base_distributions=base_distributions,
            log_masses=log_masses,
        )

    @property
    def accepted_sequences(self) -> tuple[bytes, ...]:
        """Accepted leaves in canonical order."""

        return self._accepted_sequences

    @property
    def internal_prefixes(self) -> tuple[bytes, ...]:
        """Internal prefixes in canonical order."""

        return self._internal_prefixes

    @property
    def leaf_prefixes(self) -> tuple[bytes, ...]:
        """Alias for the canonical accepted leaves."""

        return self._accepted_sequences

    @property
    def accepted_log_mass(self) -> float:
        """Log base probability of reaching any accepted completion."""

        return self._log_masses[b""]

    @property
    def accepted_mass(self) -> float:
        """Base probability of reaching any accepted completion."""

        return self._probability_from_log(self.accepted_log_mass)

    @property
    def diagnostics(self) -> CompletionPotentialDiagnostics:
        """Return compilation counts and root accepted mass."""

        finite = sum(
            math.isfinite(self._log_masses[prefix])
            for prefix in self._internal_prefixes
        )
        return CompletionPotentialDiagnostics(
            alphabet_size=self.alphabet_size,
            accepted_sequence_count=len(self._accepted_sequences),
            internal_prefix_count=len(self._internal_prefixes),
            leaf_count=len(self._accepted_sequences),
            maximum_depth=max(map(len, self._accepted_sequences)),
            finite_mass_internal_prefix_count=finite,
            zero_mass_internal_prefix_count=len(self._internal_prefixes) - finite,
            root_log_mass=self.accepted_log_mass,
            root_mass=self.accepted_mass,
        )

    def status(self, prefix: ByteSequence) -> PrefixStatus:
        """Classify a byte prefix without conflating leaves and misses."""

        key = self._prefix(prefix)
        if key in self._leaf_set:
            return PrefixStatus.LEAF
        if key in self._base_distributions:
            return PrefixStatus.INTERNAL
        return PrefixStatus.OFF_TRIE

    def base_distribution(self, prefix: ByteSequence) -> np.ndarray:
        """Return a detached copy of the compiled base distribution."""

        key = self._prefix(prefix)
        self._require_internal(key)
        return np.array(self._base_distributions[key], copy=True)

    def log_mass(self, prefix: ByteSequence) -> float:
        """Return exact log accepted-subtree mass; a leaf has log mass zero."""

        key = self._prefix(prefix)
        if key not in self._log_masses:
            raise OffTriePrefixError(f"prefix {key!r} is outside the completion trie")
        return self._log_masses[key]

    def mass(self, prefix: ByteSequence) -> float:
        """Return accepted-subtree mass in probability space."""

        return self._probability_from_log(self.log_mass(prefix))

    def successor_log_masses(self, prefix: ByteSequence) -> np.ndarray:
        """Return log h for every next byte, with ``-inf`` off the trie."""

        key = self._prefix(prefix)
        self._require_internal(key)
        values = np.full(self.alphabet_size, -np.inf, dtype=np.float64)
        for byte in self._children[key]:
            values[byte] = self._log_masses[key + bytes((byte,))]
        return values

    def distribution(
        self,
        prefix: ByteSequence,
        *,
        gain: float = 1.0,
        false_positive_floor: float = 0.0,
        off_trie_base_distribution: Distribution | None = None,
    ) -> np.ndarray:
        """Return a normalized potential-guided next-byte distribution.

        For an internal prefix the frozen compiled base distribution is always
        used.  If a positive floor has already allowed generation to leave the
        trie, the caller must supply the current base distribution through
        ``off_trie_base_distribution``.  Because every continuation of an
        off-trie prefix has the same floored potential, the stressed route then
        equals that base distribution exactly.  A zero-floor off-trie request
        always fails rather than silently falling back.
        """

        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("gain must be finite and non-negative")
        if (
            not math.isfinite(false_positive_floor)
            or not 0.0 <= false_positive_floor <= 1.0
        ):
            raise ValueError("false_positive_floor must be finite and in [0, 1]")

        key = self._prefix(prefix)
        state = self.status(key)
        if state is PrefixStatus.LEAF:
            raise CompletionLeafError(f"prefix {key!r} is an accepted leaf")
        if state is PrefixStatus.OFF_TRIE:
            if false_positive_floor == 0.0:
                raise OffTriePrefixError(
                    f"prefix {key!r} is outside the completion trie"
                )
            if off_trie_base_distribution is None:
                raise OffTriePrefixError(
                    "a positive-floor off-trie prefix requires "
                    "off_trie_base_distribution"
                )
            return np.array(
                _normalized_distribution(
                    off_trie_base_distribution,
                    alphabet_size=self.alphabet_size,
                    name="off_trie_base_distribution",
                ),
                copy=True,
            )

        base = self._base_distributions[key]
        if gain == 0.0:
            return np.array(base, copy=True)

        successor_logs = self.successor_log_masses(key)
        if false_positive_floor > 0.0:
            successor_logs = np.maximum(
                successor_logs, math.log(false_positive_floor)
            )
        support = base > 0.0
        log_weights = np.full(self.alphabet_size, -np.inf, dtype=np.float64)
        viable = support & np.isfinite(successor_logs)
        log_weights[viable] = np.log(base[viable]) + gain * successor_logs[viable]
        if not np.any(viable):
            raise ZeroAcceptedMassError(
                f"prefix {key!r} has zero base-supported accepted mass"
            )
        pivot = float(np.max(log_weights[viable]))
        result = np.zeros(self.alphabet_size, dtype=np.float64)
        result[viable] = np.exp(log_weights[viable] - pivot)
        result /= float(result.sum())
        return result

    def fingerprint(self) -> str:
        """Return a deterministic digest of the trie and frozen base policy."""

        digest = hashlib.sha256()
        digest.update(b"ExactCompletionPotential")
        digest.update(struct.pack("<II", self.FINGERPRINT_VERSION, self.alphabet_size))
        digest.update(struct.pack("<I", len(self._accepted_sequences)))
        for sequence in self._accepted_sequences:
            digest.update(struct.pack("<I", len(sequence)))
            digest.update(sequence)
        for prefix in self._internal_prefixes:
            digest.update(struct.pack("<I", len(prefix)))
            digest.update(prefix)
            little_endian = self._base_distributions[prefix].astype("<f8", copy=False)
            digest.update(little_endian.tobytes(order="C"))
        return digest.hexdigest()

    def _prefix(self, prefix: ByteSequence) -> bytes:
        key = _byte_string(prefix, name="prefix")
        if any(byte >= self.alphabet_size for byte in key):
            raise ValueError("prefix contains a byte outside the alphabet")
        return key

    def _require_internal(self, key: bytes) -> None:
        if key in self._leaf_set:
            raise CompletionLeafError(f"prefix {key!r} is an accepted leaf")
        if key not in self._base_distributions:
            raise OffTriePrefixError(f"prefix {key!r} is outside the completion trie")

    @staticmethod
    def _probability_from_log(log_probability: float) -> float:
        return math.exp(log_probability) if math.isfinite(log_probability) else 0.0
