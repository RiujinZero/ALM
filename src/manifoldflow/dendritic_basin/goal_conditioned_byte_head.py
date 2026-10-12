"""Compact matched byte heads for testing persistent goal representations.

The head deliberately knows nothing about conversations or any other modality.
Its only non-byte input is a probability distribution over goal codes supplied
by an upstream representation learner.  Both heads see the same bounded byte
suffix and the same observed response bytes:

``q0``
    A local hashed n-gram table.

``qg``
    The same table, with code-conditioned counts backed off to ``q0``.

Training is episode based: every call starts its causal suffix from that
episode's prefix, so response bytes from separate episodes are never joined
into an accidental transition.  Scoring is functional and keeps the supplied
goal distribution clamped for the complete candidate trace.  This makes it
safe to score several branches without snapshot/restore gymnastics; explicit
snapshots are provided for checkpointing and verification as well.

``MatchedSignedGoalByteHead`` provides the same boundary-safe interface for a
signed continuous future-state vector.  It keeps ``q0`` unchanged and learns a
compact, shrunk goal/byte covariance tilt rather than quantizing the vector
into a probability simplex.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

ByteLike = bytes | bytearray | memoryview


@dataclass(frozen=True)
class EpisodeByteTrace:
    """One self-supervised response and its already-inferred goal code."""

    response: bytes
    goal_distribution: Sequence[float]
    prefix: bytes = b""


@dataclass(frozen=True)
class ByteTraceScore:
    """Teacher-forced log scores for one candidate response."""

    byte_count: int
    local_log_probability: float
    goal_log_probability: float

    @property
    def log_likelihood_ratio(self) -> float:
        return self.goal_log_probability - self.local_log_probability


@dataclass(frozen=True)
class ByteHeadSnapshot:
    """Complete, detached state for a :class:`MatchedGoalByteHead`."""

    version: int
    goal_code_count: int
    context_order: int
    context_bucket_count: int
    local_prior_mass: float
    goal_backoff_mass: float
    episodes_seen: int
    bytes_seen: int
    local_counts: np.ndarray
    goal_counts: np.ndarray


def _as_bytes(value: ByteLike, *, name: str) -> bytes:
    try:
        return bytes(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be bytes-like") from exc


def _probability_vector(
    values: Sequence[float] | np.ndarray,
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return np.array(vector / total, dtype=np.float64, copy=True)


def shuffled_goal_control(
    episodes: Sequence[EpisodeByteTrace], *, seed: int
) -> tuple[list[EpisodeByteTrace], np.ndarray]:
    """Return episodes with a reproducible, no-fixed-point code permutation.

    Response bytes and prefixes stay in their original rows.  For two or more
    episodes, Sattolo's algorithm assigns every row another row's goal code,
    making this a strict shuffled-code control rather than a shuffle that can
    silently leave some labels untouched.
    """

    count = len(episodes)
    permutation = np.arange(count, dtype=np.int64)
    if count > 1:
        rng = np.random.default_rng(seed)
        for index in range(count - 1, 0, -1):
            other = int(rng.integers(0, index))
            permutation[index], permutation[other] = (
                permutation[other],
                permutation[index],
            )
    shuffled = [
        replace(
            episode,
            goal_distribution=np.array(
                episodes[int(permutation[index])].goal_distribution,
                dtype=np.float64,
                copy=True,
            ),
        )
        for index, episode in enumerate(episodes)
    ]
    return shuffled, permutation


def likelihood_ratio_correction(
    base_distribution: Sequence[float] | np.ndarray,
    local_distribution: Sequence[float] | np.ndarray,
    goal_distribution: Sequence[float] | np.ndarray,
    *,
    beta: float,
    log_ratio_clip: float = 2.0,
) -> np.ndarray:
    """Apply ``base * exp(beta * clip(log(qg/q0)))`` and renormalize.

    The input base distribution can come from any byte predictor.  ``beta=0``
    is an exact bypass: it returns a value-for-value copy with the base's dtype,
    which is useful for matched-seed ablations.
    """

    base_input = np.asarray(base_distribution)
    if base_input.shape != (256,):
        raise ValueError("base_distribution must have shape (256,)")
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

    local = _probability_vector(
        local_distribution, expected_size=256, name="local_distribution"
    )
    goal = _probability_vector(
        goal_distribution, expected_size=256, name="goal_distribution"
    )
    log_ratio = np.clip(np.log(goal) - np.log(local), -log_ratio_clip, log_ratio_clip)
    support = base > 0.0
    logits = np.full(256, -np.inf, dtype=np.float64)
    logits[support] = np.log(base[support]) + beta * log_ratio[support]
    pivot = float(np.max(logits[support]))
    corrected = np.zeros(256, dtype=np.float64)
    corrected[support] = np.exp(logits[support] - pivot)
    corrected /= corrected.sum()
    return corrected


class MatchedGoalByteHead:
    """A bounded-memory local/goal byte-head pair.

    Contexts are projected into a fixed number of deterministic hash buckets,
    bounding memory independently of corpus size.  The goal table stores soft
    counts, so one-hot prototypes and uncertain posterior distributions use the
    same API.  Goal-conditioned rows use the local posterior as their prior;
    an unseen code/context therefore falls back exactly to local evidence.
    """

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        goal_code_count: int,
        *,
        context_order: int = 4,
        context_bucket_count: int = 1024,
        local_prior_mass: float = 1.0,
        goal_backoff_mass: float = 2.0,
    ) -> None:
        if goal_code_count < 2:
            raise ValueError("goal_code_count must be at least two")
        if not 0 <= context_order <= 32:
            raise ValueError("context_order must be between zero and 32")
        if context_bucket_count < 1:
            raise ValueError("context_bucket_count must be positive")
        if not np.isfinite(local_prior_mass) or local_prior_mass <= 0.0:
            raise ValueError("local_prior_mass must be finite and positive")
        if not np.isfinite(goal_backoff_mass) or goal_backoff_mass <= 0.0:
            raise ValueError("goal_backoff_mass must be finite and positive")

        self.goal_code_count = int(goal_code_count)
        self.context_order = int(context_order)
        self.context_bucket_count = int(context_bucket_count)
        self.local_prior_mass = float(local_prior_mass)
        self.goal_backoff_mass = float(goal_backoff_mass)
        self.episodes_seen = 0
        self.bytes_seen = 0
        self._local_counts = np.zeros(
            (self.context_bucket_count, 256), dtype=np.float32
        )
        self._goal_counts = np.zeros(
            (self.goal_code_count, self.context_bucket_count, 256),
            dtype=np.float32,
        )

    @property
    def allocated_bytes(self) -> int:
        """Storage used by the two count tables."""

        return int(self._local_counts.nbytes + self._goal_counts.nbytes)

    def _goal_vector(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return _probability_vector(
            values,
            expected_size=self.goal_code_count,
            name="goal_code_distribution",
        )

    def _context_bucket(self, context: ByteLike) -> int:
        suffix = _as_bytes(context, name="context")
        if self.context_order == 0:
            suffix = b""
        else:
            suffix = suffix[-self.context_order :]
        value = (2166136261 ^ len(suffix)) & 0xFFFFFFFF
        for byte in suffix:
            value ^= byte
            value = (value * 16777619) & 0xFFFFFFFF
        return value % self.context_bucket_count

    def fit_episode(
        self,
        response: ByteLike,
        goal_code_distribution: Sequence[float] | np.ndarray,
        *,
        prefix: ByteLike = b"",
    ) -> None:
        """Learn next-byte observations from exactly one bounded episode."""

        response_bytes = _as_bytes(response, name="response")
        prefix_bytes = _as_bytes(prefix, name="prefix")
        code = self._goal_vector(goal_code_distribution)
        context = bytearray(prefix_bytes[-self.context_order :])
        if self.context_order == 0:
            context.clear()
        for target in response_bytes:
            bucket = self._context_bucket(context)
            self._local_counts[bucket, target] += 1.0
            self._goal_counts[:, bucket, target] += code
            if self.context_order:
                context.append(target)
                del context[: -self.context_order]
        self.episodes_seen += 1
        self.bytes_seen += len(response_bytes)

    def fit_episodes(self, episodes: Iterable[EpisodeByteTrace]) -> None:
        """Fit traces while preserving every caller-supplied episode boundary."""

        for episode in episodes:
            self.fit_episode(
                episode.response,
                episode.goal_distribution,
                prefix=episode.prefix,
            )

    def next_byte_distributions(
        self,
        context: ByteLike,
        goal_code_distribution: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return matched ``(q0, qg)`` distributions without changing state."""

        code = self._goal_vector(goal_code_distribution)
        bucket = self._context_bucket(context)
        local_counts = self._local_counts[bucket].astype(np.float64)
        local_total = float(local_counts.sum())
        local = (local_counts + self.local_prior_mass / 256.0) / (
            local_total + self.local_prior_mass
        )

        goal_counts = self._goal_counts[:, bucket].astype(np.float64)
        goal_totals = goal_counts.sum(axis=1, keepdims=True)
        per_code = (goal_counts + self.goal_backoff_mass * local[None, :]) / (
            goal_totals + self.goal_backoff_mass
        )
        conditioned = code @ per_code
        conditioned /= conditioned.sum()
        return local, conditioned

    def trace_probabilities(
        self,
        prefix: ByteLike,
        candidate: ByteLike,
        goal_code_distribution: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Teacher-force one candidate while clamping a copied goal state."""

        prefix_bytes = _as_bytes(prefix, name="prefix")
        candidate_bytes = _as_bytes(candidate, name="candidate")
        code = self._goal_vector(goal_code_distribution)
        local_trace = np.empty((len(candidate_bytes), 256), dtype=np.float64)
        goal_trace = np.empty_like(local_trace)
        context = bytearray(prefix_bytes[-self.context_order :])
        if self.context_order == 0:
            context.clear()
        for index, target in enumerate(candidate_bytes):
            local, conditioned = self.next_byte_distributions(context, code)
            local_trace[index] = local
            goal_trace[index] = conditioned
            if self.context_order:
                context.append(target)
                del context[: -self.context_order]
        return local_trace, goal_trace

    def score_trace(
        self,
        prefix: ByteLike,
        candidate: ByteLike,
        goal_code_distribution: Sequence[float] | np.ndarray,
    ) -> ByteTraceScore:
        """Score an observed candidate under both matched heads."""

        candidate_bytes = _as_bytes(candidate, name="candidate")
        local, conditioned = self.trace_probabilities(
            prefix, candidate_bytes, goal_code_distribution
        )
        if not candidate_bytes:
            return ByteTraceScore(0, 0.0, 0.0)
        indexes = np.arange(len(candidate_bytes))
        targets = np.frombuffer(candidate_bytes, dtype=np.uint8)
        local_log_probability = float(np.log(local[indexes, targets]).sum())
        goal_log_probability = float(np.log(conditioned[indexes, targets]).sum())
        return ByteTraceScore(
            len(candidate_bytes), local_log_probability, goal_log_probability
        )

    def correct_base_distribution(
        self,
        base_distribution: Sequence[float] | np.ndarray,
        context: ByteLike,
        goal_code_distribution: Sequence[float] | np.ndarray,
        *,
        beta: float,
        log_ratio_clip: float = 2.0,
    ) -> np.ndarray:
        """Correct one arbitrary base-model byte distribution."""

        local, conditioned = self.next_byte_distributions(
            context, goal_code_distribution
        )
        return likelihood_ratio_correction(
            base_distribution,
            local,
            conditioned,
            beta=beta,
            log_ratio_clip=log_ratio_clip,
        )

    def snapshot(self) -> ByteHeadSnapshot:
        """Return a detached, read-only checkpoint of all learned state."""

        local_counts = self._local_counts.copy()
        goal_counts = self._goal_counts.copy()
        local_counts.setflags(write=False)
        goal_counts.setflags(write=False)
        return ByteHeadSnapshot(
            version=self.SNAPSHOT_VERSION,
            goal_code_count=self.goal_code_count,
            context_order=self.context_order,
            context_bucket_count=self.context_bucket_count,
            local_prior_mass=self.local_prior_mass,
            goal_backoff_mass=self.goal_backoff_mass,
            episodes_seen=self.episodes_seen,
            bytes_seen=self.bytes_seen,
            local_counts=local_counts,
            goal_counts=goal_counts,
        )

    @classmethod
    def from_snapshot(cls, snapshot: ByteHeadSnapshot) -> MatchedGoalByteHead:
        """Construct an independent head from a validated snapshot."""

        if snapshot.version != cls.SNAPSHOT_VERSION:
            raise ValueError(f"unsupported snapshot version {snapshot.version}")
        head = cls(
            snapshot.goal_code_count,
            context_order=snapshot.context_order,
            context_bucket_count=snapshot.context_bucket_count,
            local_prior_mass=snapshot.local_prior_mass,
            goal_backoff_mass=snapshot.goal_backoff_mass,
        )
        expected_local_shape = (snapshot.context_bucket_count, 256)
        expected_goal_shape = (
            snapshot.goal_code_count,
            snapshot.context_bucket_count,
            256,
        )
        if snapshot.local_counts.shape != expected_local_shape:
            raise ValueError("snapshot local-count shape does not match its metadata")
        if snapshot.goal_counts.shape != expected_goal_shape:
            raise ValueError("snapshot goal-count shape does not match its metadata")
        if snapshot.episodes_seen < 0 or snapshot.bytes_seen < 0:
            raise ValueError("snapshot counters must be non-negative")
        if (
            not np.all(np.isfinite(snapshot.local_counts))
            or not np.all(np.isfinite(snapshot.goal_counts))
            or np.any(snapshot.local_counts < 0.0)
            or np.any(snapshot.goal_counts < 0.0)
        ):
            raise ValueError("snapshot counts must be finite and non-negative")
        head._local_counts[...] = snapshot.local_counts
        head._goal_counts[...] = snapshot.goal_counts
        head.episodes_seen = int(snapshot.episodes_seen)
        head.bytes_seen = int(snapshot.bytes_seen)
        return head


@dataclass(frozen=True)
class SignedEpisodeByteTrace:
    """One response paired with an upstream signed continuous goal vector."""

    response: bytes
    goal_vector: Sequence[float]
    prefix: bytes = b""


@dataclass(frozen=True)
class SignedByteHeadSnapshot:
    """Complete detached state for :class:`MatchedSignedGoalByteHead`."""

    version: int
    goal_dimensions: int
    context_order: int
    context_bucket_count: int
    association_bucket_count: int
    local_prior_mass: float
    association_prior_mass: float
    goal_gain: float
    max_goal_norm: float
    logit_clip: float
    minimum_probability: float
    episodes_seen: int
    bytes_seen: int
    local_counts: np.ndarray
    association_counts: np.ndarray
    goal_sums: np.ndarray


def _signed_goal_vector(
    values: Sequence[float] | np.ndarray,
    *,
    expected_size: int,
    max_norm: float,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (expected_size,):
        raise ValueError(f"goal_vector must have shape ({expected_size},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("goal_vector must contain finite values")
    vector = np.array(vector, dtype=np.float64, copy=True)
    largest = float(np.max(np.abs(vector), initial=0.0))
    if largest == 0.0:
        return vector
    scaled_norm = float(np.linalg.norm(vector / largest))
    # Compare as a ratio so even finite values near float64's maximum are
    # clamped without overflowing while computing their Euclidean norm.
    if largest > max_norm / scaled_norm:
        vector *= (max_norm / largest) / scaled_norm
    return vector


def shuffled_signed_goal_control(
    episodes: Sequence[SignedEpisodeByteTrace], *, seed: int
) -> tuple[list[SignedEpisodeByteTrace], np.ndarray]:
    """Return a reproducible strict shuffle of signed episode goal vectors."""

    count = len(episodes)
    permutation = np.arange(count, dtype=np.int64)
    if count > 1:
        rng = np.random.default_rng(seed)
        for index in range(count - 1, 0, -1):
            other = int(rng.integers(0, index))
            permutation[index], permutation[other] = (
                permutation[other],
                permutation[index],
            )
    shuffled = [
        replace(
            episode,
            goal_vector=np.array(
                episodes[int(permutation[index])].goal_vector,
                dtype=np.float64,
                copy=True,
            ),
        )
        for index, episode in enumerate(episodes)
    ]
    return shuffled, permutation


class MatchedSignedGoalByteHead:
    """Compact byte action head for signed continuous goal vectors.

    ``q0`` is the same smoothed, fine-context count model used when the goal is
    enabled.  The extra pathway stores only per-byte sums of supplied goal
    vectors in a smaller set of context buckets.  At inference it computes a
    shrunk covariance contrast against that bucket's mean goal, then applies a
    bounded exponential tilt to ``q0``.  No target code, label, or modality
    semantics are inferred inside this class.

    Passing a zero vector returns ``q0`` bit-identically.  Thus goal-off keeps
    the entire head allocated (a matched-memory control) while removing only
    its conditional effect.
    """

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        goal_dimensions: int,
        *,
        context_order: int = 4,
        context_bucket_count: int = 1024,
        association_bucket_count: int = 64,
        local_prior_mass: float = 1.0,
        association_prior_mass: float = 4.0,
        goal_gain: float = 1.0,
        max_goal_norm: float = 8.0,
        logit_clip: float = 4.0,
        minimum_probability: float = 1e-12,
    ) -> None:
        if goal_dimensions < 1:
            raise ValueError("goal_dimensions must be positive")
        if not 0 <= context_order <= 32:
            raise ValueError("context_order must be between zero and 32")
        if context_bucket_count < 1 or association_bucket_count < 1:
            raise ValueError("context bucket counts must be positive")
        for name, value in (
            ("local_prior_mass", local_prior_mass),
            ("association_prior_mass", association_prior_mass),
            ("goal_gain", goal_gain),
            ("max_goal_norm", max_goal_norm),
            ("logit_clip", logit_clip),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(minimum_probability)
            or not 0.0 <= minimum_probability < 1.0 / 256.0
        ):
            raise ValueError("minimum_probability must lie in [0, 1/256)")

        self.goal_dimensions = int(goal_dimensions)
        self.context_order = int(context_order)
        self.context_bucket_count = int(context_bucket_count)
        self.association_bucket_count = int(association_bucket_count)
        self.local_prior_mass = float(local_prior_mass)
        self.association_prior_mass = float(association_prior_mass)
        self.goal_gain = float(goal_gain)
        self.max_goal_norm = float(max_goal_norm)
        self.logit_clip = float(logit_clip)
        self.minimum_probability = float(minimum_probability)
        self.episodes_seen = 0
        self.bytes_seen = 0
        self._local_counts = np.zeros(
            (self.context_bucket_count, 256), dtype=np.float32
        )
        self._association_counts = np.zeros(
            (self.association_bucket_count, 256), dtype=np.float32
        )
        self._goal_sums = np.zeros(
            (self.association_bucket_count, self.goal_dimensions, 256),
            dtype=np.float32,
        )
        # Derived matrices only: omitted from snapshots and persistent-memory
        # accounting, and safe to discard at any time.
        self._association_cache: dict[int, np.ndarray] = {}

    @property
    def allocated_bytes(self) -> int:
        """Persistent statistical storage, excluding the derived cache."""

        return int(
            self._local_counts.nbytes
            + self._association_counts.nbytes
            + self._goal_sums.nbytes
        )

    @property
    def transient_cache_bytes(self) -> int:
        """Current derived association-cache storage."""

        return int(sum(value.nbytes for value in self._association_cache.values()))

    def _goal_vector(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return _signed_goal_vector(
            values,
            expected_size=self.goal_dimensions,
            max_norm=self.max_goal_norm,
        )

    def _context_hash(self, context: ByteLike) -> int:
        suffix = _as_bytes(context, name="context")
        if self.context_order == 0:
            suffix = b""
        else:
            suffix = suffix[-self.context_order :]
        value = (2166136261 ^ len(suffix)) & 0xFFFFFFFF
        for byte in suffix:
            value ^= byte
            value = (value * 16777619) & 0xFFFFFFFF
        return value

    def _buckets(self, context: ByteLike) -> tuple[int, int]:
        value = self._context_hash(context)
        return (
            value % self.context_bucket_count,
            value % self.association_bucket_count,
        )

    def fit_episode(
        self,
        response: ByteLike,
        goal_vector: Sequence[float] | np.ndarray,
        *,
        prefix: ByteLike = b"",
    ) -> None:
        """Learn byte events from one episode and one supplied goal vector."""

        response_bytes = _as_bytes(response, name="response")
        prefix_bytes = _as_bytes(prefix, name="prefix")
        goal = self._goal_vector(goal_vector)
        context = bytearray(prefix_bytes[-self.context_order :])
        dirty_association_buckets: set[int] = set()
        if self.context_order == 0:
            context.clear()
        for target in response_bytes:
            local_bucket, association_bucket = self._buckets(context)
            self._local_counts[local_bucket, target] += 1.0
            self._association_counts[association_bucket, target] += 1.0
            self._goal_sums[association_bucket, :, target] += goal
            dirty_association_buckets.add(association_bucket)
            if self.context_order:
                context.append(target)
                del context[: -self.context_order]
        for association_bucket in dirty_association_buckets:
            self._association_cache.pop(association_bucket, None)
        self.episodes_seen += 1
        self.bytes_seen += len(response_bytes)

    def fit_episodes(self, episodes: Iterable[SignedEpisodeByteTrace]) -> None:
        """Fit supplied traces without joining bytes across episode boundaries."""

        for episode in episodes:
            self.fit_episode(
                episode.response,
                episode.goal_vector,
                prefix=episode.prefix,
            )

    def _association_matrix(self, association_bucket: int) -> np.ndarray:
        cached = self._association_cache.get(association_bucket)
        if cached is not None:
            return cached
        association_counts = self._association_counts[association_bucket].astype(
            np.float64
        )
        association_total = float(association_counts.sum())
        if association_total == 0.0:
            associations = np.zeros((self.goal_dimensions, 256), dtype=np.float64)
        else:
            goal_sums = self._goal_sums[association_bucket].astype(np.float64)
            bucket_mean = goal_sums.sum(axis=1) / association_total
            centered_sums = (
                goal_sums - bucket_mean[:, None] * association_counts[None, :]
            )
            associations = centered_sums / (
                association_counts[None, :] + self.association_prior_mass
            )
        associations.setflags(write=False)
        self._association_cache[association_bucket] = associations
        return associations

    def next_byte_distributions(
        self,
        context: ByteLike,
        goal_vector: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return matched ``(q0, qg)`` distributions without learned updates."""

        goal = self._goal_vector(goal_vector)
        local_bucket, association_bucket = self._buckets(context)
        local_counts = self._local_counts[local_bucket].astype(np.float64)
        local_total = float(local_counts.sum())
        local = (local_counts + self.local_prior_mass / 256.0) / (
            local_total + self.local_prior_mass
        )
        if not np.any(goal):
            return local, local.copy()

        associations = self._association_matrix(association_bucket)
        shift = self.goal_gain * (goal @ associations)
        shift = np.clip(shift, -self.logit_clip, self.logit_clip)
        if float(np.ptp(shift)) == 0.0:
            return local, local.copy()
        shift -= np.max(shift)
        conditioned = local * np.exp(shift)
        conditioned /= conditioned.sum()
        if self.minimum_probability:
            conditioned = (
                1.0 - 256.0 * self.minimum_probability
            ) * conditioned + self.minimum_probability
        return local, conditioned

    def trace_probabilities(
        self,
        prefix: ByteLike,
        candidate: ByteLike,
        goal_vector: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Teacher-force a candidate while holding its continuous goal fixed."""

        prefix_bytes = _as_bytes(prefix, name="prefix")
        candidate_bytes = _as_bytes(candidate, name="candidate")
        goal = self._goal_vector(goal_vector)
        local_trace = np.empty((len(candidate_bytes), 256), dtype=np.float64)
        goal_trace = np.empty_like(local_trace)
        context = bytearray(prefix_bytes[-self.context_order :])
        if self.context_order == 0:
            context.clear()
        for index, target in enumerate(candidate_bytes):
            local, conditioned = self.next_byte_distributions(context, goal)
            local_trace[index] = local
            goal_trace[index] = conditioned
            if self.context_order:
                context.append(target)
                del context[: -self.context_order]
        return local_trace, goal_trace

    def score_trace(
        self,
        prefix: ByteLike,
        candidate: ByteLike,
        goal_vector: Sequence[float] | np.ndarray,
    ) -> ByteTraceScore:
        """Score one observed byte trace under ``q0`` and continuous ``qg``."""

        candidate_bytes = _as_bytes(candidate, name="candidate")
        local, conditioned = self.trace_probabilities(
            prefix, candidate_bytes, goal_vector
        )
        if not candidate_bytes:
            return ByteTraceScore(0, 0.0, 0.0)
        indexes = np.arange(len(candidate_bytes))
        targets = np.frombuffer(candidate_bytes, dtype=np.uint8)
        return ByteTraceScore(
            len(candidate_bytes),
            float(np.log(local[indexes, targets]).sum()),
            float(np.log(conditioned[indexes, targets]).sum()),
        )

    def correct_base_distribution(
        self,
        base_distribution: Sequence[float] | np.ndarray,
        context: ByteLike,
        goal_vector: Sequence[float] | np.ndarray,
        *,
        beta: float,
        log_ratio_clip: float = 2.0,
    ) -> np.ndarray:
        """Apply the continuous head's likelihood ratio to any byte base."""

        local, conditioned = self.next_byte_distributions(context, goal_vector)
        # Preserve an exact matched-memory goal-off path even when beta is kept
        # at its goal-on value for the branch.
        effective_beta = 0.0 if np.array_equal(local, conditioned) else beta
        return likelihood_ratio_correction(
            base_distribution,
            local,
            conditioned,
            beta=effective_beta,
            log_ratio_clip=log_ratio_clip,
        )

    def snapshot(self) -> SignedByteHeadSnapshot:
        """Return a detached, read-only snapshot of all sufficient statistics."""

        local_counts = self._local_counts.copy()
        association_counts = self._association_counts.copy()
        goal_sums = self._goal_sums.copy()
        local_counts.setflags(write=False)
        association_counts.setflags(write=False)
        goal_sums.setflags(write=False)
        return SignedByteHeadSnapshot(
            version=self.SNAPSHOT_VERSION,
            goal_dimensions=self.goal_dimensions,
            context_order=self.context_order,
            context_bucket_count=self.context_bucket_count,
            association_bucket_count=self.association_bucket_count,
            local_prior_mass=self.local_prior_mass,
            association_prior_mass=self.association_prior_mass,
            goal_gain=self.goal_gain,
            max_goal_norm=self.max_goal_norm,
            logit_clip=self.logit_clip,
            minimum_probability=self.minimum_probability,
            episodes_seen=self.episodes_seen,
            bytes_seen=self.bytes_seen,
            local_counts=local_counts,
            association_counts=association_counts,
            goal_sums=goal_sums,
        )

    @classmethod
    def from_snapshot(
        cls, snapshot: SignedByteHeadSnapshot
    ) -> MatchedSignedGoalByteHead:
        """Construct an independent continuous head from a checked snapshot."""

        if snapshot.version != cls.SNAPSHOT_VERSION:
            raise ValueError(f"unsupported signed snapshot version {snapshot.version}")
        head = cls(
            snapshot.goal_dimensions,
            context_order=snapshot.context_order,
            context_bucket_count=snapshot.context_bucket_count,
            association_bucket_count=snapshot.association_bucket_count,
            local_prior_mass=snapshot.local_prior_mass,
            association_prior_mass=snapshot.association_prior_mass,
            goal_gain=snapshot.goal_gain,
            max_goal_norm=snapshot.max_goal_norm,
            logit_clip=snapshot.logit_clip,
            minimum_probability=snapshot.minimum_probability,
        )
        expected_local = (snapshot.context_bucket_count, 256)
        expected_association = (snapshot.association_bucket_count, 256)
        expected_goal = (
            snapshot.association_bucket_count,
            snapshot.goal_dimensions,
            256,
        )
        if snapshot.local_counts.shape != expected_local:
            raise ValueError("signed snapshot local-count shape is invalid")
        if snapshot.association_counts.shape != expected_association:
            raise ValueError("signed snapshot association-count shape is invalid")
        if snapshot.goal_sums.shape != expected_goal:
            raise ValueError("signed snapshot goal-sum shape is invalid")
        if snapshot.episodes_seen < 0 or snapshot.bytes_seen < 0:
            raise ValueError("signed snapshot counters must be non-negative")
        if (
            not np.all(np.isfinite(snapshot.local_counts))
            or not np.all(np.isfinite(snapshot.association_counts))
            or not np.all(np.isfinite(snapshot.goal_sums))
            or np.any(snapshot.local_counts < 0.0)
            or np.any(snapshot.association_counts < 0.0)
        ):
            raise ValueError("signed snapshot statistics are invalid")
        head._local_counts[...] = snapshot.local_counts
        head._association_counts[...] = snapshot.association_counts
        head._goal_sums[...] = snapshot.goal_sums
        head.episodes_seen = int(snapshot.episodes_seen)
        head.bytes_seen = int(snapshot.bytes_seen)
        return head
