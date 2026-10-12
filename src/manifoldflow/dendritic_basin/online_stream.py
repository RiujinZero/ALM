"""Bounded, strictly online dendritic stream learner.

This is the Phase 12 integration point for the mechanisms validated in
Phases 0-11.  Earlier experiments discovered relations by retaining a diary
of the stream and repeatedly sampling that diary.  That was useful for
isolating mechanisms, but it was neither a bounded-memory animal nor a
causal online learner.  This module never retains the stream.  Each relation
slot owns only a small local sample cache, local predictive counts, a local
interaction dendrite, and its own lifecycle statistics.

The organization is deliberately biological rather than layer-centric:

* sensory channels and older relation assemblies are possible afferents;
* a dendritic branch binds one current afferent to one eligibility trace;
* branches compete locally for the right to influence prediction;
* prediction error is a scalar surprise/modulatory signal, not a transported
  gradient;
* surprise recruits immature branches onto afferents active just before the
  surprising observation;
* useful branches consolidate a slowly decaying assembly state;
* weak, silent branches are recycled, keeping anatomy strictly bounded.

The implementation is pure NumPy.  No autograd-capable framework is imported,
preserving the package-wide hard no-backprop guarantee.
"""

from __future__ import annotations

import copy
import math
import pickle
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from manifoldflow.dendritic_basin.compiled_kernels import (
    NUMBA_AVAILABLE,
    adapt_slots_compiled,
    advance_dense_chunk_compiled,
    advance_dense_features_compiled,
    advance_response_batch_compiled,
    compete_predictors_compiled,
    counter_perturbations_compiled,
    interaction_objective_compiled,
    select_column_winners_compiled,
    update_interaction_stats_compiled,
    update_local_outcomes_compiled,
)
from manifoldflow.dendritic_basin.interaction_dendrite import (
    InteractionDendrite,
    RunningJointStats,
)
from manifoldflow.dendritic_basin.temporal_trace import lambda_from_halflife

JUVENILE = 0
CONSOLIDATED = 1
DORMANT = 2
LIFECYCLE_NAMES = ("juvenile", "consolidated", "dormant")


@dataclass(frozen=True)
class OnlineDendriticConfig:
    """Fixed anatomy and local plasticity constants for one stream learner."""

    vocab_size: int
    timescales: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    max_relations: int = 128
    max_nursery_candidates: int = 256
    max_alm_bytes: int | None = None
    max_active_relations: int = 8
    max_depth: int = 4
    column_size: int = 256
    max_active_columns: int = 4
    max_active_per_column: int = 2
    inter_column_birth_probability: float = 0.05
    local_buffer_size: int = 64
    base_decay: float = 0.9995
    context_decay: float = 1.0
    predictor_decay: float = 0.995
    utility_decay: float = 0.98
    trace_mean_decay: float = 0.995
    surprise_birth_bits: float = 4.0
    birth_cooldown: int = 4
    nursery_confirmations: int = 3
    nursery_target_consistency: float = 0.60
    nursery_max_age: int = 2048
    min_prune_age: int = 128
    max_staleness: int = 4096
    prune_utility: float = -0.05
    activation_threshold: float = 0.55
    support_threshold: float = 0.10
    initial_sigma: float = 0.20
    perturb_step: float = 0.08
    adapt_interval: int = 32
    min_adapt_samples: int = 24
    persistence_decay: float = 0.97
    persistence_utility_threshold: float = 0.05
    persistence_min_observations: int = 32
    consolidation_min_age: int = 16_384
    consolidation_min_fires: int = 64
    consolidation_utility_threshold: float = 0.05
    consolidated_adapt_interval: int = 256
    consolidated_predictor_decay: float = 0.9995
    dormancy_staleness: int = 65_536
    dormant_min_residence: int = 4_096
    byte_budget_check_interval: int = 10_000
    relation_prior: float = 0.25
    base_prior: float = 0.5
    context_prior: float = 0.1
    recursive_birth_probability: float = 0.25
    metabolic_backend: Literal["dense", "event"] = "dense"
    lifecycle_sweep_interval: int = 256
    outcome_backend: Literal["scalar", "compiled"] = "compiled"
    response_backend: Literal["scalar", "batched", "compiled"] = "compiled"
    adaptation_backend: Literal["scalar", "compiled"] = "compiled"
    adaptation_rng: Literal["global", "counter"] = "counter"
    competition_backend: Literal["scalar", "batched", "compiled"] = "batched"
    contextual_residual: bool = False
    utility_gated_persistence: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_alm_bytes is not None:
            if self.max_alm_bytes <= 0:
                raise ValueError("max_alm_bytes must be positive")
            capacity = self.relation_capacity_for_bytes(
                self.max_alm_bytes,
                vocab_size=self.vocab_size,
                timescale_count=len(self.timescales),
                local_buffer_size=self.local_buffer_size,
                contextual_residual=self.contextual_residual,
            )
            object.__setattr__(self, "max_relations", capacity)
            object.__setattr__(self, "max_nursery_candidates", capacity)
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if not self.timescales or any(t <= 0 for t in self.timescales):
            raise ValueError("timescales must contain positive values")
        for name in (
            "max_relations",
            "max_nursery_candidates",
            "max_active_relations",
            "max_depth",
            "column_size",
            "max_active_columns",
            "max_active_per_column",
            "local_buffer_size",
            "birth_cooldown",
            "nursery_confirmations",
            "nursery_max_age",
            "min_prune_age",
            "max_staleness",
            "adapt_interval",
            "min_adapt_samples",
            "persistence_min_observations",
            "consolidation_min_age",
            "consolidation_min_fires",
            "consolidated_adapt_interval",
            "dormancy_staleness",
            "dormant_min_residence",
            "byte_budget_check_interval",
            "lifecycle_sweep_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_active_relations > self.max_relations:
            raise ValueError("max_active_relations cannot exceed max_relations")
        for name in (
            "base_decay",
            "context_decay",
            "predictor_decay",
            "utility_decay",
            "trace_mean_decay",
            "persistence_decay",
            "consolidated_predictor_decay",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.base_prior <= 0.0 or self.context_prior <= 0.0:
            raise ValueError("base priors must be positive")
        if not 0.0 <= self.recursive_birth_probability <= 1.0:
            raise ValueError("recursive_birth_probability must be in [0, 1]")
        if not 0.0 <= self.inter_column_birth_probability <= 1.0:
            raise ValueError("inter_column_birth_probability must be in [0, 1]")
        if not 0.0 <= self.nursery_target_consistency <= 1.0:
            raise ValueError("nursery_target_consistency must be in [0, 1]")
        if self.response_backend not in ("scalar", "batched", "compiled"):
            raise ValueError(
                "response_backend must be 'scalar', 'batched', or 'compiled'"
            )
        if self.adaptation_backend not in ("scalar", "compiled"):
            raise ValueError("adaptation_backend must be 'scalar' or 'compiled'")
        if self.adaptation_rng not in ("global", "counter"):
            raise ValueError("adaptation_rng must be 'global' or 'counter'")
        if self.competition_backend not in ("scalar", "batched", "compiled"):
            raise ValueError(
                "competition_backend must be 'scalar', 'batched', or 'compiled'"
            )
        if self.metabolic_backend not in ("dense", "event"):
            raise ValueError("metabolic_backend must be 'dense' or 'event'")
        if self.outcome_backend not in ("scalar", "compiled"):
            raise ValueError("outcome_backend must be 'scalar' or 'compiled'")

    @staticmethod
    def relation_capacity_for_bytes(
        max_alm_bytes: int,
        *,
        vocab_size: int,
        timescale_count: int,
        local_buffer_size: int,
        contextual_residual: bool = False,
    ) -> int:
        """Conservative mature+nursery capacity for a durable byte class.

        The estimate reserves worst-case predictor distributions in the
        pending causal state, full mature local buffers, full nursery target
        evidence, traces, and structural serialization overhead. Exact
        checkpoint serialization is still checked before every write.
        """
        fixed_bytes = 65_536 + 16 * timescale_count * vocab_size
        if contextual_residual:
            fixed_bytes += 8 * vocab_size * vocab_size
        per_relation_bytes = (
            24 * vocab_size
            + 16 * local_buffer_size
            + 32 * timescale_count
            + 1_536
        )
        capacity = (max_alm_bytes - fixed_bytes) // per_relation_bytes
        if capacity < 1:
            raise ValueError(
                "max_alm_bytes is too small for one mature/nursery unit"
            )
        return int(capacity)

    @property
    def estimated_max_serialized_bytes(self) -> int | None:
        if self.max_alm_bytes is None:
            return None
        fixed_bytes = 65_536 + 16 * len(self.timescales) * self.vocab_size
        if self.contextual_residual:
            fixed_bytes += 8 * self.vocab_size * self.vocab_size
        per_relation_bytes = (
            24 * self.vocab_size
            + 16 * self.local_buffer_size
            + 32 * len(self.timescales)
            + 1_536
        )
        return fixed_bytes + self.max_relations * per_relation_bytes


@dataclass
class OnlineRelationSlot:
    """One dendritic microcircuit and all state available locally to it."""

    slot_id: int
    current_channel: int
    trace_channel: int
    timescale_index: int
    depth: int
    column_id: int
    dendrite: InteractionDendrite
    predictor_counts: np.ndarray
    state_predictor_counts: np.ndarray
    local_samples: deque[tuple[float, float]]
    born_step: int
    observations: int = 0
    fires: int = 0
    last_fire_step: int = -1
    utility_bits: float = 0.0
    state_observations: int = 0
    state_utility_bits: float = 0.0
    persistence: float = 0.0
    lifecycle: Literal["juvenile", "consolidated", "dormant"] = "juvenile"
    consolidated_step: int = -1
    dormant_step: int = -1

    def distribution(self, prior: float) -> np.ndarray:
        counts = self.predictor_counts + prior
        return counts / counts.sum()

    def confidence(self) -> float:
        """Bounded evidence strength; young branches influence gently."""
        return self.observations / (self.observations + 8.0)

    def state_distribution(self, prior: float) -> np.ndarray:
        counts = self.state_predictor_counts + prior
        return counts / counts.sum()

    def state_confidence(self) -> float:
        return self.state_observations / (self.state_observations + 16.0)


@dataclass
class DevelopmentalCandidate:
    """A provisional spine cluster waiting for repeated local evidence."""

    current_channel: int
    trace_channel: int
    timescale_index: int
    depth: int
    column_id: int
    mean_a: float
    mean_b: float
    target_counts: np.ndarray
    confirmations: int
    first_seen_step: int
    last_seen_step: int


@dataclass(frozen=True)
class StreamStep:
    """Prequential result after one observation has been incorporated."""

    prediction: np.ndarray
    observed_probability: float | None
    base_observed_probability: float | None
    surprise_bits: float | None
    active_relations: tuple[int, ...]
    born_relation: int | None
    pruned_relation: int | None


@dataclass
class OnlinePopulationStats:
    steps: int = 0
    predictions_scored: int = 0
    births: int = 0
    nursery_proposals: int = 0
    nursery_evictions: int = 0
    prunes: int = 0
    skipped_births: int = 0
    relation_firings: int = 0
    max_live_relations: int = 0
    consolidations: int = 0
    dormancies: int = 0
    reactivations: int = 0
    budget_evictions: int = 0
    branch_evaluations: int = 0


@dataclass
class _PendingPrediction:
    distribution: np.ndarray
    base_distribution: np.ndarray
    relation_ids: np.ndarray
    relation_is_state: np.ndarray
    active_relations: tuple[int, ...]
    features: np.ndarray
    traces: np.ndarray


@dataclass(frozen=True)
class _ResponseBatch:
    """Branches at one causal depth, represented as dense array indices."""

    slot_ids: np.ndarray
    current_channels: np.ndarray
    trace_channels: np.ndarray
    timescale_indices: np.ndarray


class OnlineDendriticStreamLearner:
    """Causal categorical stream learner with bounded structural plasticity.

    ``observe(token)`` first scores the prediction made before ``token`` was
    visible, performs only local updates, and then returns a prediction for
    the following token.  No method accepts a future target alongside the
    context that predicts it.
    """

    FORMAT = "dendritic-online-stream-v2"

    def __init__(self, config: OnlineDendriticConfig):
        self.config = config
        self.n_channels = config.vocab_size + config.max_relations
        self.lambdas = np.array(
            [lambda_from_halflife(t) for t in config.timescales],
            dtype=np.float64,
        )
        self.trace_scales = np.maximum(1.0 - self.lambdas, 1e-8)
        self.traces = np.zeros((len(config.timescales), self.n_channels))
        self.trace_means = np.full_like(self.traces, 0.05)
        self.base_counts = np.zeros(config.vocab_size)
        self.context_counts = (
            np.zeros((config.vocab_size, config.vocab_size))
            if config.contextual_residual
            else np.empty((0, 0))
        )
        self.previous_token = -1
        self.slots: list[OnlineRelationSlot | None] = [
            None for _ in range(config.max_relations)
        ]
        self.nursery: dict[tuple[int, int, int], DevelopmentalCandidate] = {}
        self.stats = OnlinePopulationStats()
        self.rng = np.random.default_rng(config.seed)
        self._pending: _PendingPrediction | None = None
        self._last_birth_step = -config.birth_cooldown
        self._response_batches_cache: tuple[_ResponseBatch, ...] | None = None
        self._event_consumers_cache: tuple[dict[int, np.ndarray], ...] | None = (
            None
        )
        self._persistent_active: set[int] = set()
        self._persistent_active_by_depth: list[set[int]] = [
            set() for _ in range(config.max_depth)
        ]
        self._w_a = np.zeros(config.max_relations)
        self._w_b = np.zeros(config.max_relations)
        self._mu = np.zeros(config.max_relations)
        self._sigma = np.ones(config.max_relations)
        self._persistence = np.zeros(config.max_relations)
        self._state_observations = np.zeros(config.max_relations, dtype=np.int64)
        self._state_utility_bits = np.zeros(config.max_relations)
        self._live = np.zeros(config.max_relations, dtype=np.bool_)
        self._current_channels = np.full(config.max_relations, -1, dtype=np.int64)
        self._trace_channels = np.full(config.max_relations, -1, dtype=np.int64)
        self._timescale_indices = np.full(
            config.max_relations, -1, dtype=np.int64
        )
        self._depths = np.zeros(config.max_relations, dtype=np.int64)
        self._observations = np.zeros(config.max_relations, dtype=np.int64)
        self._fires = np.zeros(config.max_relations, dtype=np.int64)
        self._utility_bits = np.zeros(config.max_relations)
        self._last_fire_steps = np.full(config.max_relations, -1, dtype=np.int64)
        self._born_steps = np.full(config.max_relations, -1, dtype=np.int64)
        self._lifecycle = np.full(
            config.max_relations, JUVENILE, dtype=np.uint8
        )
        self._consolidated_steps = np.full(
            config.max_relations, -1, dtype=np.int64
        )
        self._dormant_steps = np.full(
            config.max_relations, -1, dtype=np.int64
        )
        matrix_shape = (config.max_relations, config.vocab_size)
        self._predictor_counts = np.zeros(matrix_shape)
        self._state_predictor_counts = np.zeros(matrix_shape)
        self._local_samples = np.zeros(
            (config.max_relations, config.local_buffer_size, 2)
        )
        self._local_sample_counts = np.zeros(
            config.max_relations, dtype=np.int64
        )
        self._local_sample_positions = np.zeros(
            config.max_relations, dtype=np.int64
        )
        self._stats_n = np.zeros(config.max_relations, dtype=np.int64)
        self._stats_mean_a = np.zeros(config.max_relations)
        self._stats_mean_b = np.zeros(config.max_relations)
        self._stats_mean_d = np.zeros(config.max_relations)
        self._stats_m2_a = np.zeros(config.max_relations)
        self._stats_m2_b = np.zeros(config.max_relations)
        self._stats_m2_d = np.zeros(config.max_relations)
        self._stats_cov_da_acc = np.zeros(config.max_relations)
        self._stats_cov_db_acc = np.zeros(config.max_relations)

    @property
    def live_relation_count(self) -> int:
        return int(np.count_nonzero(self._live))

    @property
    def live_nursery_count(self) -> int:
        return len(self.nursery)

    @property
    def column_count(self) -> int:
        return math.ceil(self.config.max_relations / self.config.column_size)

    def _column_for_parents(
        self, current_channel: int, trace_channel: int, timescale_index: int
    ) -> tuple[int, bool]:
        parent_columns = {
            (channel - self.config.vocab_size) // self.config.column_size
            for channel in (current_channel, trace_channel)
            if channel >= self.config.vocab_size
        }
        if parent_columns:
            return max(parent_columns), len(parent_columns) > 1
        niche_hash = (
            257 * current_channel + 17 * trace_channel + 65537 * timescale_index
        )
        return niche_hash % self.column_count, False

    def _base_distribution(self) -> np.ndarray:
        if self.config.contextual_residual and self.previous_token >= 0:
            counts = (
                self.context_counts[self.previous_token]
                + self.config.context_prior
            )
        else:
            counts = self.base_counts + self.config.base_prior
        return counts / counts.sum()

    def _normalized_traces(self) -> np.ndarray:
        # The design's q <- lambda*q+x accumulator has a timescale-dependent
        # range.  Multiplying by (1-lambda) turns it into a bounded local
        # activity estimate without changing its causal information.
        return np.clip(self.trace_scales[:, None] * self.traces, 0.0, 1.0)

    def _reset_dynamic_state(self) -> None:
        self.traces.fill(0.0)
        self.trace_means.fill(0.05)
        self._persistence.fill(0.0)
        self._pending = None
        self.previous_token = -1
        self._persistent_active.clear()
        for active_ids in self._persistent_active_by_depth:
            active_ids.clear()
        for slot in self.slots:
            if slot is not None:
                slot.persistence = 0.0

    def reset_episode(self) -> None:
        """Clear transient/persistent activity, retaining learned anatomy."""
        self._reset_dynamic_state()

    def _slot_parent_value(
        self, features: np.ndarray, channel: int, slot_id: int
    ) -> float:
        if channel >= self.config.vocab_size:
            parent_slot = channel - self.config.vocab_size
            if parent_slot >= slot_id:
                return 0.0
        return float(features[channel])

    def _invalidate_response_batches(self) -> None:
        self._response_batches_cache = None
        self._event_consumers_cache = None

    def _sync_slot_parameters(self, slot_id: int) -> None:
        slot = self.slots[slot_id]
        if slot is None:
            self._persistent_active.discard(slot_id)
            self._w_a[slot_id] = 0.0
            self._w_b[slot_id] = 0.0
            self._mu[slot_id] = 0.0
            self._sigma[slot_id] = 1.0
            return
        self._w_a[slot_id] = slot.dendrite.w_a
        self._w_b[slot_id] = slot.dendrite.w_b
        self._mu[slot_id] = slot.dendrite.mu
        self._sigma[slot_id] = slot.dendrite.sigma

    def _sync_slot_evidence(self, slot_id: int) -> None:
        """Copy durable Welford evidence into the contiguous execution state."""
        slot = self.slots[slot_id]
        if slot is None:
            stats = RunningJointStats()
        else:
            stats = slot.dendrite.stats
        self._stats_n[slot_id] = stats.n
        self._stats_mean_a[slot_id] = stats.mean_a
        self._stats_mean_b[slot_id] = stats.mean_b
        self._stats_mean_d[slot_id] = stats.mean_d
        self._stats_m2_a[slot_id] = stats.m2_a
        self._stats_m2_b[slot_id] = stats.m2_b
        self._stats_m2_d[slot_id] = stats.m2_d
        self._stats_cov_da_acc[slot_id] = stats.cov_da_acc
        self._stats_cov_db_acc[slot_id] = stats.cov_db_acc

    def _sync_slot_runtime(self, slot_id: int) -> None:
        slot = self.slots[slot_id]
        if slot is None:
            self._persistent_active.discard(slot_id)
            for active_ids in self._persistent_active_by_depth:
                active_ids.discard(slot_id)
            self._persistence[slot_id] = 0.0
            self._state_observations[slot_id] = 0
            self._state_utility_bits[slot_id] = 0.0
            self._live[slot_id] = False
            self._current_channels[slot_id] = -1
            self._trace_channels[slot_id] = -1
            self._timescale_indices[slot_id] = -1
            self._depths[slot_id] = 0
            self._observations[slot_id] = 0
            self._fires[slot_id] = 0
            self._utility_bits[slot_id] = 0.0
            self._last_fire_steps[slot_id] = -1
            self._born_steps[slot_id] = -1
            self._lifecycle[slot_id] = JUVENILE
            self._consolidated_steps[slot_id] = -1
            self._dormant_steps[slot_id] = -1
            self._predictor_counts[slot_id].fill(0.0)
            self._state_predictor_counts[slot_id].fill(0.0)
            self._local_samples[slot_id].fill(0.0)
            self._local_sample_counts[slot_id] = 0
            self._local_sample_positions[slot_id] = 0
            return
        self._persistence[slot_id] = slot.persistence
        for active_ids in self._persistent_active_by_depth:
            active_ids.discard(slot_id)
        self._state_observations[slot_id] = slot.state_observations
        self._state_utility_bits[slot_id] = slot.state_utility_bits
        self._live[slot_id] = True
        self._current_channels[slot_id] = slot.current_channel
        self._trace_channels[slot_id] = slot.trace_channel
        self._timescale_indices[slot_id] = slot.timescale_index
        self._depths[slot_id] = slot.depth
        self._set_persistent_active(
            slot_id, slot.persistence > self.config.support_threshold
        )
        self._observations[slot_id] = slot.observations
        self._fires[slot_id] = slot.fires
        self._utility_bits[slot_id] = slot.utility_bits
        self._last_fire_steps[slot_id] = slot.last_fire_step
        self._born_steps[slot_id] = slot.born_step
        self._lifecycle[slot_id] = LIFECYCLE_NAMES.index(slot.lifecycle)
        self._consolidated_steps[slot_id] = slot.consolidated_step
        self._dormant_steps[slot_id] = slot.dormant_step
        self._predictor_counts[slot_id] = slot.predictor_counts
        self._state_predictor_counts[slot_id] = slot.state_predictor_counts
        samples = list(slot.local_samples)
        sample_count = min(len(samples), self.config.local_buffer_size)
        self._local_samples[slot_id].fill(0.0)
        if sample_count:
            self._local_samples[slot_id, :sample_count] = np.asarray(
                samples[-sample_count:], dtype=np.float64
            )
        self._local_sample_counts[slot_id] = sample_count
        self._local_sample_positions[slot_id] = (
            sample_count % self.config.local_buffer_size
        )
        slot.predictor_counts = self._predictor_counts[slot_id]
        slot.state_predictor_counts = self._state_predictor_counts[slot_id]

    def _ordered_local_samples(self, slot_id: int) -> np.ndarray:
        count = int(self._local_sample_counts[slot_id])
        if count < self.config.local_buffer_size:
            return self._local_samples[slot_id, :count].copy()
        position = int(self._local_sample_positions[slot_id])
        return np.concatenate(
            (
                self._local_samples[slot_id, position:],
                self._local_samples[slot_id, :position],
            )
        )

    def _response_batches(self) -> tuple[_ResponseBatch, ...]:
        if self._response_batches_cache is not None:
            return self._response_batches_cache
        batches = []
        for depth in range(1, self.config.max_depth + 1):
            slots = [
                slot
                for slot in self.slots
                if slot is not None and slot.depth == depth
            ]
            if not slots:
                continue
            batches.append(
                _ResponseBatch(
                    slot_ids=np.asarray(
                        [slot.slot_id for slot in slots], dtype=np.intp
                    ),
                    current_channels=np.asarray(
                        [slot.current_channel for slot in slots], dtype=np.intp
                    ),
                    trace_channels=np.asarray(
                        [slot.trace_channel for slot in slots], dtype=np.intp
                    ),
                    timescale_indices=np.asarray(
                        [slot.timescale_index for slot in slots], dtype=np.intp
                    ),
                )
            )
        self._response_batches_cache = tuple(batches)
        return self._response_batches_cache

    def _event_consumers(self) -> tuple[dict[int, np.ndarray], ...]:
        """Index branch consumers by causal depth and current afferent."""
        if self._event_consumers_cache is not None:
            return self._event_consumers_cache
        depth_maps: list[dict[int, list[int]]] = [
            {} for _ in range(self.config.max_depth)
        ]
        for slot in self.slots:
            if slot is None:
                continue
            consumers = depth_maps[slot.depth - 1]
            consumers.setdefault(slot.current_channel, []).append(slot.slot_id)
        self._event_consumers_cache = tuple(
            {
                channel: np.asarray(slot_ids, dtype=np.intp)
                for channel, slot_ids in consumers.items()
            }
            for consumers in depth_maps
        )
        return self._event_consumers_cache

    def _batch_for_slot_ids(self, slot_ids: np.ndarray) -> _ResponseBatch:
        return _ResponseBatch(
            slot_ids=slot_ids,
            current_channels=self._current_channels[slot_ids],
            trace_channels=self._trace_channels[slot_ids],
            timescale_indices=self._timescale_indices[slot_ids],
        )

    def _set_persistent_active(self, slot_id: int, active: bool) -> None:
        depth = int(self._depths[slot_id])
        if active:
            self._persistent_active.add(slot_id)
            if depth > 0:
                self._persistent_active_by_depth[depth - 1].add(slot_id)
        else:
            self._persistent_active.discard(slot_id)
            if depth > 0:
                self._persistent_active_by_depth[depth - 1].discard(slot_id)

    def _batch_responses(
        self,
        batch: _ResponseBatch,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gather one causal stratum and evaluate its branches densely."""
        a = features[batch.current_channels]
        b = np.minimum(
            1.0,
            self.trace_scales[batch.timescale_indices]
            * self.traces[batch.timescale_indices, batch.trace_channels],
        )
        response = np.zeros(len(batch.slot_ids))
        supported = a > self.config.support_threshold
        if not np.any(supported):
            return a, b, response

        if self.config.response_backend == "scalar":
            for position in np.flatnonzero(supported):
                slot = self.slots[int(batch.slot_ids[position])]
                assert slot is not None
                response[position] = slot.dendrite.respond(
                    float(a[position]), float(b[position])
                )
            return a, b, response

        ids = batch.slot_ids[supported]
        drive = self._w_a[ids] * a[supported] + self._w_b[ids] * b[supported]
        response[supported] = np.exp(
            -((drive - self._mu[ids]) ** 2) / (2.0 * self._sigma[ids] ** 2)
        )
        return a, b, response

    def _advance_response_batch(
        self, batch: _ResponseBatch, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance response and retained state for one causal stratum."""
        if self.effective_response_backend == "compiled":
            return advance_response_batch_compiled(
                batch.slot_ids,
                batch.current_channels,
                batch.trace_channels,
                batch.timescale_indices,
                features,
                self.trace_scales,
                self.traces,
                self._w_a,
                self._w_b,
                self._mu,
                self._sigma,
                self._persistence,
                self._state_observations,
                self._state_utility_bits,
                self.config.vocab_size,
                self.config.support_threshold,
                self.config.activation_threshold,
                self.config.persistence_decay,
                self.config.persistence_min_observations,
                self.config.persistence_utility_threshold,
            )

        a_values, b_values, responses = self._batch_responses(batch, features)
        ids = batch.slot_ids
        event_active = responses >= self.config.activation_threshold
        persistence = self._persistence[ids]
        persistence = np.where(
            event_active,
            responses,
            persistence * self.config.persistence_decay,
        )
        self._persistence[ids] = persistence
        state_active = np.logical_and(
            ~event_active,
            persistence >= self.config.activation_threshold,
        )
        state_probes = np.where(state_active, persistence, 0.0)
        observations = self._state_observations[ids]
        state_eligible = np.logical_and(
            observations >= self.config.persistence_min_observations,
            self._state_utility_bits[ids]
            >= self.config.persistence_utility_threshold,
        )
        state_confidence = observations / (observations + 16.0)
        persistent_activation = np.where(
            state_eligible, persistence * state_confidence, 0.0
        )
        activation = np.maximum(responses, persistent_activation)
        relation_channels = self.config.vocab_size + ids
        features[relation_channels] = activation
        self.traces[:, relation_channels] += activation
        return a_values, b_values, responses, state_probes

    @property
    def effective_adaptation_backend(self) -> str:
        if self.config.adaptation_backend == "compiled" and NUMBA_AVAILABLE:
            return "compiled"
        return "scalar"

    @property
    def effective_response_backend(self) -> str:
        if self.config.response_backend == "compiled" and NUMBA_AVAILABLE:
            return "compiled"
        if self.config.response_backend == "scalar":
            return "scalar"
        return "batched"

    @property
    def effective_outcome_backend(self) -> str:
        if self.config.outcome_backend == "compiled" and NUMBA_AVAILABLE:
            return "compiled"
        return "scalar"

    @property
    def effective_competition_backend(self) -> str:
        if self.config.competition_backend == "compiled" and NUMBA_AVAILABLE:
            return "compiled"
        if self.config.competition_backend == "scalar":
            return "scalar"
        return "batched"

    def warm_execution_backend(self) -> None:
        """Compile optional local kernels before a timed stream begins."""
        if self.effective_adaptation_backend == "compiled":
            interaction_objective_compiled(
                np.zeros((1, 2), dtype=np.float64), 1.0, 1.0, 0.0, 1.0
            )
            update_interaction_stats_compiled(
                np.zeros(0, dtype=np.int64),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1, dtype=np.int64),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
            )
            adapt_slots_compiled(
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 4)),
                self._local_samples,
                self._local_sample_counts,
                self._local_sample_positions,
                self._w_a,
                self._w_b,
                self._mu,
                self._sigma,
                self.config.initial_sigma / 4.0,
            )
            counter_perturbations_compiled(
                np.zeros(0, dtype=np.int64),
                self._observations,
                self.config.seed,
                self.config.perturb_step,
            )
            select_column_winners_compiled(
                np.ones(1),
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                1,
                1,
                1,
                1,
                1,
            )
        if self.effective_outcome_backend == "compiled":
            update_local_outcomes_compiled(
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.bool_),
                0,
                1.0,
                self.config.relation_prior,
                self.config.utility_decay,
                self.config.predictor_decay,
                self.config.consolidated_predictor_decay,
                CONSOLIDATED,
                np.zeros(1, dtype=np.uint8),
                np.zeros((1, self.config.vocab_size)),
                np.zeros((1, self.config.vocab_size)),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1, dtype=np.int64),
            )
        if self.effective_response_backend == "compiled":
            advance_dense_features_compiled(
                0,
                np.zeros(0, dtype=np.int64),
                self._current_channels,
                self._trace_channels,
                self._timescale_indices,
                self.lambdas,
                self.trace_scales,
                np.zeros_like(self.traces),
                np.zeros_like(self.trace_means),
                self._w_a,
                self._w_b,
                self._mu,
                self._sigma,
                self._persistence.copy(),
                self._state_observations,
                self._state_utility_bits,
                self.config.vocab_size,
                self.config.max_relations,
                self.config.support_threshold,
                self.config.activation_threshold,
                self.config.persistence_decay,
                self.config.persistence_min_observations,
                self.config.persistence_utility_threshold,
                self.config.trace_mean_decay,
                True,
            )
            advance_response_batch_compiled(
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(self.n_channels),
                self.trace_scales,
                np.zeros_like(self.traces),
                self._w_a,
                self._w_b,
                self._mu,
                self._sigma,
                self._persistence.copy(),
                self._state_observations,
                self._state_utility_bits,
                self.config.vocab_size,
                self.config.support_threshold,
                self.config.activation_threshold,
                self.config.persistence_decay,
                self.config.persistence_min_observations,
                self.config.persistence_utility_threshold,
            )
        if self.effective_competition_backend == "compiled":
            compete_predictors_compiled(
                np.zeros(0, dtype=np.int64),
                np.zeros(self.n_channels),
                np.zeros(self.config.max_relations),
                np.zeros(self.config.max_relations),
                self._live,
                self._lifecycle,
                self._observations,
                self._fires,
                self._utility_bits,
                self._state_observations,
                self._state_utility_bits,
                self._predictor_counts,
                self._state_predictor_counts,
                self.base_counts,
                self.context_counts,
                self.previous_token,
                self.config.contextual_residual,
                self.config.vocab_size,
                DORMANT,
                self.config.activation_threshold,
                self.config.relation_prior,
                self.config.base_prior,
                self.config.context_prior,
                self.config.column_size,
                self.column_count,
                self.config.max_active_per_column,
                self.config.max_active_columns,
                self.config.max_active_relations,
            )
        if self.supports_compiled_chunks:
            # Compile the full sequential signature outside measured corpus
            # time, then restore every causal and structural bit exactly.
            snapshot = self.state_dict()
            self.observe_chunk_probabilities(
                np.zeros(1, dtype=np.int64), learn=True
            )
            self.load_state_dict(snapshot)

    def _adapt_slot(self, slot_id: int, slot: OnlineRelationSlot) -> None:
        samples = self._ordered_local_samples(slot_id)
        if self.effective_adaptation_backend == "scalar":
            slot.dendrite.perturb_and_select(
                [tuple(row) for row in samples],
                self.config.perturb_step,
                self.rng,
            )
            return

        candidate_w_a = slot.dendrite.w_a + float(
            self.rng.normal(0.0, self.config.perturb_step)
        )
        candidate_w_b = slot.dendrite.w_b + float(
            self.rng.normal(0.0, self.config.perturb_step)
        )
        candidate_mu = slot.dendrite.mu + float(
            self.rng.normal(0.0, self.config.perturb_step)
        )
        candidate_sigma = max(
            1e-3,
            slot.dendrite.sigma
            + float(self.rng.normal(0.0, self.config.perturb_step)),
        )
        current_j = interaction_objective_compiled(
            samples,
            slot.dendrite.w_a,
            slot.dendrite.w_b,
            slot.dendrite.mu,
            slot.dendrite.sigma,
        )
        candidate_j = interaction_objective_compiled(
            samples,
            candidate_w_a,
            candidate_w_b,
            candidate_mu,
            candidate_sigma,
        )
        if candidate_j > current_j:
            slot.dendrite.w_a = candidate_w_a
            slot.dendrite.w_b = candidate_w_b
            slot.dendrite.mu = candidate_mu
            slot.dendrite.sigma = candidate_sigma

    def _adapt_slots(self, slot_ids: np.ndarray) -> None:
        """Adapt local branches while preserving anatomical RNG ordering."""
        if len(slot_ids) == 0:
            return
        if self.config.adaptation_rng == "counter":
            perturbations = counter_perturbations_compiled(
                slot_ids,
                self._observations,
                self.config.seed,
                self.config.perturb_step,
            )
        else:
            perturbations = self.rng.normal(
                0.0,
                self.config.perturb_step,
                size=(len(slot_ids), 4),
            )
        if self.effective_adaptation_backend == "compiled":
            adapt_slots_compiled(
                slot_ids,
                perturbations,
                self._local_samples,
                self._local_sample_counts,
                self._local_sample_positions,
                self._w_a,
                self._w_b,
                self._mu,
                self._sigma,
                self.config.initial_sigma / 4.0,
            )
            for raw_slot_id in slot_ids:
                slot_id = int(raw_slot_id)
                slot = self.slots[slot_id]
                assert slot is not None
                slot.dendrite.w_a = float(self._w_a[slot_id])
                slot.dendrite.w_b = float(self._w_b[slot_id])
                slot.dendrite.mu = float(self._mu[slot_id])
                slot.dendrite.sigma = float(self._sigma[slot_id])
            return
        for row, raw_slot_id in enumerate(slot_ids):
            slot_id = int(raw_slot_id)
            slot = self.slots[slot_id]
            assert slot is not None
            samples = self._ordered_local_samples(slot_id)
            proposal = perturbations[row]
            candidate_w_a = slot.dendrite.w_a + float(proposal[0])
            candidate_w_b = slot.dendrite.w_b + float(proposal[1])
            candidate_mu = slot.dendrite.mu + float(proposal[2])
            candidate_sigma = max(
                1e-3, slot.dendrite.sigma + float(proposal[3])
            )
            current_j = interaction_objective_compiled(
                samples,
                slot.dendrite.w_a,
                slot.dendrite.w_b,
                slot.dendrite.mu,
                slot.dendrite.sigma,
            )
            candidate_j = interaction_objective_compiled(
                samples,
                candidate_w_a,
                candidate_w_b,
                candidate_mu,
                candidate_sigma,
            )
            if candidate_j > current_j:
                slot.dendrite.w_a = candidate_w_a
                slot.dendrite.w_b = candidate_w_b
                slot.dendrite.mu = candidate_mu
                slot.dendrite.sigma = candidate_sigma
            slot.dendrite.sigma = max(
                self.config.initial_sigma / 4.0, slot.dendrite.sigma
            )
            self._sync_slot_parameters(slot_id)

    def _materialize_slot_views(self) -> None:
        """Refresh inspectable objects from the contiguous execution state."""
        for raw_slot_id in np.flatnonzero(self._live):
            slot_id = int(raw_slot_id)
            slot = self.slots[slot_id]
            assert slot is not None
            slot.dendrite.w_a = float(self._w_a[slot_id])
            slot.dendrite.w_b = float(self._w_b[slot_id])
            slot.dendrite.mu = float(self._mu[slot_id])
            slot.dendrite.sigma = float(self._sigma[slot_id])
            slot.dendrite.stats = RunningJointStats(
                n=int(self._stats_n[slot_id]),
                mean_a=float(self._stats_mean_a[slot_id]),
                mean_b=float(self._stats_mean_b[slot_id]),
                mean_d=float(self._stats_mean_d[slot_id]),
                m2_a=float(self._stats_m2_a[slot_id]),
                m2_b=float(self._stats_m2_b[slot_id]),
                m2_d=float(self._stats_m2_d[slot_id]),
                cov_da_acc=float(self._stats_cov_da_acc[slot_id]),
                cov_db_acc=float(self._stats_cov_db_acc[slot_id]),
            )

    def _advance_features_dense(
        self, token_id: int, *, learn: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        live_ids = np.flatnonzero(self._live)
        normalized: np.ndarray | None = None
        if self.effective_response_backend == "compiled":
            (
                features,
                normalized,
                raw_responses,
                state_probes,
                current_values,
                trace_values,
            ) = advance_dense_features_compiled(
                token_id,
                live_ids,
                self._current_channels,
                self._trace_channels,
                self._timescale_indices,
                self.lambdas,
                self.trace_scales,
                self.traces,
                self.trace_means,
                self._w_a,
                self._w_b,
                self._mu,
                self._sigma,
                self._persistence,
                self._state_observations,
                self._state_utility_bits,
                self.config.vocab_size,
                self.config.max_relations,
                self.config.support_threshold,
                self.config.activation_threshold,
                self.config.persistence_decay,
                self.config.persistence_min_observations,
                self.config.persistence_utility_threshold,
                self.config.trace_mean_decay,
                learn,
            )
        else:
            features = np.zeros(self.n_channels)
            features[token_id] = 1.0
            raw_responses = np.zeros(self.config.max_relations)
            state_probes = np.zeros(self.config.max_relations)
            current_values = np.zeros(self.config.max_relations)
            trace_values = np.zeros(self.config.max_relations)

            # Decay all eligibility traces, add the current sensory event, then
            # evaluate relation slots in birth/topological order.  A newer branch
            # may therefore read an older assembly, but never itself or the future.
            self.traces *= self.lambdas[:, None]
            self.traces[:, token_id] += 1.0

            # A branch can only read lower-depth relation assemblies.  Responses
            # within one depth are therefore independent and can be gathered as a
            # dense cortical stratum without introducing recurrence or changing
            # the local learning rule.
            for batch in self._response_batches():
                a_values, b_values, responses, batch_state_probes = (
                    self._advance_response_batch(batch, features)
                )
                ids = batch.slot_ids
                current_values[ids] = a_values
                trace_values[ids] = b_values
                raw_responses[ids] = responses
                state_probes[ids] = batch_state_probes

        # Plasticity remains branch-local and is applied in anatomical slot
        # order, preserving deterministic perturbation/RNG order.
        if learn:
            self.stats.branch_evaluations += len(live_ids)
            firing_ids = live_ids[
                raw_responses[live_ids] >= self.config.activation_threshold
            ]
            self._last_fire_steps[firing_ids] = self.stats.steps
            reactivated_ids = firing_ids[
                np.logical_and(
                    self._lifecycle[firing_ids] == DORMANT,
                    self._utility_bits[firing_ids]
                    >= self.config.consolidation_utility_threshold,
                )
            ]
            if len(reactivated_ids):
                self._lifecycle[reactivated_ids] = CONSOLIDATED
                self._dormant_steps[reactivated_ids] = -1
                self.stats.reactivations += len(reactivated_ids)

            ages = self.stats.steps - self._born_steps[live_ids]
            consolidating_ids = live_ids[
                np.logical_and.reduce(
                    (
                        self._lifecycle[live_ids] == JUVENILE,
                        ages >= self.config.consolidation_min_age,
                        self._fires[live_ids]
                        >= self.config.consolidation_min_fires,
                        self._utility_bits[live_ids]
                        >= self.config.consolidation_utility_threshold,
                    )
                )
            ]
            if len(consolidating_ids):
                self._lifecycle[consolidating_ids] = CONSOLIDATED
                self._consolidated_steps[consolidating_ids] = self.stats.steps
                self.stats.consolidations += len(consolidating_ids)

            dormancy_ids = live_ids[
                np.logical_and(
                    self._lifecycle[live_ids] == CONSOLIDATED,
                    np.logical_or(
                        self.stats.steps - self._last_fire_steps[live_ids]
                        >= self.config.dormancy_staleness,
                        self._utility_bits[live_ids]
                        < self.config.prune_utility,
                    ),
                )
            ]
            if len(dormancy_ids):
                self._lifecycle[dormancy_ids] = DORMANT
                self._dormant_steps[dormancy_ids] = self.stats.steps
                self.stats.dormancies += len(dormancy_ids)

            plastic_ids = live_ids[self._lifecycle[live_ids] != DORMANT]
            positions = self._local_sample_positions[plastic_ids]
            self._local_samples[plastic_ids, positions, 0] = current_values[
                plastic_ids
            ]
            self._local_samples[plastic_ids, positions, 1] = trace_values[
                plastic_ids
            ]
            self._local_sample_positions[plastic_ids] = (
                positions + 1
            ) % self.config.local_buffer_size
            self._local_sample_counts[plastic_ids] = np.minimum(
                self.config.local_buffer_size,
                self._local_sample_counts[plastic_ids] + 1,
            )
            self._observations[plastic_ids] += 1

            supported_ids = plastic_ids[
                current_values[plastic_ids] > self.config.support_threshold
            ]
            update_interaction_stats_compiled(
                supported_ids,
                current_values,
                trace_values,
                raw_responses,
                self._stats_n,
                self._stats_mean_a,
                self._stats_mean_b,
                self._stats_mean_d,
                self._stats_m2_a,
                self._stats_m2_b,
                self._stats_m2_d,
                self._stats_cov_da_acc,
                self._stats_cov_db_acc,
            )

            self._fires[firing_ids] += 1
            self.stats.relation_firings += len(firing_ids)

            normal_interval = self.config.adapt_interval
            consolidated_interval = self.config.consolidated_adapt_interval
            adapt_intervals = np.where(
                self._lifecycle[plastic_ids] == CONSOLIDATED,
                consolidated_interval,
                normal_interval,
            )
            adapting_ids = plastic_ids[
                np.logical_and(
                    self._observations[plastic_ids] % adapt_intervals == 0,
                    self._local_sample_counts[plastic_ids]
                    >= self.config.min_adapt_samples,
                )
            ]
            # A vanishingly narrow receptive field is episodic memorization,
            # not a reusable dendritic branch.
            self._adapt_slots(adapting_ids)

        if normalized is None:
            normalized = self._normalized_traces()
            if learn:
                self.trace_means = (
                    self.config.trace_mean_decay * self.trace_means
                    + (1.0 - self.config.trace_mean_decay) * normalized
                )
        return features, normalized, raw_responses, state_probes, live_ids

    def _advance_features_event(
        self, token_id: int, *, learn: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance only branches reached by afferent events or live persistence."""
        features = np.zeros(self.n_channels)
        features[token_id] = 1.0
        raw_responses = np.zeros(self.config.max_relations)
        state_probes = np.zeros(self.config.max_relations)
        current_values = np.zeros(self.config.max_relations)
        trace_values = np.zeros(self.config.max_relations)

        self.traces *= self.lambdas[:, None]
        self.traces[:, token_id] += 1.0
        evaluated: set[int] = set()
        consumers_by_depth = self._event_consumers()
        scheduled: list[set[int]] = [
            set(active_ids) for active_ids in self._persistent_active_by_depth
        ]
        for depth_index, consumers in enumerate(consumers_by_depth):
            token_consumers = consumers.get(token_id)
            if token_consumers is not None:
                scheduled[depth_index].update(
                    int(slot_id) for slot_id in token_consumers
                )

        for depth_index, _consumers in enumerate(consumers_by_depth):
            selected = scheduled[depth_index]
            if not selected:
                continue

            ids = np.asarray(sorted(selected), dtype=np.intp)
            batch = self._batch_for_slot_ids(ids)
            a_values, b_values, responses, batch_state_probes = (
                self._advance_response_batch(batch, features)
            )
            current_values[ids] = a_values
            trace_values[ids] = b_values
            raw_responses[ids] = responses
            state_probes[ids] = batch_state_probes
            evaluated.update(selected)
            persistence = self._persistence[ids]
            relation_channels = self.config.vocab_size + ids
            activation = features[relation_channels]

            for slot_id, value in zip(ids, persistence, strict=True):
                useful_state = (
                    not self.config.utility_gated_persistence
                    or self._state_observations[slot_id]
                    < self.config.persistence_min_observations
                    or self._state_utility_bits[slot_id]
                    >= self.config.persistence_utility_threshold
                )
                self._set_persistent_active(
                    int(slot_id),
                    value > self.config.support_threshold and useful_state,
                )
            for channel, value in zip(
                relation_channels, activation, strict=True
            ):
                if value <= self.config.support_threshold:
                    continue
                for downstream_depth in range(
                    depth_index + 1, self.config.max_depth
                ):
                    downstream = consumers_by_depth[downstream_depth].get(
                        int(channel)
                    )
                    if downstream is not None:
                        scheduled[downstream_depth].update(
                            int(slot_id) for slot_id in downstream
                        )

        evaluated_ids = np.asarray(sorted(evaluated), dtype=np.intp)
        if learn:
            self.stats.branch_evaluations += len(evaluated_ids)
            firing_ids = evaluated_ids[
                raw_responses[evaluated_ids] >= self.config.activation_threshold
            ]
            self._last_fire_steps[firing_ids] = self.stats.steps
            reactivated_ids = firing_ids[
                np.logical_and(
                    self._lifecycle[firing_ids] == DORMANT,
                    self._utility_bits[firing_ids]
                    >= self.config.consolidation_utility_threshold,
                )
            ]
            if len(reactivated_ids):
                self._lifecycle[reactivated_ids] = CONSOLIDATED
                self._dormant_steps[reactivated_ids] = -1
                self.stats.reactivations += len(reactivated_ids)

            if self.stats.steps % self.config.lifecycle_sweep_interval == 0:
                live_ids = np.flatnonzero(self._live)
                ages = self.stats.steps - self._born_steps[live_ids]
                consolidating_ids = live_ids[
                    np.logical_and.reduce(
                        (
                            self._lifecycle[live_ids] == JUVENILE,
                            ages >= self.config.consolidation_min_age,
                            self._fires[live_ids]
                            >= self.config.consolidation_min_fires,
                            self._utility_bits[live_ids]
                            >= self.config.consolidation_utility_threshold,
                        )
                    )
                ]
                if len(consolidating_ids):
                    self._lifecycle[consolidating_ids] = CONSOLIDATED
                    self._consolidated_steps[consolidating_ids] = self.stats.steps
                    self.stats.consolidations += len(consolidating_ids)

                dormancy_ids = live_ids[
                    np.logical_and(
                        self._lifecycle[live_ids] == CONSOLIDATED,
                        np.logical_or(
                            self.stats.steps - self._last_fire_steps[live_ids]
                            >= self.config.dormancy_staleness,
                            self._utility_bits[live_ids]
                            < self.config.prune_utility,
                        ),
                    )
                ]
                if len(dormancy_ids):
                    self._lifecycle[dormancy_ids] = DORMANT
                    self._dormant_steps[dormancy_ids] = self.stats.steps
                    self.stats.dormancies += len(dormancy_ids)

            plastic_ids = evaluated_ids[
                np.logical_and(
                    self._lifecycle[evaluated_ids] != DORMANT,
                    current_values[evaluated_ids] > self.config.support_threshold,
                )
            ]
            positions = self._local_sample_positions[plastic_ids]
            self._local_samples[plastic_ids, positions, 0] = current_values[
                plastic_ids
            ]
            self._local_samples[plastic_ids, positions, 1] = trace_values[
                plastic_ids
            ]
            self._local_sample_positions[plastic_ids] = (
                positions + 1
            ) % self.config.local_buffer_size
            self._local_sample_counts[plastic_ids] = np.minimum(
                self.config.local_buffer_size,
                self._local_sample_counts[plastic_ids] + 1,
            )
            self._observations[plastic_ids] += 1

            update_interaction_stats_compiled(
                plastic_ids,
                current_values,
                trace_values,
                raw_responses,
                self._stats_n,
                self._stats_mean_a,
                self._stats_mean_b,
                self._stats_mean_d,
                self._stats_m2_a,
                self._stats_m2_b,
                self._stats_m2_d,
                self._stats_cov_da_acc,
                self._stats_cov_db_acc,
            )

            self._fires[firing_ids] += 1
            self.stats.relation_firings += len(firing_ids)
            adapt_intervals = np.where(
                self._lifecycle[plastic_ids] == CONSOLIDATED,
                self.config.consolidated_adapt_interval,
                self.config.adapt_interval,
            )
            adapting_ids = plastic_ids[
                np.logical_and(
                    self._observations[plastic_ids] % adapt_intervals == 0,
                    self._local_sample_counts[plastic_ids]
                    >= self.config.min_adapt_samples,
                )
            ]
            self._adapt_slots(adapting_ids)

        normalized = self._normalized_traces()
        if learn:
            self.trace_means = (
                self.config.trace_mean_decay * self.trace_means
                + (1.0 - self.config.trace_mean_decay) * normalized
            )
        return (
            features,
            normalized,
            raw_responses,
            state_probes,
            evaluated_ids,
        )

    def _advance_features(
        self, token_id: int, *, learn: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.config.metabolic_backend == "event":
            return self._advance_features_event(token_id, learn=learn)
        return self._advance_features_dense(token_id, learn=learn)

    def _competing_predictions(
        self,
        features: np.ndarray,
        raw_responses: np.ndarray,
        state_probes: np.ndarray,
        evaluated_ids: np.ndarray,
    ) -> tuple[
        np.ndarray,
        tuple[int, ...],
        np.ndarray,
        np.ndarray,
    ]:
        if self.effective_competition_backend == "compiled":
            (
                mixture,
                winner_ids,
                winner_count,
                local_ids,
                local_is_state,
                local_count,
            ) = compete_predictors_compiled(
                evaluated_ids,
                features,
                raw_responses,
                state_probes,
                self._live,
                self._lifecycle,
                self._observations,
                self._fires,
                self._utility_bits,
                self._state_observations,
                self._state_utility_bits,
                self._predictor_counts,
                self._state_predictor_counts,
                self.base_counts,
                self.context_counts,
                self.previous_token,
                self.config.contextual_residual,
                self.config.vocab_size,
                DORMANT,
                self.config.activation_threshold,
                self.config.relation_prior,
                self.config.base_prior,
                self.config.context_prior,
                self.config.column_size,
                self.column_count,
                self.config.max_active_per_column,
                self.config.max_active_columns,
                self.config.max_active_relations,
            )
            active = tuple(int(slot_id) for slot_id in winner_ids[:winner_count])
            return (
                mixture,
                active,
                local_ids[:local_count].copy(),
                local_is_state[:local_count].copy(),
            )
        if self.effective_competition_backend == "batched":
            return self._competing_predictions_batched(
                features, raw_responses, state_probes, evaluated_ids
            )
        return self._competing_predictions_scalar(
            features, raw_responses, state_probes, evaluated_ids
        )

    def _fuse_predictions(
        self,
        base: np.ndarray,
        weighted_predictions: list[tuple[float, np.ndarray]],
    ) -> np.ndarray:
        """Combine basal drive with bounded apical log-probability residuals."""
        if not self.config.contextual_residual:
            mixture = base.copy()
            total_weight = 1.0
            for weight, distribution in weighted_predictions:
                mixture += weight * distribution
                total_weight += weight
            return mixture / total_weight

        # A weighted geometric pool is exactly a bounded residual in log space:
        # log P = log B + sum alpha_i * (log D_i - log B), where the
        # residual weights alpha_i sum to less than one.
        logits = np.log(np.maximum(base, 1e-300))
        total_weight = 1.0
        for weight, distribution in weighted_predictions:
            logits += weight * np.log(np.maximum(distribution, 1e-300))
            total_weight += weight
        logits /= total_weight
        logits -= logits.max()
        mixture = np.exp(logits)
        return mixture / mixture.sum()

    def _competing_predictions_scalar(
        self,
        features: np.ndarray,
        raw_responses: np.ndarray,
        state_probes: np.ndarray,
        evaluated_ids: np.ndarray,
    ) -> tuple[
        np.ndarray,
        tuple[int, ...],
        np.ndarray,
        np.ndarray,
    ]:
        base = self._base_distribution()
        candidates: list[tuple[float, int, np.ndarray]] = []
        local_kinds: dict[int, str] = {}
        for raw_slot_id in evaluated_ids:
            slot_id = int(raw_slot_id)
            slot = self.slots[slot_id]
            if (
                slot is None
                or self._observations[slot_id] < 2
                or self._fires[slot_id] < 1
                or self._lifecycle[slot_id] == DORMANT
            ):
                continue
            raw_active = raw_responses[slot_id] >= self.config.activation_threshold
            state_active = state_probes[slot_id] >= self.config.activation_threshold
            if raw_active:
                counts = self._predictor_counts[slot_id] + self.config.relation_prior
                distribution = counts / counts.sum()
                local_kinds[slot_id] = "event"
            elif state_active:
                counts = (
                    self._state_predictor_counts[slot_id]
                    + self.config.relation_prior
                )
                distribution = counts / counts.sum()
                local_kinds[slot_id] = "state"
            else:
                continue
            # Every firing branch gets the next local outcome, whether or not
            # it wins the soma-level competition.  Local cortical circuits do
            # not need permission from the readout to gather evidence.
            activation = float(features[self.config.vocab_size + slot_id])
            if activation < self.config.activation_threshold:
                continue
            observations = self._observations[slot_id]
            evidence = observations / (observations + 8.0)
            utility = self._utility_bits[slot_id]
            if local_kinds[slot_id] == "state":
                state_observations = self._state_observations[slot_id]
                evidence = state_observations / (state_observations + 16.0)
                utility = self._state_utility_bits[slot_id]
            # Immature branches receive a small exploratory voice.  After
            # several firings, only locally positive predictive utility can
            # win competition; structurally interesting but useless branches
            # no longer dilute the soma-level vote.
            if self._fires[slot_id] < 8:
                utility_factor = 0.10
            else:
                utility_factor = (
                    min(4.0, 0.25 + 4.0 * utility)
                    if utility > 0.0
                    else 0.0
                )
            weight = activation * evidence * utility_factor
            if weight <= 0.0:
                continue
            candidates.append((weight, slot_id, distribution))

        candidates.sort(key=lambda item: item[0], reverse=True)
        winners = candidates[: self.config.max_active_relations]
        mixture = self._fuse_predictions(
            base,
            [(weight, distribution) for weight, _slot_id, distribution in winners],
        )
        active = tuple(slot_id for _weight, slot_id, _dist in winners)
        relation_ids = np.asarray(list(local_kinds), dtype=np.intp)
        relation_is_state = np.asarray(
            [local_kinds[int(slot_id)] == "state" for slot_id in relation_ids],
            dtype=np.bool_,
        )
        return mixture, active, relation_ids, relation_is_state

    def _competing_predictions_batched(
        self,
        features: np.ndarray,
        raw_responses: np.ndarray,
        state_probes: np.ndarray,
        evaluated_ids: np.ndarray,
    ) -> tuple[
        np.ndarray,
        tuple[int, ...],
        np.ndarray,
        np.ndarray,
    ]:
        base = self._base_distribution()
        eligible_ids = evaluated_ids[
            np.logical_and.reduce(
                (
                    self._live[evaluated_ids],
                    self._lifecycle[evaluated_ids] != DORMANT,
                    self._observations[evaluated_ids] >= 2,
                    self._fires[evaluated_ids] >= 1,
                )
            )
        ]
        event_mask = (
            raw_responses[eligible_ids] >= self.config.activation_threshold
        )
        state_mask = np.logical_and(
            ~event_mask,
            state_probes[eligible_ids] >= self.config.activation_threshold,
        )
        local_mask = np.logical_or(event_mask, state_mask)
        ids = eligible_ids[local_mask]
        if len(ids) == 0:
            return (
                base,
                (),
                ids,
                np.empty(0, dtype=np.bool_),
            )

        is_state = state_mask[local_mask]
        activation = features[self.config.vocab_size + ids]
        evidence = np.where(
            is_state,
            self._state_observations[ids]
            / (self._state_observations[ids] + 16.0),
            self._observations[ids] / (self._observations[ids] + 8.0),
        )
        utility = np.where(
            is_state, self._state_utility_bits[ids], self._utility_bits[ids]
        )
        utility_factor = np.where(
            self._fires[ids] < 8,
            0.10,
            np.where(utility > 0.0, np.minimum(4.0, 0.25 + 4.0 * utility), 0.0),
        )
        weights = activation * evidence * utility_factor
        candidate_positions = np.flatnonzero(
            np.logical_and(
                activation >= self.config.activation_threshold, weights > 0.0
            )
        )
        if len(candidate_positions) == 0:
            return base, (), ids, is_state

        if self.column_count == 1:
            order = np.argsort(-weights[candidate_positions], kind="stable")
            winner_positions = candidate_positions[
                order[: self.config.max_active_relations]
            ]
        else:
            winner_buffer, winner_count = select_column_winners_compiled(
                weights,
                ids,
                candidate_positions,
                self.config.column_size,
                self.column_count,
                self.config.max_active_per_column,
                self.config.max_active_columns,
                self.config.max_active_relations,
            )
            winner_positions = winner_buffer[:winner_count]
        winner_ids = ids[winner_positions]
        winner_is_state = is_state[winner_positions]
        winner_counts = np.where(
            winner_is_state[:, None],
            self._state_predictor_counts[winner_ids],
            self._predictor_counts[winner_ids],
        )
        winner_distributions = winner_counts + self.config.relation_prior
        winner_distributions /= winner_distributions.sum(axis=1, keepdims=True)
        mixture = self._fuse_predictions(
            base,
            [
                (float(weights[position]), winner_distributions[winner_index])
                for winner_index, position in enumerate(winner_positions)
            ],
        )
        active = tuple(int(ids[position]) for position in winner_positions)
        return mixture, active, ids, is_state

    def _resolve_pending(
        self, token_id: int, *, learn: bool
    ) -> tuple[float | None, float | None, float | None]:
        if self._pending is None:
            return None, None, None

        observed_p = float(self._pending.distribution[token_id])
        surprise = -math.log2(max(observed_p, 1e-12))
        base_p = float(self._pending.base_distribution[token_id])
        if learn and len(self._pending.relation_ids):
            ids = self._pending.relation_ids
            state_selector = self._pending.relation_is_state
            state_ids = ids[state_selector]
            event_ids = ids[~state_selector]
            if self.effective_outcome_backend == "compiled":
                update_local_outcomes_compiled(
                    ids,
                    state_selector,
                    token_id,
                    base_p,
                    self.config.relation_prior,
                    self.config.utility_decay,
                    self.config.predictor_decay,
                    self.config.consolidated_predictor_decay,
                    CONSOLIDATED,
                    self._lifecycle,
                    self._predictor_counts,
                    self._state_predictor_counts,
                    self._utility_bits,
                    self._state_utility_bits,
                    self._state_observations,
                )
            else:
                slot_probabilities = np.empty(len(ids))
                prior = self.config.relation_prior
                if len(state_ids):
                    state_counts = self._state_predictor_counts[state_ids]
                    slot_probabilities[state_selector] = (
                        state_counts[:, token_id] + prior
                    ) / (
                        state_counts.sum(axis=1)
                        + prior * self.config.vocab_size
                    )
                if len(event_ids):
                    event_counts = self._predictor_counts[event_ids]
                    slot_probabilities[~state_selector] = (
                        event_counts[:, token_id] + prior
                    ) / (
                        event_counts.sum(axis=1)
                        + prior * self.config.vocab_size
                    )
                gains = np.log2(
                    np.maximum(slot_probabilities, 1e-12)
                    / max(base_p, 1e-12)
                )
                if len(state_ids):
                    state_gains = gains[state_selector]
                    self._state_utility_bits[state_ids] = (
                        self.config.utility_decay
                        * self._state_utility_bits[state_ids]
                        + (1.0 - self.config.utility_decay) * state_gains
                    )
                    state_decays = np.where(
                        self._lifecycle[state_ids] == CONSOLIDATED,
                        self.config.consolidated_predictor_decay,
                        self.config.predictor_decay,
                    )
                    self._state_predictor_counts[state_ids] *= state_decays[
                        :, None
                    ]
                    self._state_predictor_counts[state_ids, token_id] += 1.0
                    self._state_observations[state_ids] += 1
                if len(event_ids):
                    event_gains = gains[~state_selector]
                    self._utility_bits[event_ids] = (
                        self.config.utility_decay * self._utility_bits[event_ids]
                        + (1.0 - self.config.utility_decay) * event_gains
                    )
                    event_decays = np.where(
                        self._lifecycle[event_ids] == CONSOLIDATED,
                        self.config.consolidated_predictor_decay,
                        self.config.predictor_decay,
                    )
                    self._predictor_counts[event_ids] *= event_decays[:, None]
                    self._predictor_counts[event_ids, token_id] += 1.0

            # Scalar mirrors exist for checkpoint compatibility and
            # introspection; dense matrices remain the arithmetic source.
            for raw_slot_id in ids:
                slot_id = int(raw_slot_id)
                slot = self.slots[slot_id]
                assert slot is not None
                slot.utility_bits = float(self._utility_bits[slot_id])
                slot.state_utility_bits = float(
                    self._state_utility_bits[slot_id]
                )
                slot.state_observations = int(
                    self._state_observations[slot_id]
                )
            self.stats.predictions_scored += 1
        elif learn:
            self.stats.predictions_scored += 1
        return observed_p, base_p, surprise

    def _relation_depth_for_channel(self, channel: int) -> int:
        if channel < self.config.vocab_size:
            return 0
        slot = self.slots[channel - self.config.vocab_size]
        return 0 if slot is None else slot.depth

    def _pick_birth_parents(
        self, pending: _PendingPrediction, target_slot_id: int
    ) -> tuple[int, int, int, float, float] | None:
        active_sensory = np.flatnonzero(
            pending.features[: self.config.vocab_size] > self.config.support_threshold
        )
        active_relations = np.flatnonzero(
            pending.features[self.config.vocab_size :]
            > self.config.activation_threshold
        )
        active_relations = np.asarray(
            [
                slot_id
                for slot_id in active_relations
                if self.slots[int(slot_id)] is not None
                and self._fires[int(slot_id)] >= 8
                and self._utility_bits[int(slot_id)] > 0.0
            ],
            dtype=np.int64,
        )
        active_relations = active_relations[active_relations < target_slot_id]
        use_recursive = (
            len(active_relations) > 0
            and self.rng.random() < self.config.recursive_birth_probability
        )
        if use_recursive:
            relation_scores = pending.features[
                self.config.vocab_size + active_relations
            ]
            current_channel = self.config.vocab_size + int(
                active_relations[int(np.argmax(relation_scores))]
            )
        elif len(active_sensory) > 0:
            current_channel = int(active_sensory[0])
        else:
            return None

        # Homeostatic novelty: a trace is interesting when it is active now
        # relative to how often that same receptor is normally active.
        eligible_relation_ids = np.flatnonzero(
            np.logical_and.reduce(
                (
                    self._live,
                    np.arange(self.config.max_relations) < target_slot_id,
                    self._fires >= 8,
                    self._utility_bits > 0.0,
                )
            )
        )
        candidate_channels = np.concatenate(
            (
                np.arange(self.config.vocab_size, dtype=np.int64),
                self.config.vocab_size + eligible_relation_ids,
            )
        )
        novelty = pending.traces[:, candidate_channels] / (
            self.trace_means[:, candidate_channels] + 0.05
        )
        timescale_bias = 1.0 / np.sqrt(np.asarray(self.config.timescales))
        novelty *= timescale_bias[:, None]
        novelty[:, candidate_channels == current_channel] = -np.inf

        existing = {
            (slot.current_channel, slot.trace_channel, slot.timescale_index)
            for slot in self.slots
            if slot is not None
        }
        order = np.argsort(novelty.ravel())[::-1]
        for flat_index in order:
            timescale_index, channel_position = np.unravel_index(
                int(flat_index), novelty.shape
            )
            if not np.isfinite(novelty[timescale_index, channel_position]):
                continue
            trace_channel = int(candidate_channels[channel_position])
            b = float(pending.traces[timescale_index, trace_channel])
            if b <= self.config.support_threshold:
                break
            key = (current_channel, int(trace_channel), int(timescale_index))
            if key in existing:
                continue
            depth = 1 + max(
                self._relation_depth_for_channel(current_channel),
                self._relation_depth_for_channel(int(trace_channel)),
            )
            if depth > self.config.max_depth:
                continue
            a = float(pending.features[current_channel])
            return current_channel, int(trace_channel), int(timescale_index), a, b
        return None

    def _weakest_recyclable_slot(
        self, minimum_slot_id: int = 0, *, column_id: int | None = None
    ) -> int | None:
        column_start = (
            0 if column_id is None else column_id * self.config.column_size
        )
        column_stop = (
            self.config.max_relations
            if column_id is None
            else min(
                self.config.max_relations,
                (column_id + 1) * self.config.column_size,
            )
        )
        empty = next(
            (
                i
                for i, slot in enumerate(self.slots)
                if max(minimum_slot_id, column_start) <= i < column_stop
                and slot is None
            ),
            None,
        )
        if empty is not None:
            return empty

        referenced_slots = {
            channel - self.config.vocab_size
            for slot in self.slots
            if slot is not None
            for channel in (slot.current_channel, slot.trace_channel)
            if channel >= self.config.vocab_size
        }
        referenced_slots.update(
            channel - self.config.vocab_size
            for candidate in self.nursery.values()
            for channel in (candidate.current_channel, candidate.trace_channel)
            if channel >= self.config.vocab_size
        )
        recyclable: list[tuple[int, float, int]] = []
        for slot_id, slot in enumerate(self.slots):
            if slot is None:
                continue
            if slot_id < minimum_slot_id:
                continue
            if not column_start <= slot_id < column_stop:
                continue
            if slot_id in referenced_slots:
                continue
            age = self.stats.steps - slot.born_step
            stale = self.stats.steps - self._last_fire_steps[slot_id]
            low_utility = self._utility_bits[slot_id] < self.config.prune_utility
            silent = stale > self.config.max_staleness
            underused = self._fires[slot_id] < 2
            lifecycle = int(self._lifecycle[slot_id])
            if lifecycle == CONSOLIDATED:
                continue
            if (
                lifecycle == DORMANT
                and self.stats.steps - self._dormant_steps[slot_id]
                < self.config.dormant_min_residence
            ):
                continue
            recyclable_now = low_utility or silent or underused
            if age >= self.config.min_prune_age and recyclable_now:
                survival = self._utility_bits[slot_id] + 0.02 * math.log1p(
                    self._fires[slot_id]
                )
                survival += 0.005 * math.log1p(age)
                lifecycle_priority = 0 if lifecycle == DORMANT else 1
                recyclable.append((lifecycle_priority, survival, slot_id))
        if not recyclable:
            return None
        return min(recyclable)[2]

    def _nursery_candidate(
        self,
        parents: tuple[int, int, int, float, float],
        token_id: int,
    ) -> DevelopmentalCandidate:
        current_channel, trace_channel, timescale_index, a, b = parents
        key = (current_channel, trace_channel, timescale_index)
        self.stats.nursery_proposals += 1

        stale_keys = [
            candidate_key
            for candidate_key, candidate in self.nursery.items()
            if self.stats.steps - candidate.last_seen_step
            > self.config.nursery_max_age
        ]
        for candidate_key in stale_keys:
            del self.nursery[candidate_key]
            self.stats.nursery_evictions += 1

        candidate = self.nursery.get(key)
        if candidate is None:
            if len(self.nursery) >= self.config.max_nursery_candidates:
                evict_key = min(
                    self.nursery,
                    key=lambda item: (
                        self.nursery[item].confirmations,
                        self.nursery[item].last_seen_step,
                    ),
                )
                del self.nursery[evict_key]
                self.stats.nursery_evictions += 1
            target_counts = np.zeros(self.config.vocab_size)
            target_counts[token_id] = 1.0
            depth = 1 + max(
                self._relation_depth_for_channel(current_channel),
                self._relation_depth_for_channel(trace_channel),
            )
            candidate = DevelopmentalCandidate(
                current_channel=current_channel,
                trace_channel=trace_channel,
                timescale_index=timescale_index,
                depth=depth,
                column_id=self._column_for_parents(
                    current_channel, trace_channel, timescale_index
                )[0],
                mean_a=a,
                mean_b=b,
                target_counts=target_counts,
                confirmations=1,
                first_seen_step=self.stats.steps,
                last_seen_step=self.stats.steps,
            )
            self.nursery[key] = candidate
            return candidate

        candidate.confirmations += 1
        rate = 1.0 / candidate.confirmations
        candidate.mean_a += rate * (a - candidate.mean_a)
        candidate.mean_b += rate * (b - candidate.mean_b)
        candidate.target_counts[token_id] += 1.0
        candidate.last_seen_step = self.stats.steps
        return candidate

    def _candidate_is_ready(self, candidate: DevelopmentalCandidate) -> bool:
        if candidate.confirmations < self.config.nursery_confirmations:
            return False
        consistency = float(
            candidate.target_counts.max() / candidate.target_counts.sum()
        )
        return consistency >= self.config.nursery_target_consistency

    def _birth_relation(self, token_id: int) -> tuple[int | None, int | None]:
        if self._pending is None:
            return None, None
        parents = self._pick_birth_parents(
            self._pending, self.config.max_relations
        )
        if parents is None:
            self.stats.skipped_births += 1
            return None, None
        current_channel, trace_channel, timescale_index, _a, _b = parents
        _column_id, crosses_columns = self._column_for_parents(
            current_channel, trace_channel, timescale_index
        )
        if (
            crosses_columns
            and self.rng.random() >= self.config.inter_column_birth_probability
        ):
            self.stats.skipped_births += 1
            return None, None
        candidate = self._nursery_candidate(parents, token_id)
        if not self._candidate_is_ready(candidate):
            return None, None

        relation_parents = [
            channel - self.config.vocab_size
            for channel in (candidate.current_channel, candidate.trace_channel)
            if channel >= self.config.vocab_size
        ]
        minimum_slot_id = max(relation_parents, default=-1) + 1
        slot_id = self._weakest_recyclable_slot(
            minimum_slot_id, column_id=candidate.column_id
        )
        if slot_id is None:
            self.stats.skipped_births += 1
            return None, None

        pruned = slot_id if self.slots[slot_id] is not None else None
        if pruned is not None:
            self.stats.prunes += 1

        current_channel = candidate.current_channel
        trace_channel = candidate.trace_channel
        timescale_index = candidate.timescale_index
        a = candidate.mean_a
        b = candidate.mean_b
        dendrite_seed = self.config.seed + 104729 * (self.stats.births + 1)
        dendrite = InteractionDendrite(dendrite_seed)
        dendrite.w_a = 1.0
        dendrite.w_b = 1.0
        dendrite.mu = a + b
        dendrite.sigma = self.config.initial_sigma
        predictor_counts = candidate.target_counts.copy()
        depth = candidate.depth
        self.slots[slot_id] = OnlineRelationSlot(
            slot_id=slot_id,
            current_channel=current_channel,
            trace_channel=trace_channel,
            timescale_index=timescale_index,
            depth=depth,
            column_id=candidate.column_id,
            dendrite=dendrite,
            predictor_counts=predictor_counts,
            state_predictor_counts=np.zeros(self.config.vocab_size),
            local_samples=deque(
                [(a, b)], maxlen=self.config.local_buffer_size
            ),
            born_step=self.stats.steps,
            observations=1,
            fires=1,
            last_fire_step=self.stats.steps - 1,
        )
        self._sync_slot_parameters(slot_id)
        self._sync_slot_evidence(slot_id)
        self._sync_slot_runtime(slot_id)
        self._invalidate_response_batches()
        relation_channel = self.config.vocab_size + slot_id
        self.traces[:, relation_channel] = 0.0
        self.trace_means[:, relation_channel] = 0.05
        del self.nursery[(current_channel, trace_channel, timescale_index)]
        self.stats.births += 1
        self.stats.max_live_relations = max(
            self.stats.max_live_relations, self.live_relation_count
        )
        self._last_birth_step = self.stats.steps
        return slot_id, pruned

    def observe(
        self, token_id: int, *, reset: bool = False, learn: bool = True
    ) -> StreamStep:
        """Observe one symbol and return the prediction for its successor.

        If ``reset`` is true, the transition into this symbol is not scored or
        learned, and all activity state is cleared before processing it.
        Learned dendrites and local predictor counts survive the boundary.
        """
        if not 0 <= token_id < self.config.vocab_size:
            raise ValueError("token_id outside configured vocabulary")
        if reset:
            self._reset_dynamic_state()

        observed_p, base_observed_p, surprise = self._resolve_pending(
            token_id, learn=learn
        )
        born = None
        pruned = None
        if (
            learn
            and surprise is not None
            and surprise >= self.config.surprise_birth_bits
            and self.stats.steps - self._last_birth_step >= self.config.birth_cooldown
        ):
            born, pruned = self._birth_relation(token_id)

        if learn:
            self.base_counts *= self.config.base_decay
            self.base_counts[token_id] += 1.0
            if self.config.contextual_residual and self.previous_token >= 0:
                self.context_counts[self.previous_token] *= self.config.context_decay
                self.context_counts[self.previous_token, token_id] += 1.0
        self.previous_token = token_id
        (
            features,
            traces,
            raw_responses,
            state_probes,
            evaluated_ids,
        ) = self._advance_features(token_id, learn=learn)
        (
            prediction,
            active,
            relation_ids,
            relation_is_state,
        ) = self._competing_predictions(
            features, raw_responses, state_probes, evaluated_ids
        )
        self._pending = _PendingPrediction(
            distribution=prediction.copy(),
            base_distribution=self._base_distribution().copy(),
            relation_ids=relation_ids,
            relation_is_state=relation_is_state,
            active_relations=active,
            features=features.copy(),
            traces=traces.copy(),
        )
        if learn:
            self.stats.steps += 1
            if (
                self.config.max_alm_bytes is not None
                and self.live_relation_count
                >= math.ceil(0.9 * self.config.max_relations)
                and self.stats.steps % self.config.byte_budget_check_interval == 0
            ):
                self._payload_within_byte_budget()
        return StreamStep(
            prediction=prediction,
            observed_probability=observed_p,
            base_observed_probability=base_observed_p,
            surprise_bits=surprise,
            active_relations=active,
            born_relation=born,
            pruned_relation=pruned,
        )

    @property
    def supports_compiled_chunks(self) -> bool:
        """Whether ordinary bytes can run wholly inside the compiled loop."""
        return bool(
            NUMBA_AVAILABLE
            and self.config.metabolic_backend == "dense"
            and self.effective_response_backend == "compiled"
            and self.effective_outcome_backend == "compiled"
            and self.effective_adaptation_backend == "compiled"
            and self.config.adaptation_rng == "counter"
        )

    def observe_chunk_probabilities(
        self, tokens: np.ndarray, *, learn: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score a bounded byte chunk with compiled causal structural barriers.

        Ordinary bytes remain inside one sequential numeric loop. A surprising
        byte that is eligible to recruit anatomy returns to :meth:`observe`
        before it is processed, so nursery growth and pruning retain their
        exact Python semantics and affect every subsequent byte.
        """
        token_array = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if len(token_array) == 0:
            return np.empty(0), np.empty(0)
        if np.any(token_array < 0) or np.any(token_array >= self.config.vocab_size):
            raise ValueError("token_id outside configured vocabulary")
        if not self.supports_compiled_chunks:
            observed = np.full(len(token_array), np.nan)
            base = np.full(len(token_array), np.nan)
            for index, token_id in enumerate(token_array):
                step = self.observe(int(token_id), learn=learn)
                if step.observed_probability is not None:
                    observed[index] = step.observed_probability
                    assert step.base_observed_probability is not None
                    base[index] = step.base_observed_probability
            return observed, base

        observed_parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        offset = 0
        while offset < len(token_array):
            stop = len(token_array)
            budget_check_due = False
            if (
                learn
                and self.config.max_alm_bytes is not None
                and self.live_relation_count
                >= math.ceil(0.9 * self.config.max_relations)
            ):
                interval = self.config.byte_budget_check_interval
                until_check = interval - self.stats.steps % interval
                stop = min(stop, offset + until_check)
                budget_check_due = stop == offset + until_check

            if self._pending is None:
                has_pending = False
                pending_distribution = np.empty(self.config.vocab_size)
                pending_base_distribution = np.empty(self.config.vocab_size)
                pending_ids = np.empty(0, dtype=np.int64)
                pending_is_state = np.empty(0, dtype=np.bool_)
                pending_features = np.zeros(self.n_channels)
                pending_traces = np.zeros_like(self.traces)
            else:
                has_pending = True
                pending_distribution = self._pending.distribution
                pending_base_distribution = self._pending.base_distribution
                pending_ids = self._pending.relation_ids
                pending_is_state = self._pending.relation_is_state
                pending_features = self._pending.features
                pending_traces = self._pending.traces

            live_ids = np.flatnonzero(self._live)
            result = advance_dense_chunk_compiled(
                token_array[offset:stop],
                has_pending,
                pending_distribution,
                pending_base_distribution,
                pending_ids,
                pending_is_state,
                pending_features,
                pending_traces,
                live_ids,
                self._live,
                self._current_channels,
                self._trace_channels,
                self._timescale_indices,
                self._born_steps,
                self._last_fire_steps,
                self._lifecycle,
                self._consolidated_steps,
                self._dormant_steps,
                self._observations,
                self._fires,
                self._utility_bits,
                self._state_observations,
                self._state_utility_bits,
                self._predictor_counts,
                self._state_predictor_counts,
                self.base_counts,
                self.context_counts,
                self.previous_token,
                self.config.contextual_residual,
                self.lambdas,
                self.trace_scales,
                self.traces,
                self.trace_means,
                self._persistence,
                self._w_a,
                self._w_b,
                self._mu,
                self._sigma,
                self._local_samples,
                self._local_sample_counts,
                self._local_sample_positions,
                self._stats_n,
                self._stats_mean_a,
                self._stats_mean_b,
                self._stats_mean_d,
                self._stats_m2_a,
                self._stats_m2_b,
                self._stats_m2_d,
                self._stats_cov_da_acc,
                self._stats_cov_db_acc,
                self.stats.steps,
                self._last_birth_step,
                self.config.seed,
                self.config.vocab_size,
                self.config.max_relations,
                JUVENILE,
                CONSOLIDATED,
                DORMANT,
                self.config.support_threshold,
                self.config.activation_threshold,
                self.config.persistence_decay,
                self.config.persistence_min_observations,
                self.config.persistence_utility_threshold,
                self.config.trace_mean_decay,
                self.config.base_decay,
                self.config.context_decay,
                self.config.base_prior,
                self.config.context_prior,
                self.config.relation_prior,
                self.config.utility_decay,
                self.config.predictor_decay,
                self.config.consolidated_predictor_decay,
                self.config.consolidation_min_age,
                self.config.consolidation_min_fires,
                self.config.consolidation_utility_threshold,
                self.config.dormancy_staleness,
                self.config.prune_utility,
                self.config.adapt_interval,
                self.config.consolidated_adapt_interval,
                self.config.min_adapt_samples,
                self.config.perturb_step,
                self.config.initial_sigma / 4.0,
                self.config.surprise_birth_bits,
                self.config.birth_cooldown,
                self.config.column_size,
                self.column_count,
                self.config.max_active_per_column,
                self.config.max_active_columns,
                self.config.max_active_relations,
                learn,
            )
            (
                processed,
                barrier,
                observed,
                base_observed,
                has_pending,
                pending_distribution,
                pending_base_distribution,
                pending_ids,
                pending_is_state,
                winner_ids,
                pending_features,
                pending_traces,
                previous_token,
                steps,
                scored_delta,
                branch_evaluations_delta,
                firings_delta,
                consolidations_delta,
                dormancies_delta,
                reactivations_delta,
            ) = result
            self.stats.steps = int(steps)
            self.previous_token = int(previous_token)
            self.stats.predictions_scored += int(scored_delta)
            self.stats.branch_evaluations += int(branch_evaluations_delta)
            self.stats.relation_firings += int(firings_delta)
            self.stats.consolidations += int(consolidations_delta)
            self.stats.dormancies += int(dormancies_delta)
            self.stats.reactivations += int(reactivations_delta)
            if has_pending:
                self._pending = _PendingPrediction(
                    distribution=pending_distribution.copy(),
                    base_distribution=pending_base_distribution.copy(),
                    relation_ids=pending_ids.copy(),
                    relation_is_state=pending_is_state.copy(),
                    active_relations=tuple(int(value) for value in winner_ids),
                    features=pending_features.copy(),
                    traces=pending_traces.copy(),
                )
            observed_parts.append(observed)
            base_parts.append(base_observed)
            offset += int(processed)
            if budget_check_due and offset == stop and learn:
                self._payload_within_byte_budget()
            if barrier:
                step = self.observe(int(token_array[offset]), learn=learn)
                observed_parts.append(
                    np.asarray(
                        [
                            np.nan
                            if step.observed_probability is None
                            else step.observed_probability
                        ]
                    )
                )
                base_parts.append(
                    np.asarray(
                        [
                            np.nan
                            if step.base_observed_probability is None
                            else step.base_observed_probability
                        ]
                    )
                )
                offset += 1
            elif processed == 0:
                raise RuntimeError("compiled byte loop made no progress")
        return np.concatenate(observed_parts), np.concatenate(base_parts)

    def state_dict(self) -> dict[str, Any]:
        """Return all anatomy, activity, local evidence, and RNG state."""
        self._materialize_slot_views()
        slot_states = []
        for slot in self.slots:
            if slot is None:
                slot_states.append(None)
                continue
            slot_states.append(
                {
                    "slot_id": slot.slot_id,
                    "current_channel": slot.current_channel,
                    "trace_channel": slot.trace_channel,
                    "timescale_index": slot.timescale_index,
                    "depth": slot.depth,
                    "column_id": slot.column_id,
                    "dendrite": {
                        "w_a": slot.dendrite.w_a,
                        "w_b": slot.dendrite.w_b,
                        "mu": slot.dendrite.mu,
                        "sigma": slot.dendrite.sigma,
                        "stats": {
                            "n": int(self._stats_n[slot.slot_id]),
                            "mean_a": float(self._stats_mean_a[slot.slot_id]),
                            "mean_b": float(self._stats_mean_b[slot.slot_id]),
                            "mean_d": float(self._stats_mean_d[slot.slot_id]),
                            "m2_a": float(self._stats_m2_a[slot.slot_id]),
                            "m2_b": float(self._stats_m2_b[slot.slot_id]),
                            "m2_d": float(self._stats_m2_d[slot.slot_id]),
                            "cov_da_acc": float(
                                self._stats_cov_da_acc[slot.slot_id]
                            ),
                            "cov_db_acc": float(
                                self._stats_cov_db_acc[slot.slot_id]
                            ),
                        },
                    },
                    "predictor_counts": slot.predictor_counts.copy(),
                    "state_predictor_counts": slot.state_predictor_counts.copy(),
                    "local_samples": [
                        tuple(row) for row in self._ordered_local_samples(slot.slot_id)
                    ],
                    "born_step": slot.born_step,
                    "observations": int(self._observations[slot.slot_id]),
                    "fires": int(self._fires[slot.slot_id]),
                    "last_fire_step": int(self._last_fire_steps[slot.slot_id]),
                    "utility_bits": float(self._utility_bits[slot.slot_id]),
                    "state_observations": int(
                        self._state_observations[slot.slot_id]
                    ),
                    "state_utility_bits": float(
                        self._state_utility_bits[slot.slot_id]
                    ),
                    "persistence": float(self._persistence[slot.slot_id]),
                    "lifecycle": LIFECYCLE_NAMES[
                        int(self._lifecycle[slot.slot_id])
                    ],
                    "consolidated_step": int(
                        self._consolidated_steps[slot.slot_id]
                    ),
                    "dormant_step": int(self._dormant_steps[slot.slot_id]),
                }
            )
        pending = None
        if self._pending is not None:
            pending = {
                "distribution": self._pending.distribution.copy(),
                "base_distribution": self._pending.base_distribution.copy(),
                "relation_ids": self._pending.relation_ids.copy(),
                "relation_is_state": self._pending.relation_is_state.copy(),
                "active_relations": self._pending.active_relations,
                "features": self._pending.features.copy(),
                "traces": self._pending.traces.copy(),
            }
        return {
            "format": self.FORMAT,
            "config": asdict(self.config),
            "traces": self.traces.copy(),
            "trace_means": self.trace_means.copy(),
            "base_counts": self.base_counts.copy(),
            "context_counts": self.context_counts.copy(),
            "previous_token": self.previous_token,
            "slots": slot_states,
            "nursery": [
                {
                    "current_channel": candidate.current_channel,
                    "trace_channel": candidate.trace_channel,
                    "timescale_index": candidate.timescale_index,
                    "depth": candidate.depth,
                    "column_id": candidate.column_id,
                    "mean_a": candidate.mean_a,
                    "mean_b": candidate.mean_b,
                    "target_counts": candidate.target_counts.copy(),
                    "confirmations": candidate.confirmations,
                    "first_seen_step": candidate.first_seen_step,
                    "last_seen_step": candidate.last_seen_step,
                }
                for candidate in self.nursery.values()
            ],
            "stats": asdict(self.stats),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "pending": pending,
            "last_birth_step": self._last_birth_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != self.FORMAT:
            raise ValueError("unsupported online dendritic checkpoint format")
        saved_config = dict(state["config"])
        saved_config["timescales"] = tuple(saved_config["timescales"])
        saved_config.setdefault("metabolic_backend", "dense")
        saved_config.setdefault("outcome_backend", "scalar")
        saved_config.setdefault("adaptation_rng", "global")
        saved_config.setdefault("contextual_residual", False)
        saved_config.setdefault("context_decay", 1.0)
        saved_config.setdefault("context_prior", 0.1)
        if OnlineDendriticConfig(**saved_config) != self.config:
            raise ValueError("checkpoint config does not match learner config")
        self.traces = np.asarray(state["traces"], dtype=np.float64).copy()
        self.trace_means = np.asarray(state["trace_means"], dtype=np.float64).copy()
        self.base_counts = np.asarray(state["base_counts"], dtype=np.float64).copy()
        self.context_counts = np.asarray(
            state.get(
                "context_counts",
                self.context_counts,
            ),
            dtype=np.float64,
        ).copy()
        self.previous_token = int(state.get("previous_token", -1))
        self.slots = []
        for slot_state in state["slots"]:
            if slot_state is None:
                self.slots.append(None)
                continue
            dendrite_state = slot_state["dendrite"]
            dendrite = InteractionDendrite.__new__(InteractionDendrite)
            dendrite.w_a = float(dendrite_state["w_a"])
            dendrite.w_b = float(dendrite_state["w_b"])
            dendrite.mu = float(dendrite_state["mu"])
            dendrite.sigma = float(dendrite_state["sigma"])
            dendrite.stats = RunningJointStats(**dendrite_state["stats"])
            self.slots.append(
                OnlineRelationSlot(
                    slot_id=int(slot_state["slot_id"]),
                    current_channel=int(slot_state["current_channel"]),
                    trace_channel=int(slot_state["trace_channel"]),
                    timescale_index=int(slot_state["timescale_index"]),
                    depth=int(slot_state["depth"]),
                    column_id=int(
                        slot_state.get(
                            "column_id",
                            int(slot_state["slot_id"])
                            // self.config.column_size,
                        )
                    ),
                    dendrite=dendrite,
                    predictor_counts=np.asarray(
                        slot_state["predictor_counts"], dtype=np.float64
                    ).copy(),
                    state_predictor_counts=np.asarray(
                        slot_state["state_predictor_counts"], dtype=np.float64
                    ).copy(),
                    local_samples=deque(
                        slot_state["local_samples"],
                        maxlen=self.config.local_buffer_size,
                    ),
                    born_step=int(slot_state["born_step"]),
                    observations=int(slot_state["observations"]),
                    fires=int(slot_state["fires"]),
                    last_fire_step=int(slot_state["last_fire_step"]),
                    utility_bits=float(slot_state["utility_bits"]),
                    state_observations=int(slot_state["state_observations"]),
                    state_utility_bits=float(slot_state["state_utility_bits"]),
                    persistence=float(slot_state["persistence"]),
                    lifecycle=str(slot_state.get("lifecycle", "juvenile")),
                    consolidated_step=int(
                        slot_state.get("consolidated_step", -1)
                    ),
                    dormant_step=int(slot_state.get("dormant_step", -1)),
                )
            )
        self.nursery = {}
        for candidate_state in state["nursery"]:
            candidate = DevelopmentalCandidate(
                current_channel=int(candidate_state["current_channel"]),
                trace_channel=int(candidate_state["trace_channel"]),
                timescale_index=int(candidate_state["timescale_index"]),
                depth=int(candidate_state["depth"]),
                column_id=int(
                    candidate_state.get(
                        "column_id",
                        self._column_for_parents(
                            int(candidate_state["current_channel"]),
                            int(candidate_state["trace_channel"]),
                            int(candidate_state["timescale_index"]),
                        )[0],
                    )
                ),
                mean_a=float(candidate_state["mean_a"]),
                mean_b=float(candidate_state["mean_b"]),
                target_counts=np.asarray(
                    candidate_state["target_counts"], dtype=np.float64
                ).copy(),
                confirmations=int(candidate_state["confirmations"]),
                first_seen_step=int(candidate_state["first_seen_step"]),
                last_seen_step=int(candidate_state["last_seen_step"]),
            )
            key = (
                candidate.current_channel,
                candidate.trace_channel,
                candidate.timescale_index,
            )
            self.nursery[key] = candidate
        self.stats = OnlinePopulationStats(**state["stats"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        pending_state = state["pending"]
        self._pending = None
        if pending_state is not None:
            if "relation_ids" in pending_state:
                relation_ids = np.asarray(
                    pending_state["relation_ids"], dtype=np.intp
                ).copy()
                relation_is_state = np.asarray(
                    pending_state["relation_is_state"], dtype=np.bool_
                ).copy()
            else:
                legacy_distributions = pending_state["relation_distributions"]
                relation_ids = np.asarray(
                    list(legacy_distributions), dtype=np.intp
                )
                legacy_kinds = pending_state["relation_kinds"]
                relation_is_state = np.asarray(
                    [legacy_kinds[int(slot_id)] == "state" for slot_id in relation_ids],
                    dtype=np.bool_,
                )
            self._pending = _PendingPrediction(
                distribution=np.asarray(
                    pending_state["distribution"], dtype=np.float64
                ).copy(),
                base_distribution=np.asarray(
                    pending_state["base_distribution"], dtype=np.float64
                ).copy(),
                relation_ids=relation_ids,
                relation_is_state=relation_is_state,
                active_relations=tuple(pending_state["active_relations"]),
                features=np.asarray(pending_state["features"], dtype=np.float64).copy(),
                traces=np.asarray(pending_state["traces"], dtype=np.float64).copy(),
            )
        self._last_birth_step = int(state["last_birth_step"])
        self._invalidate_response_batches()
        for slot_id in range(self.config.max_relations):
            self._sync_slot_parameters(slot_id)
            self._sync_slot_evidence(slot_id)
            self._sync_slot_runtime(slot_id)
        self._persistent_active = set(
            int(slot_id)
            for slot_id in np.flatnonzero(
                np.logical_and(
                    self._live,
                    self._persistence > self.config.support_threshold,
                )
            )
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically save a trusted local checkpoint."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        payload = self._payload_within_byte_budget()
        with temporary.open("wb") as handle:
            handle.write(payload)
        temporary.replace(destination)

    def _payload_within_byte_budget(self) -> bytes:
        """Serialize exactly, shedding weakest provisional tissue if needed."""
        payload = pickle.dumps(self.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
        limit = self.config.max_alm_bytes
        while limit is not None and len(payload) > limit and self.nursery:
            excess = len(payload) - limit
            approximate_candidate_bytes = 512 + 8 * self.config.vocab_size
            eviction_count = max(
                1, math.ceil(excess / approximate_candidate_bytes)
            )
            weakest = sorted(
                self.nursery,
                key=lambda key: (
                    self.nursery[key].confirmations,
                    self.nursery[key].last_seen_step,
                ),
            )[:eviction_count]
            for key in weakest:
                del self.nursery[key]
            self.stats.nursery_evictions += len(weakest)
            self.stats.budget_evictions += len(weakest)
            payload = pickle.dumps(
                self.state_dict(), protocol=pickle.HIGHEST_PROTOCOL
            )
        if limit is not None and len(payload) > limit:
            raise RuntimeError(
                f"serialized ALM is {len(payload):,} bytes, exceeding its "
                f"{limit:,}-byte class after provisional homeostasis"
            )
        return payload

    def serialized_size_bytes(self) -> int:
        """Return the exact durable body size under the checkpoint codec."""
        return len(self._payload_within_byte_budget())

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> OnlineDendriticStreamLearner:
        """Load a checkpoint created locally by :meth:`save_checkpoint`."""
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)  # noqa: S301 - explicitly trusted local file
        config_state = dict(state["config"])
        config_state["timescales"] = tuple(config_state["timescales"])
        # Checkpoints predating event routing retain their exact dense
        # trajectory; newly constructed organisms default to sparse metabolism.
        config_state.setdefault("metabolic_backend", "dense")
        config_state.setdefault("outcome_backend", "scalar")
        config_state.setdefault("adaptation_rng", "global")
        config_state.setdefault("contextual_residual", False)
        config_state.setdefault("context_decay", 1.0)
        config_state.setdefault("context_prior", 0.1)
        learner = cls(OnlineDendriticConfig(**config_state))
        learner.load_state_dict(state)
        return learner
