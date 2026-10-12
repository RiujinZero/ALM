"""Resident many-core CPU execution for cortical-column ALMs."""

from __future__ import annotations

import math

import numpy as np

from manifoldflow.dendritic_basin.axonal_index import AxonalConsumerIndex
from manifoldflow.dendritic_basin.column_partition import partition_learner_columns
from manifoldflow.dendritic_basin.compiled_kernels import (
    NUMBA_AVAILABLE,
    _ring_interaction_objective_compiled,
    _splitmix64,
)
from manifoldflow.dendritic_basin.online_stream import (
    CONSOLIDATED,
    DORMANT,
    JUVENILE,
    LIFECYCLE_NAMES,
    OnlineDendriticStreamLearner,
    _PendingPrediction,
)

try:
    from numba import njit, prange, set_num_threads
except ImportError:  # pragma: no cover - optional accelerated backend
    set_num_threads = None

    def njit(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        def decorate(function):  # type: ignore[no-untyped-def]
            return function

        return decorate

    prange = range


@njit(cache=True)
def _silent_mean_advance(
    mean: float,
    trace: float,
    decay: float,
    scale: float,
    mean_decay: float,
    silent_steps: int,
) -> float:
    if silent_steps <= 0:
        return mean
    # Normalized traces are clipped at one.  Repeated afferent events can leave
    # a trace above that ceiling, so consume the (usually very short) saturated
    # prefix before applying the geometric closed form to the remaining tail.
    while silent_steps > 0 and scale * trace * decay >= 1.0:
        mean = mean_decay * mean + (1.0 - mean_decay)
        trace *= decay
        silent_steps -= 1
    if silent_steps <= 0:
        return mean
    mean_power = mean_decay**silent_steps
    trace_power = decay**silent_steps
    if abs(mean_decay - decay) < 1e-12:
        weighted = silent_steps * trace_power
    else:
        weighted = decay * (mean_power - trace_power) / (mean_decay - decay)
    return mean_power * mean + (1.0 - mean_decay) * scale * trace * weighted


@njit(cache=True, parallel=True)
def _materialize_lazy_traces(
    traces: np.ndarray,
    trace_means: np.ndarray,
    trace_last_steps: np.ndarray,
    lambdas: np.ndarray,
    trace_scales: np.ndarray,
    final_step: int,
    trace_mean_decay: float,
    learn: bool,
) -> None:
    channel_count = trace_last_steps.shape[0]
    for index in prange(traces.shape[0]):
        scale_index = index // channel_count
        channel = index - scale_index * channel_count
        gap = final_step - trace_last_steps[channel]
        if gap <= 0:
            continue
        old_trace = traces[index]
        if learn:
            trace_means[index] = _silent_mean_advance(
                trace_means[index],
                old_trace,
                lambdas[scale_index],
                trace_scales[scale_index],
                trace_mean_decay,
                gap,
            )
        traces[index] = old_trace * lambdas[scale_index] ** gap
    for channel in range(channel_count):
        trace_last_steps[channel] = final_step


@njit(cache=True)
def _adapt_slot_counter(
    slot: int,
    observation_count: int,
    seed: int,
    perturb_step: float,
    local_samples: np.ndarray,
    local_sample_counts: np.ndarray,
    local_sample_positions: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    minimum_sigma: float,
) -> None:
    proposal = np.empty(4, dtype=np.float64)
    base = (
        np.uint64(seed)
        ^ (np.uint64(slot + 1) * np.uint64(0xD2B74407B1CE6E93))
        ^ (np.uint64(observation_count + 1) * np.uint64(0xCA5A826395121157))
    )
    inverse_53 = 1.0 / 9007199254740992.0
    for pair in range(2):
        first = _splitmix64(base ^ np.uint64(2 * pair + 1))
        second = _splitmix64(base ^ np.uint64(2 * pair + 2))
        u1 = ((first >> np.uint64(11)) + 0.5) * inverse_53
        u2 = ((second >> np.uint64(11)) + 0.5) * inverse_53
        radius = math.sqrt(-2.0 * math.log(u1)) * perturb_step
        angle = 2.0 * math.pi * u2
        proposal[2 * pair] = radius * math.cos(angle)
        proposal[2 * pair + 1] = radius * math.sin(angle)

    candidate_w_a = w_a[slot] + proposal[0]
    candidate_w_b = w_b[slot] + proposal[1]
    candidate_mu = mu[slot] + proposal[2]
    candidate_sigma = max(1e-3, sigma[slot] + proposal[3])
    count = int(local_sample_counts[slot])
    position = int(local_sample_positions[slot])
    current_j = _ring_interaction_objective_compiled(
        local_samples[slot],
        count,
        position,
        w_a[slot],
        w_b[slot],
        mu[slot],
        sigma[slot],
    )
    candidate_j = _ring_interaction_objective_compiled(
        local_samples[slot],
        count,
        position,
        candidate_w_a,
        candidate_w_b,
        candidate_mu,
        candidate_sigma,
    )
    if candidate_j > current_j:
        w_a[slot] = candidate_w_a
        w_b[slot] = candidate_w_b
        mu[slot] = candidate_mu
        sigma[slot] = candidate_sigma
    sigma[slot] = max(minimum_sigma, sigma[slot])


@njit(cache=True, parallel=True)
def _lifecycle_sweep_parallel(
    worker_starts: np.ndarray,
    worker_stops: np.ndarray,
    organism_step: int,
    live: np.ndarray,
    lifecycle: np.ndarray,
    born_steps: np.ndarray,
    last_fire_steps: np.ndarray,
    fires: np.ndarray,
    utility_bits: np.ndarray,
    consolidated_steps: np.ndarray,
    dormant_steps: np.ndarray,
    consolidation_min_age: int,
    consolidation_min_fires: int,
    consolidation_utility_threshold: float,
    dormancy_staleness: int,
    prune_utility: float,
) -> None:
    for worker in prange(worker_starts.shape[0]):
        for slot in range(worker_starts[worker], worker_stops[worker]):
            if not live[slot]:
                continue
            if (
                lifecycle[slot] == JUVENILE
                and organism_step - born_steps[slot] >= consolidation_min_age
                and fires[slot] >= consolidation_min_fires
                and utility_bits[slot] >= consolidation_utility_threshold
            ):
                lifecycle[slot] = CONSOLIDATED
                consolidated_steps[slot] = organism_step
            if lifecycle[slot] == CONSOLIDATED and (
                organism_step - last_fire_steps[slot] >= dormancy_staleness
                or utility_bits[slot] < prune_utility
            ):
                lifecycle[slot] = DORMANT
                dormant_steps[slot] = organism_step


@njit(cache=True, parallel=True)
def _evaluate_depth_parallel(
    worker_starts: np.ndarray,
    worker_stops: np.ndarray,
    depth: int,
    step: int,
    organism_step: int,
    generation: int,
    vocab_size: int,
    current_channels: np.ndarray,
    trace_channels: np.ndarray,
    timescale_indices: np.ndarray,
    depths: np.ndarray,
    live: np.ndarray,
    lifecycle: np.ndarray,
    observations: np.ndarray,
    fires: np.ndarray,
    last_fire_steps: np.ndarray,
    dormant_steps: np.ndarray,
    utility_bits: np.ndarray,
    state_observations: np.ndarray,
    state_utility_bits: np.ndarray,
    consumer_offsets: np.ndarray,
    consumer_ids: np.ndarray,
    lambdas: np.ndarray,
    trace_scales: np.ndarray,
    traces: np.ndarray,
    trace_means: np.ndarray,
    trace_last_steps: np.ndarray,
    persistence: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    features: np.ndarray,
    local_ids: np.ndarray,
    local_states: np.ndarray,
    local_counts: np.ndarray,
    column_winner_ids: np.ndarray,
    column_winner_weights: np.ndarray,
    column_winner_states: np.ndarray,
    column_size: int,
    max_per_column: int,
    scheduled_stamps: np.ndarray,
    local_samples: np.ndarray,
    local_sample_counts: np.ndarray,
    local_sample_positions: np.ndarray,
    stats_n: np.ndarray,
    stats_mean_a: np.ndarray,
    stats_mean_b: np.ndarray,
    stats_mean_d: np.ndarray,
    stats_m2_a: np.ndarray,
    stats_m2_b: np.ndarray,
    stats_m2_d: np.ndarray,
    stats_cov_da_acc: np.ndarray,
    stats_cov_db_acc: np.ndarray,
    support_threshold: float,
    activation_threshold: float,
    persistence_decay: float,
    persistence_min_observations: int,
    persistence_utility_threshold: float,
    consolidation_utility_threshold: float,
    utility_gated_persistence: bool,
    trace_mean_decay: float,
    adapt_interval: int,
    consolidated_adapt_interval: int,
    min_adapt_samples: int,
    seed: int,
    perturb_step: float,
    minimum_sigma: float,
    learn: bool,
) -> None:
    channel_count = vocab_size + live.shape[0]
    for worker in prange(worker_starts.shape[0]):
        start = worker_starts[worker]
        stop = worker_stops[worker]
        for slot in range(start, stop):
            if not live[slot] or depths[slot] != depth:
                continue
            useful_state = (
                not utility_gated_persistence
                or state_observations[slot] < persistence_min_observations
                or state_utility_bits[slot] >= persistence_utility_threshold
            )
            retained = persistence[slot]
            if (
                scheduled_stamps[slot] != generation
                and not (retained > support_threshold and useful_state)
            ):
                continue
            current = int(current_channels[slot])
            trace_channel = int(trace_channels[slot])
            scale = int(timescale_indices[slot])
            a = features[current]
            gap = step - trace_last_steps[trace_channel]
            trace_value = (
                traces[scale * channel_count + trace_channel]
                * lambdas[scale] ** gap
            )
            b = min(1.0, trace_scales[scale] * trace_value)
            response = 0.0
            if a > support_threshold:
                drive = w_a[slot] * a + w_b[slot] * b
                delta = drive - mu[slot]
                response = math.exp(
                    -(delta * delta) / (2.0 * sigma[slot] * sigma[slot])
                )
            event_active = response >= activation_threshold
            if (
                learn
                and event_active
                and lifecycle[slot] == DORMANT
                and utility_bits[slot] >= consolidation_utility_threshold
            ):
                lifecycle[slot] = CONSOLIDATED
                dormant_steps[slot] = -1
            retained *= persistence_decay
            if event_active:
                retained = response
            persistence[slot] = retained
            state_active = not event_active and retained >= activation_threshold
            persistent_activation = 0.0
            if (
                state_observations[slot] >= persistence_min_observations
                and state_utility_bits[slot] >= persistence_utility_threshold
            ):
                count = float(state_observations[slot])
                persistent_activation = retained * count / (count + 16.0)
            activation = max(response, persistent_activation)
            relation_channel = vocab_size + slot
            features[relation_channel] = activation
            # Zero activation is literal axonal silence.  Leave the channel's
            # clock untouched and fold the elapsed decay into its next event
            # (or the materialization boundary) instead of writing six scales.
            if activation > 0.0:
                relation_gap = step - trace_last_steps[relation_channel]
                for trace_scale in range(lambdas.shape[0]):
                    index = trace_scale * channel_count + relation_channel
                    old_trace = traces[index]
                    value = (
                        old_trace * lambdas[trace_scale] ** relation_gap
                        + activation
                    )
                    traces[index] = value
                    if learn:
                        mean = _silent_mean_advance(
                            trace_means[index],
                            old_trace,
                            lambdas[trace_scale],
                            trace_scales[trace_scale],
                            trace_mean_decay,
                            relation_gap - 1,
                        )
                        trace_means[index] = (
                            trace_mean_decay * mean
                            + (1.0 - trace_mean_decay)
                            * min(1.0, trace_scales[trace_scale] * value)
                        )
                trace_last_steps[relation_channel] = step

            if learn:
                if lifecycle[slot] != DORMANT and a > support_threshold:
                    position = local_sample_positions[slot]
                    local_samples[slot, position, 0] = a
                    local_samples[slot, position, 1] = b
                    local_sample_positions[slot] = (
                        position + 1
                    ) % local_samples.shape[1]
                    local_sample_counts[slot] = min(
                        local_samples.shape[1], local_sample_counts[slot] + 1
                    )
                    observations[slot] += 1
                    n = stats_n[slot] + 1
                    da = a - stats_mean_a[slot]
                    db = b - stats_mean_b[slot]
                    dd = response - stats_mean_d[slot]
                    stats_mean_a[slot] += da / n
                    stats_mean_b[slot] += db / n
                    stats_mean_d[slot] += dd / n
                    stats_m2_a[slot] += da * (a - stats_mean_a[slot])
                    stats_m2_b[slot] += db * (b - stats_mean_b[slot])
                    stats_m2_d[slot] += dd * (response - stats_mean_d[slot])
                    stats_cov_da_acc[slot] += da * (
                        response - stats_mean_d[slot]
                    )
                    stats_cov_db_acc[slot] += db * (
                        response - stats_mean_d[slot]
                    )
                    stats_n[slot] = n
                    interval = adapt_interval
                    if lifecycle[slot] == CONSOLIDATED:
                        interval = consolidated_adapt_interval
                    if (
                        observations[slot] % interval == 0
                        and local_sample_counts[slot] >= min_adapt_samples
                    ):
                        _adapt_slot_counter(
                            slot,
                            observations[slot],
                            seed,
                            perturb_step,
                            local_samples,
                            local_sample_counts,
                            local_sample_positions,
                            w_a,
                            w_b,
                            mu,
                            sigma,
                            minimum_sigma,
                        )
                if event_active:
                    fires[slot] += 1
                    last_fire_steps[slot] = organism_step

            if (
                lifecycle[slot] != DORMANT
                and observations[slot] >= 2
                and fires[slot] >= 1
                and (event_active or state_active)
            ):
                state = 1 if state_active else 0
                local_position = local_counts[worker]
                local_ids[worker, local_position] = slot
                local_states[worker, local_position] = state
                local_counts[worker] = local_position + 1
                if state_active:
                    count = float(state_observations[slot])
                    evidence = count / (count + 16.0)
                    utility = state_utility_bits[slot]
                else:
                    count = float(observations[slot])
                    evidence = count / (count + 8.0)
                    utility = utility_bits[slot]
                if fires[slot] < 8:
                    utility_factor = 0.10
                elif utility > 0.0:
                    utility_factor = min(4.0, 0.25 + 4.0 * utility)
                else:
                    utility_factor = 0.0
                weight = activation * evidence * utility_factor
                if weight > 0.0:
                    column = slot // column_size
                    for rank in range(max_per_column):
                        old_weight = column_winner_weights[column, rank]
                        old_slot = column_winner_ids[column, rank]
                        if weight > old_weight or (
                            weight == old_weight and slot < old_slot
                        ):
                            for shift in range(max_per_column - 1, rank, -1):
                                column_winner_ids[column, shift] = (
                                    column_winner_ids[column, shift - 1]
                                )
                                column_winner_weights[column, shift] = (
                                    column_winner_weights[column, shift - 1]
                                )
                                column_winner_states[column, shift] = (
                                    column_winner_states[column, shift - 1]
                                )
                            column_winner_ids[column, rank] = slot
                            column_winner_weights[column, rank] = weight
                            column_winner_states[column, rank] = state
                            break

            if activation > support_threshold:
                edge_start = consumer_offsets[relation_channel]
                edge_stop = consumer_offsets[relation_channel + 1]
                for edge in range(edge_start, edge_stop):
                    scheduled_stamps[consumer_ids[edge]] = generation


@njit(cache=True, parallel=True)
def _resolve_outcomes_parallel(
    local_ids: np.ndarray,
    local_states: np.ndarray,
    local_counts: np.ndarray,
    token: int,
    base_probability: float,
    lifecycle: np.ndarray,
    predictor_counts: np.ndarray,
    state_predictor_counts: np.ndarray,
    log_predictor: np.ndarray,
    log_state_predictor: np.ndarray,
    utility_bits: np.ndarray,
    state_utility_bits: np.ndarray,
    state_observations: np.ndarray,
    relation_prior: float,
    utility_decay: float,
    predictor_decay: float,
    consolidated_predictor_decay: float,
) -> None:
    vocab_size = predictor_counts.shape[1]
    for worker in prange(local_counts.shape[0]):
        for position in range(local_counts[worker]):
            slot = local_ids[worker, position]
            state = local_states[worker, position]
            if lifecycle[slot] == DORMANT:
                continue
            if state:
                row = state_predictor_counts[slot]
                log_row = log_state_predictor[slot]
            else:
                row = predictor_counts[slot]
                log_row = log_predictor[slot]
            total = relation_prior * vocab_size
            for byte_id in range(vocab_size):
                total += row[byte_id]
            probability = (row[token] + relation_prior) / total
            gain = math.log2(max(probability, 1e-12) / max(base_probability, 1e-12))
            if state:
                state_utility_bits[slot] = (
                    utility_decay * state_utility_bits[slot]
                    + (1.0 - utility_decay) * gain
                )
                state_observations[slot] += 1
            else:
                utility_bits[slot] = (
                    utility_decay * utility_bits[slot]
                    + (1.0 - utility_decay) * gain
                )
            decay = predictor_decay
            if lifecycle[slot] == 1:
                decay = consolidated_predictor_decay
            total = relation_prior * vocab_size
            for byte_id in range(vocab_size):
                row[byte_id] *= decay
                if byte_id == token:
                    row[byte_id] += 1.0
                total += row[byte_id]
            for byte_id in range(vocab_size):
                log_row[byte_id] = math.log(
                    (row[byte_id] + relation_prior) / total
                )


@njit(cache=True)
def _observe_frozen_columns(
    tokens: np.ndarray,
    starting_step: int,
    starting_organism_step: int,
    has_pending: bool,
    pending: np.ndarray,
    pending_base: np.ndarray,
    local_ids: np.ndarray,
    local_states: np.ndarray,
    local_counts: np.ndarray,
    previous_token: int,
    features: np.ndarray,
    worker_starts: np.ndarray,
    worker_stops: np.ndarray,
    column_size: int,
    column_count: int,
    max_depth: int,
    current_channels: np.ndarray,
    trace_channels: np.ndarray,
    timescale_indices: np.ndarray,
    depths: np.ndarray,
    live: np.ndarray,
    lifecycle: np.ndarray,
    observations: np.ndarray,
    fires: np.ndarray,
    last_fire_steps: np.ndarray,
    born_steps: np.ndarray,
    consolidated_steps: np.ndarray,
    dormant_steps: np.ndarray,
    utility_bits: np.ndarray,
    state_observations: np.ndarray,
    state_utility_bits: np.ndarray,
    log_predictor: np.ndarray,
    log_state_predictor: np.ndarray,
    predictor_counts: np.ndarray,
    state_predictor_counts: np.ndarray,
    base_counts: np.ndarray,
    context_counts: np.ndarray,
    consumer_offsets: np.ndarray,
    consumer_ids: np.ndarray,
    lambdas: np.ndarray,
    trace_scales: np.ndarray,
    traces: np.ndarray,
    trace_means: np.ndarray,
    trace_last_steps: np.ndarray,
    persistence: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    local_samples: np.ndarray,
    local_sample_counts: np.ndarray,
    local_sample_positions: np.ndarray,
    stats_n: np.ndarray,
    stats_mean_a: np.ndarray,
    stats_mean_b: np.ndarray,
    stats_mean_d: np.ndarray,
    stats_m2_a: np.ndarray,
    stats_m2_b: np.ndarray,
    stats_m2_d: np.ndarray,
    stats_cov_da_acc: np.ndarray,
    stats_cov_db_acc: np.ndarray,
    support_threshold: float,
    activation_threshold: float,
    persistence_decay: float,
    persistence_min_observations: int,
    persistence_utility_threshold: float,
    max_per_column: int,
    max_columns: int,
    max_active: int,
    utility_gated_persistence: bool,
    context_prior: float,
    context_decay: float,
    base_decay: float,
    relation_prior: float,
    utility_decay: float,
    predictor_decay: float,
    consolidated_predictor_decay: float,
    trace_mean_decay: float,
    adapt_interval: int,
    consolidated_adapt_interval: int,
    min_adapt_samples: int,
    seed: int,
    perturb_step: float,
    minimum_sigma: float,
    lifecycle_sweep_interval: int,
    consolidation_min_age: int,
    consolidation_min_fires: int,
    consolidation_utility_threshold: float,
    dormancy_staleness: int,
    prune_utility: float,
    structural_barriers: bool,
    surprise_birth_bits: float,
    birth_cooldown: int,
    last_birth_step: int,
    learn: bool,
) -> tuple[np.ndarray, np.ndarray, int, int, int, bool]:
    vocab_size = 256
    relation_count = live.shape[0]
    channel_count = vocab_size + relation_count
    scheduled_stamps = np.zeros(relation_count, dtype=np.int64)
    column_winner_ids = np.full(
        (column_count, max_per_column), -1, dtype=np.int64
    )
    column_winner_weights = np.zeros(
        (column_count, max_per_column), dtype=np.float32
    )
    column_winner_states = np.zeros(
        (column_count, max_per_column), dtype=np.uint8
    )
    column_scores = np.zeros(column_count, dtype=np.float32)
    observed = np.full(tokens.shape[0], np.nan, dtype=np.float32)
    base_observed = np.full(tokens.shape[0], np.nan, dtype=np.float32)
    evaluated_total = 0
    processed = 0
    barrier = False

    for step in range(tokens.shape[0]):
        causal_step = starting_step + step
        organism_step = starting_organism_step + step
        token = int(tokens[step])
        if has_pending:
            if (
                learn
                and structural_barriers
                and -math.log2(max(pending[token], 1e-12))
                >= surprise_birth_bits
                and organism_step - last_birth_step >= birth_cooldown
            ):
                barrier = True
                break
            observed[step] = pending[token]
            base_observed[step] = pending_base[token]
            if learn:
                _resolve_outcomes_parallel(
                    local_ids,
                    local_states,
                    local_counts,
                    token,
                    pending_base[token],
                    lifecycle,
                    predictor_counts,
                    state_predictor_counts,
                    log_predictor,
                    log_state_predictor,
                    utility_bits,
                    state_utility_bits,
                    state_observations,
                    relation_prior,
                    utility_decay,
                    predictor_decay,
                    consolidated_predictor_decay,
                )
        if learn and previous_token >= 0:
            for byte_id in range(vocab_size):
                context_counts[previous_token, byte_id] *= context_decay
            context_counts[previous_token, token] += 1.0
        if learn:
            for byte_id in range(vocab_size):
                base_counts[byte_id] *= base_decay
            base_counts[token] += 1.0
        previous_token = token
        for worker in range(local_counts.shape[0]):
            local_counts[worker] = 0
        for column in range(column_count):
            for rank in range(max_per_column):
                column_winner_ids[column, rank] = -1
                column_winner_weights[column, rank] = -math.inf
        for byte_id in range(vocab_size):
            if features[byte_id] > 0.0:
                features[byte_id] = 0.0
        features[token] = 1.0
        for worker in range(worker_starts.shape[0]):
            for slot in range(worker_starts[worker], worker_stops[worker]):
                features[vocab_size + slot] = 0.0

        generation = causal_step + 1
        sensory_start = consumer_offsets[token]
        sensory_stop = consumer_offsets[token + 1]
        for edge in range(sensory_start, sensory_stop):
            scheduled_stamps[consumer_ids[edge]] = generation

        for scale in range(lambdas.shape[0]):
            index = scale * channel_count + token
            gap = causal_step - trace_last_steps[token]
            old_trace = traces[index]
            value = old_trace * lambdas[scale] ** gap + 1.0
            traces[index] = value
            if learn:
                mean = _silent_mean_advance(
                    trace_means[index],
                    old_trace,
                    lambdas[scale],
                    trace_scales[scale],
                    trace_mean_decay,
                    gap - 1,
                )
                trace_means[index] = (
                    trace_mean_decay * mean
                    + (1.0 - trace_mean_decay)
                    * min(1.0, trace_scales[scale] * value)
                )
        trace_last_steps[token] = causal_step

        for depth in range(1, max_depth + 1):
            _evaluate_depth_parallel(
                worker_starts,
                worker_stops,
                depth,
                causal_step,
                organism_step,
                generation,
                vocab_size,
                current_channels,
                trace_channels,
                timescale_indices,
                depths,
                live,
                lifecycle,
                observations,
                fires,
                last_fire_steps,
                dormant_steps,
                utility_bits,
                state_observations,
                state_utility_bits,
                consumer_offsets,
                consumer_ids,
                lambdas,
                trace_scales,
                traces,
                trace_means,
                trace_last_steps,
                persistence,
                w_a,
                w_b,
                mu,
                sigma,
                features,
                local_ids,
                local_states,
                local_counts,
                column_winner_ids,
                column_winner_weights,
                column_winner_states,
                column_size,
                max_per_column,
                scheduled_stamps,
                local_samples,
                local_sample_counts,
                local_sample_positions,
                stats_n,
                stats_mean_a,
                stats_mean_b,
                stats_mean_d,
                stats_m2_a,
                stats_m2_b,
                stats_m2_d,
                stats_cov_da_acc,
                stats_cov_db_acc,
                support_threshold,
                activation_threshold,
                persistence_decay,
                persistence_min_observations,
                persistence_utility_threshold,
                consolidation_utility_threshold,
                utility_gated_persistence,
                trace_mean_decay,
                adapt_interval,
                consolidated_adapt_interval,
                min_adapt_samples,
                seed,
                perturb_step,
                minimum_sigma,
                learn,
            )

        if learn and organism_step % lifecycle_sweep_interval == 0:
            _lifecycle_sweep_parallel(
                worker_starts,
                worker_stops,
                organism_step,
                live,
                lifecycle,
                born_steps,
                last_fire_steps,
                fires,
                utility_bits,
                consolidated_steps,
                dormant_steps,
                consolidation_min_age,
                consolidation_min_fires,
                consolidation_utility_threshold,
                dormancy_staleness,
                prune_utility,
            )

        for column in range(column_count):
            score = 0.0
            for rank in range(max_per_column):
                slot = column_winner_ids[column, rank]
                if slot >= 0 and lifecycle[slot] != DORMANT:
                    score += column_winner_weights[column, rank]
            column_scores[column] = score

        selected_columns = np.zeros(column_count, dtype=np.uint8)
        for _rank in range(min(max_columns, column_count)):
            best_column = -1
            best_score = 0.0
            for column in range(column_count):
                if not selected_columns[column] and column_scores[column] > best_score:
                    best_column = column
                    best_score = column_scores[column]
            if best_column < 0:
                break
            selected_columns[best_column] = 1

        winner_ids = np.full(max_active, -1, dtype=np.int64)
        winner_weights = np.full(max_active, -math.inf, dtype=np.float32)
        winner_states = np.zeros(max_active, dtype=np.uint8)
        winner_count = 0
        for column in range(column_count):
            if not selected_columns[column]:
                continue
            for local_rank in range(max_per_column):
                slot = column_winner_ids[column, local_rank]
                if slot < 0 or lifecycle[slot] == DORMANT:
                    continue
                weight = column_winner_weights[column, local_rank]
                insertion = min(winner_count, max_active - 1)
                if winner_count >= max_active and weight <= winner_weights[insertion]:
                    continue
                while insertion > 0 and weight > winner_weights[insertion - 1]:
                    winner_ids[insertion] = winner_ids[insertion - 1]
                    winner_weights[insertion] = winner_weights[insertion - 1]
                    winner_states[insertion] = winner_states[insertion - 1]
                    insertion -= 1
                winner_ids[insertion] = slot
                winner_weights[insertion] = weight
                winner_states[insertion] = column_winner_states[
                    column, local_rank
                ]
                winner_count = min(max_active, winner_count + 1)

        context_total = 0.0
        for byte_id in range(vocab_size):
            context_total += context_counts[token, byte_id] + context_prior
        maximum_logit = -math.inf
        for byte_id in range(vocab_size):
            total_weight = 1.0
            pending_base[byte_id] = (
                context_counts[token, byte_id] + context_prior
            ) / context_total
            logit = math.log(pending_base[byte_id])
            for winner in range(winner_count):
                slot = winner_ids[winner]
                if winner_states[winner]:
                    logit += winner_weights[winner] * log_state_predictor[
                        slot, byte_id
                    ]
                else:
                    logit += winner_weights[winner] * log_predictor[slot, byte_id]
                total_weight += winner_weights[winner]
            pending[byte_id] = logit / total_weight
            maximum_logit = max(maximum_logit, pending[byte_id])
        probability_sum = 0.0
        for byte_id in range(vocab_size):
            pending[byte_id] = math.exp(pending[byte_id] - maximum_logit)
            probability_sum += pending[byte_id]
        for byte_id in range(vocab_size):
            pending[byte_id] /= probability_sum
        has_pending = True
        processed += 1
    return (
        observed[:processed],
        base_observed[:processed],
        evaluated_total,
        previous_token,
        processed,
        barrier,
    )


class CpuColumnFrozenEvaluator:
    """Fixed-anatomy ALM evaluator with resident CPU-column plasticity."""

    def __init__(
        self,
        learner: OnlineDendriticStreamLearner,
        *,
        workers: int = 30,
        resume_dynamic: bool = False,
    ) -> None:
        if not NUMBA_AVAILABLE or set_num_threads is None:
            raise RuntimeError("Numba is required for resident CPU columns")
        config = learner.config
        if config.vocab_size != 256 or not config.contextual_residual:
            raise ValueError("CPU columns require byte contextual residuals")
        partition = partition_learner_columns(
            learner, workers, strategy="contiguous"
        )
        self.workers = partition.worker_count
        self.column_count = math.ceil(
            config.max_relations / config.column_size
        )
        self.local_winners_per_column = (
            config.max_active_relations
            if self.column_count == 1
            else config.max_active_per_column
        )
        set_num_threads(self.workers)
        starts = np.zeros(self.workers, dtype=np.int64)
        stops = np.zeros(self.workers, dtype=np.int64)
        for worker in range(self.workers):
            columns = np.flatnonzero(partition.column_to_worker == worker)
            starts[worker] = int(columns[0]) * config.column_size
            stops[worker] = min(
                (int(columns[-1]) + 1) * config.column_size,
                config.max_relations,
            )
        index = AxonalConsumerIndex.from_learner(learner)
        self.config = config
        self.worker_starts = starts
        self.worker_stops = stops
        self.current_channels = learner._current_channels.copy()
        self.trace_channels = learner._trace_channels.copy()
        self.timescale_indices = learner._timescale_indices.copy()
        self.depths = learner._depths.copy()
        self.live = learner._live.copy()
        self.lifecycle = learner._lifecycle.copy()
        self.observations = learner._observations.copy()
        self.fires = learner._fires.copy()
        self.last_fire_steps = learner._last_fire_steps.copy()
        self.born_steps = learner._born_steps.copy()
        self.consolidated_steps = learner._consolidated_steps.copy()
        self.dormant_steps = learner._dormant_steps.copy()
        self.utility_bits = learner._utility_bits.astype(np.float32)
        self.state_observations = learner._state_observations.copy()
        self.state_utility_bits = learner._state_utility_bits.astype(np.float32)
        prior = config.relation_prior
        self.predictor_counts = learner._predictor_counts.copy()
        self.state_predictor_counts = learner._state_predictor_counts.copy()
        predictor = self.predictor_counts + prior
        state_predictor = self.state_predictor_counts + prior
        self.log_predictor = np.log(
            predictor / predictor.sum(axis=1, keepdims=True)
        ).astype(np.float32)
        self.log_state_predictor = np.log(
            state_predictor / state_predictor.sum(axis=1, keepdims=True)
        ).astype(np.float32)
        self.context_counts = learner.context_counts.copy()
        self.base_counts = learner.base_counts.copy()
        self.consumer_offsets = index.offsets
        self.consumer_ids = index.slot_ids
        self.lambdas = learner.lambdas.astype(np.float32)
        self.trace_scales = learner.trace_scales.astype(np.float32)
        self.w_a = learner._w_a.astype(np.float32)
        self.w_b = learner._w_b.astype(np.float32)
        self.mu = learner._mu.astype(np.float32)
        self.sigma = learner._sigma.astype(np.float32)
        self.local_samples = learner._local_samples.copy()
        self.local_sample_counts = learner._local_sample_counts.copy()
        self.local_sample_positions = learner._local_sample_positions.copy()
        for name in (
            "stats_n",
            "stats_mean_a",
            "stats_mean_b",
            "stats_mean_d",
            "stats_m2_a",
            "stats_m2_b",
            "stats_m2_d",
            "stats_cov_da_acc",
            "stats_cov_db_acc",
        ):
            setattr(self, name, getattr(learner, f"_{name}").copy())
        self.organism_steps = learner.stats.steps
        self.reset_episode()
        if resume_dynamic:
            self._resume_dynamic_from(learner)
        self._synced_lifecycle = self.lifecycle.copy()

    def _resume_dynamic_from(self, learner: OnlineDendriticStreamLearner) -> None:
        self.traces[:] = learner.traces
        self.trace_means[:] = learner.trace_means
        self.persistence[:] = learner._persistence
        self.previous_token = learner.previous_token
        pending = learner._pending
        if pending is None:
            return
        self.pending[:] = pending.distribution
        self.pending_base[:] = pending.base_distribution
        self.features[:] = pending.features
        for slot, state in zip(
            pending.relation_ids, pending.relation_is_state, strict=True
        ):
            slot_id = int(slot)
            worker = int(np.searchsorted(self.worker_stops, slot_id, side="right"))
            position = int(self.local_counts[worker])
            self.local_ids[worker, position] = slot_id
            self.local_states[worker, position] = int(state)
            self.local_counts[worker] = position + 1
        self.has_pending = True

    def reset_episode(self) -> None:
        channels = self.config.vocab_size + self.config.max_relations
        self.traces = np.zeros(
            (len(self.config.timescales), channels), dtype=np.float32
        )
        self.trace_means = np.full_like(self.traces, 0.05)
        self.trace_last_steps = np.full(channels, -1, dtype=np.int64)
        self.persistence = np.zeros(self.config.max_relations, dtype=np.float32)
        self.pending = np.zeros(self.config.vocab_size, dtype=np.float32)
        self.pending_base = np.zeros(self.config.vocab_size, dtype=np.float32)
        territory_capacity = int(np.max(self.worker_stops - self.worker_starts))
        self.local_ids = np.empty(
            (self.workers, territory_capacity), dtype=np.int64
        )
        self.local_states = np.empty(
            (self.workers, territory_capacity), dtype=np.uint8
        )
        self.local_counts = np.zeros(self.workers, dtype=np.int64)
        self.features = np.zeros(channels, dtype=np.float32)
        self.has_pending = False
        self.causal_steps = 0
        self.previous_token = -1
        self._episode_learn_mode: bool | None = None
        self._learned_steps = 0
        self._learned_scored = 0
        self._synced_steps = 0
        self._synced_scored = 0
        self._synced_fires = self.fires.copy()

    def _materialize_traces(self) -> None:
        if self.causal_steps == 0:
            return
        _materialize_lazy_traces(
            self.traces.ravel(),
            self.trace_means.ravel(),
            self.trace_last_steps,
            self.lambdas,
            self.trace_scales,
            self.causal_steps - 1,
            self.config.trace_mean_decay,
            bool(self._episode_learn_mode),
        )

    def _observe_chunk_probabilities(
        self,
        tokens: np.ndarray,
        *,
        learn: bool,
        structural_barriers: bool,
        last_birth_step: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        token_array = np.ascontiguousarray(tokens, dtype=np.uint8).reshape(-1)
        had_pending = self.has_pending
        if len(token_array):
            if self._episode_learn_mode is None:
                self._episode_learn_mode = learn
            elif self._episode_learn_mode != learn:
                raise ValueError("resident lazy traces require one mode per episode")
        (
            observed,
            base_observed,
            _evaluated,
            self.previous_token,
            processed,
            barrier,
        ) = _observe_frozen_columns(
            token_array,
            self.causal_steps,
            self.organism_steps,
            self.has_pending,
            self.pending,
            self.pending_base,
            self.local_ids,
            self.local_states,
            self.local_counts,
            self.previous_token,
            self.features,
            self.worker_starts,
            self.worker_stops,
            self.config.column_size,
            self.column_count,
            self.config.max_depth,
            self.current_channels,
            self.trace_channels,
            self.timescale_indices,
            self.depths,
            self.live,
            self.lifecycle,
            self.observations,
            self.fires,
            self.last_fire_steps,
            self.born_steps,
            self.consolidated_steps,
            self.dormant_steps,
            self.utility_bits,
            self.state_observations,
            self.state_utility_bits,
            self.log_predictor,
            self.log_state_predictor,
            self.predictor_counts,
            self.state_predictor_counts,
            self.base_counts,
            self.context_counts,
            self.consumer_offsets,
            self.consumer_ids,
            self.lambdas,
            self.trace_scales,
            self.traces.ravel(),
            self.trace_means.ravel(),
            self.trace_last_steps,
            self.persistence,
            self.w_a,
            self.w_b,
            self.mu,
            self.sigma,
            self.local_samples,
            self.local_sample_counts,
            self.local_sample_positions,
            self.stats_n,
            self.stats_mean_a,
            self.stats_mean_b,
            self.stats_mean_d,
            self.stats_m2_a,
            self.stats_m2_b,
            self.stats_m2_d,
            self.stats_cov_da_acc,
            self.stats_cov_db_acc,
            self.config.support_threshold,
            self.config.activation_threshold,
            self.config.persistence_decay,
            self.config.persistence_min_observations,
            self.config.persistence_utility_threshold,
            self.local_winners_per_column,
            self.config.max_active_columns,
            self.config.max_active_relations,
            self.config.utility_gated_persistence,
            self.config.context_prior,
            self.config.context_decay,
            self.config.base_decay,
            self.config.relation_prior,
            self.config.utility_decay,
            self.config.predictor_decay,
            self.config.consolidated_predictor_decay,
            self.config.trace_mean_decay,
            self.config.adapt_interval,
            self.config.consolidated_adapt_interval,
            self.config.min_adapt_samples,
            self.config.seed,
            self.config.perturb_step,
            self.config.initial_sigma / 4.0,
            self.config.lifecycle_sweep_interval,
            self.config.consolidation_min_age,
            self.config.consolidation_min_fires,
            self.config.consolidation_utility_threshold,
            self.config.dormancy_staleness,
            self.config.prune_utility,
            structural_barriers,
            self.config.surprise_birth_bits,
            self.config.birth_cooldown,
            last_birth_step,
            learn,
        )
        if not structural_barriers and (barrier or processed != len(token_array)):
            raise RuntimeError("fixed-anatomy resident loop stopped early")
        if processed:
            self.causal_steps += processed
            self.has_pending = True
            if learn:
                self.organism_steps += processed
                self._learned_steps += processed
                self._learned_scored += processed - int(not had_pending)
        return observed, base_observed, barrier

    def observe_chunk_probabilities(
        self, tokens: np.ndarray, *, learn: bool = False
    ) -> np.ndarray:
        observed, _base_observed, _barrier = self._observe_chunk_probabilities(
            tokens,
            learn=learn,
            structural_barriers=False,
            last_birth_step=-1,
        )
        return observed

    def observe_until_structural_barrier(
        self, tokens: np.ndarray, *, last_birth_step: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Learn until the next byte must enter developmental Python."""
        return self._observe_chunk_probabilities(
            tokens,
            learn=True,
            structural_barriers=True,
            last_birth_step=last_birth_step,
        )

    def synchronize_plasticity_to(
        self, learner: OnlineDendriticStreamLearner
    ) -> None:
        """Return resident local learning to the developmental organism."""
        if learner.config.max_relations != self.config.max_relations:
            raise ValueError("CPU organisms have different anatomy")
        self._materialize_traces()
        learner.base_counts[:] = self.base_counts
        learner.context_counts[:] = self.context_counts
        learner._predictor_counts[:] = self.predictor_counts
        learner._state_predictor_counts[:] = self.state_predictor_counts
        learner._utility_bits[:] = self.utility_bits
        learner._state_utility_bits[:] = self.state_utility_bits
        learner._state_observations[:] = self.state_observations
        learner._observations[:] = self.observations
        learner._fires[:] = self.fires
        learner._last_fire_steps[:] = self.last_fire_steps
        learner._lifecycle[:] = self.lifecycle
        learner._consolidated_steps[:] = self.consolidated_steps
        learner._dormant_steps[:] = self.dormant_steps
        learner._local_samples[:] = self.local_samples
        learner._local_sample_counts[:] = self.local_sample_counts
        learner._local_sample_positions[:] = self.local_sample_positions
        learner._w_a[:] = self.w_a
        learner._w_b[:] = self.w_b
        learner._mu[:] = self.mu
        learner._sigma[:] = self.sigma
        for name in (
            "stats_n",
            "stats_mean_a",
            "stats_mean_b",
            "stats_mean_d",
            "stats_m2_a",
            "stats_m2_b",
            "stats_m2_d",
            "stats_cov_da_acc",
            "stats_cov_db_acc",
        ):
            getattr(learner, f"_{name}")[:] = getattr(self, name)
        learner.traces[:] = self.traces
        learner.trace_means[:] = self.trace_means
        learner._persistence[:] = self.persistence
        learner.previous_token = self.previous_token

        positions = []
        states = []
        for worker in range(self.workers):
            count = int(self.local_counts[worker])
            positions.extend(int(value) for value in self.local_ids[worker, :count])
            states.extend(int(value) for value in self.local_states[worker, :count])
        if positions:
            order = np.argsort(np.asarray(positions), kind="stable")
            relation_ids = np.asarray(positions, dtype=np.intp)[order]
            relation_states = np.asarray(states, dtype=np.bool_)[order]
        else:
            relation_ids = np.empty(0, dtype=np.intp)
            relation_states = np.empty(0, dtype=np.bool_)
        normalized = np.clip(
            learner.trace_scales[:, None] * learner.traces, 0.0, 1.0
        )
        learner._pending = (
            _PendingPrediction(
                distribution=self.pending.astype(np.float64),
                base_distribution=self.pending_base.astype(np.float64),
                relation_ids=relation_ids,
                relation_is_state=relation_states,
                active_relations=(),
                features=self.features.astype(np.float64),
                traces=normalized,
            )
            if self.has_pending
            else None
        )
        learner._materialize_slot_views()
        for raw_slot in np.flatnonzero(learner._live):
            slot_id = int(raw_slot)
            slot = learner.slots[slot_id]
            assert slot is not None
            slot.predictor_counts = learner._predictor_counts[slot_id]
            slot.state_predictor_counts = learner._state_predictor_counts[slot_id]
            slot.observations = int(learner._observations[slot_id])
            slot.fires = int(learner._fires[slot_id])
            slot.last_fire_step = int(learner._last_fire_steps[slot_id])
            slot.utility_bits = float(learner._utility_bits[slot_id])
            slot.state_observations = int(learner._state_observations[slot_id])
            slot.state_utility_bits = float(learner._state_utility_bits[slot_id])
            slot.persistence = float(learner._persistence[slot_id])
            slot.lifecycle = LIFECYCLE_NAMES[int(learner._lifecycle[slot_id])]
            slot.consolidated_step = int(learner._consolidated_steps[slot_id])
            slot.dormant_step = int(learner._dormant_steps[slot_id])

        learner.stats.steps += self._learned_steps - self._synced_steps
        learner.stats.predictions_scored += (
            self._learned_scored - self._synced_scored
        )
        learner.stats.relation_firings += int(
            np.sum(self.fires - self._synced_fires)
        )
        learner.stats.consolidations += int(
            np.sum(
                (self._synced_lifecycle == JUVENILE)
                & (self.lifecycle == CONSOLIDATED)
            )
        )
        learner.stats.dormancies += int(
            np.sum(
                (self._synced_lifecycle == CONSOLIDATED)
                & (self.lifecycle == DORMANT)
            )
        )
        learner.stats.reactivations += int(
            np.sum(
                (self._synced_lifecycle == DORMANT)
                & (self.lifecycle == CONSOLIDATED)
            )
        )
        self._synced_steps = self._learned_steps
        self._synced_scored = self._learned_scored
        self._synced_fires = self.fires.copy()
        self._synced_lifecycle = self.lifecycle.copy()


class CpuColumnDevelopmentalRunner:
    """Keep columns resident while Python handles rare anatomical changes."""

    def __init__(
        self, learner: OnlineDendriticStreamLearner, *, workers: int = 30
    ) -> None:
        self.learner = learner
        self.workers = workers
        self.resident = CpuColumnFrozenEvaluator(
            learner, workers=workers, resume_dynamic=learner._pending is not None
        )

    def observe_chunk_probabilities(
        self, tokens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        token_array = np.ascontiguousarray(tokens, dtype=np.uint8).reshape(-1)
        parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        offset = 0
        while offset < len(token_array):
            observed, base_observed, barrier = (
                self.resident.observe_until_structural_barrier(
                    token_array[offset:],
                    last_birth_step=self.learner._last_birth_step,
                )
            )
            if len(observed):
                parts.append(observed)
                base_parts.append(base_observed)
                offset += len(observed)
            if not barrier:
                break
            self.synchronize()
            step = self.learner.observe(int(token_array[offset]), learn=True)
            parts.append(
                np.asarray(
                    [
                        np.nan
                        if step.observed_probability is None
                        else step.observed_probability
                    ],
                    dtype=np.float32,
                )
            )
            base_parts.append(
                np.asarray(
                    [
                        np.nan
                        if step.base_observed_probability is None
                        else step.base_observed_probability
                    ],
                    dtype=np.float32,
                )
            )
            offset += 1
            self.resident = CpuColumnFrozenEvaluator(
                self.learner, workers=self.workers, resume_dynamic=True
            )
        if not parts:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty.copy()
        return np.concatenate(parts), np.concatenate(base_parts)

    def synchronize(self) -> None:
        self.resident.synchronize_plasticity_to(self.learner)


def observe_developmental_probabilities(
    learner: OnlineDendriticStreamLearner,
    tokens: np.ndarray,
    *,
    workers: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one resident developmental stream and return synchronized state."""
    runner = CpuColumnDevelopmentalRunner(learner, workers=workers)
    observed, base_observed = runner.observe_chunk_probabilities(tokens)
    runner.synchronize()
    return observed, base_observed
