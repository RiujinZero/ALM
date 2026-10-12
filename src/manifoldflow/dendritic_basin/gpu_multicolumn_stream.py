"""Cooperative CUDA execution for frozen multi-column dendritic ALMs."""

from __future__ import annotations

from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.axonal_index import AxonalConsumerIndex
from manifoldflow.dendritic_basin.gpu_online_stream import _cupy
from manifoldflow.dendritic_basin.online_stream import (
    CONSOLIDATED,
    DORMANT,
    JUVENILE,
    LIFECYCLE_NAMES,
    OnlineDendriticStreamLearner,
    _PendingPrediction,
)

_MAX_ACTIVE = 32
_MAX_PER_COLUMN = 32

_SOURCE = rf"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define VOCAB 256
#define MAX_ACTIVE {_MAX_ACTIVE}
#define MAX_LOCAL {_MAX_PER_COLUMN}

__device__ __forceinline__ float silent_mean_advance(
    float mean, float trace, float lambda, float scale, float mean_decay,
    long long silent_steps
) {{
    if (silent_steps <= 0) return mean;
    float steps = (float)silent_steps;
    float mean_power = powf(mean_decay, steps);
    float trace_power = powf(lambda, steps);
    float weighted;
    if (fabsf(mean_decay - lambda) < 1.0e-7f) {{
        weighted = steps * trace_power;
    }} else {{
        weighted = lambda * (mean_power - trace_power) /
            (mean_decay - lambda);
    }}
    return mean_power * mean +
        (1.0f - mean_decay) * scale * trace * weighted;
}}

__device__ __forceinline__ unsigned long long splitmix64(
    unsigned long long value
) {{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}}

__device__ __forceinline__ double ring_interaction_objective(
    const float* samples, long long count, long long position, int capacity,
    double w_a, double w_b, double mu, double sigma
) {{
    double mean_a = 0.0, mean_b = 0.0, mean_d = 0.0;
    double m2_a = 0.0, m2_b = 0.0, m2_d = 0.0;
    double cov_da = 0.0, cov_db = 0.0;
    long long start = count == capacity ? position : 0;
    for (long long offset = 0; offset < count; ++offset) {{
        long long index = (start + offset) % capacity;
        double a = samples[2 * index];
        double b = samples[2 * index + 1];
        double drive = w_a * a + w_b * b;
        double delta = drive - mu;
        double d = exp(-(delta * delta) / (2.0 * sigma * sigma));
        double n = (double)(offset + 1);
        double da = a - mean_a;
        double db = b - mean_b;
        double dd = d - mean_d;
        mean_a += da / n;
        mean_b += db / n;
        mean_d += dd / n;
        m2_a += da * (a - mean_a);
        m2_b += db * (b - mean_b);
        m2_d += dd * (d - mean_d);
        cov_da += da * (d - mean_d);
        cov_db += db * (d - mean_d);
    }}
    if (count == 0) return 0.0;
    double n = (double)count;
    double var_a = m2_a / n;
    double var_b = m2_b / n;
    double var_d = m2_d / n;
    double relation_a = cov_da / n;
    double relation_b = cov_db / n;
    return var_d
        - relation_a * relation_a / (var_a + 1.0e-8)
        - relation_b * relation_b / (var_b + 1.0e-8);
}}

__device__ __forceinline__ void adapt_slot_counter(
    int slot, long long observation_count, long long seed, float perturb_step,
    float* local_samples, long long sample_count, long long sample_position,
    int local_buffer_size, float* w_a, float* w_b, float* mu, float* sigma,
    float minimum_sigma
) {{
    unsigned long long base = (unsigned long long)seed
        ^ ((unsigned long long)(slot + 1) * 0xD2B74407B1CE6E93ULL)
        ^ ((unsigned long long)(observation_count + 1)
            * 0xCA5A826395121157ULL);
    double proposal[4];
    const double inverse_53 = 1.0 / 9007199254740992.0;
    for (int pair = 0; pair < 2; ++pair) {{
        unsigned long long first = splitmix64(base ^ (2 * pair + 1));
        unsigned long long second = splitmix64(base ^ (2 * pair + 2));
        double u1 = ((double)(first >> 11) + 0.5) * inverse_53;
        double u2 = ((double)(second >> 11) + 0.5) * inverse_53;
        double radius = sqrt(-2.0 * log(u1)) * (double)perturb_step;
        double angle = 6.2831853071795864769 * u2;
        proposal[2 * pair] = radius * cos(angle);
        proposal[2 * pair + 1] = radius * sin(angle);
    }}
    double candidate_w_a = (double)w_a[slot] + proposal[0];
    double candidate_w_b = (double)w_b[slot] + proposal[1];
    double candidate_mu = (double)mu[slot] + proposal[2];
    double candidate_sigma = fmax(1.0e-3, (double)sigma[slot] + proposal[3]);
    float* samples = local_samples
        + (long long)slot * local_buffer_size * 2;
    double current_j = ring_interaction_objective(
        samples, sample_count, sample_position, local_buffer_size,
        w_a[slot], w_b[slot], mu[slot], sigma[slot]
    );
    double candidate_j = ring_interaction_objective(
        samples, sample_count, sample_position, local_buffer_size,
        candidate_w_a, candidate_w_b, candidate_mu, candidate_sigma
    );
    if (candidate_j > current_j) {{
        w_a[slot] = (float)candidate_w_a;
        w_b[slot] = (float)candidate_w_b;
        mu[slot] = (float)candidate_mu;
        sigma[slot] = (float)candidate_sigma;
    }}
    sigma[slot] = fmaxf(minimum_sigma, sigma[slot]);
}}

extern "C" __global__ void frozen_multicolumn_chunk(
    const unsigned char* tokens, int steps,
    const unsigned char* live, const long long* depths,
    const long long* current_channels, const long long* trace_channels,
    const long long* timescale_indices, signed char* lifecycle,
    const long long* consumer_offsets, const int* consumer_ids,
    const int* consumer_depths,
    long long* observations, long long* fires, const long long* born_steps,
    long long* last_fire_steps, long long* consolidated_steps,
    long long* dormant_steps,
    float* utility_bits, long long* state_observations,
    float* state_utility_bits,
    float* base_counts, float* context_counts,
    float* predictor_counts, float* state_predictor_counts,
    float* log_context_probability,
    float* log_predictor_probability,
    float* log_state_predictor_probability,
    const float* lambdas, const float* trace_scales,
    float* traces, float* trace_means, long long* trace_last_steps,
    float* persistence,
    float* w_a, float* w_b, float* mu, float* sigma,
    float* features, float* raw, float* state_probe,
    float* current_values, float* trace_values,
    unsigned char* evaluated, unsigned long long* evaluated_total,
    unsigned long long* consolidation_total,
    unsigned long long* dormancy_total,
    unsigned long long* reactivation_total,
    int* scheduled_stamps, int* depth_queues, int* depth_counts,
    int* persistent_a, int* persistent_b,
    int* persistent_count_a, int* persistent_count_b,
    int* persistent_toggle, int* sparse_generation,
    float* local_samples, long long* local_sample_counts,
    long long* local_sample_positions, long long* stats_n,
    float* stats_mean_a, float* stats_mean_b, float* stats_mean_d,
    float* stats_m2_a, float* stats_m2_b, float* stats_m2_d,
    float* stats_cov_da_acc, float* stats_cov_db_acc,
    float* candidate_weight, int* candidate_state,
    int* column_winner_ids, float* column_winner_weights,
    float* column_scores, int* final_winner_ids, int* final_winner_state,
    float* final_winner_weights, int* final_winner_count,
    float* pending, int* has_pending, int* previous_token,
    int* pending_local_state, float* pending_base,
    float* observed_probabilities, float* base_observed_probabilities,
    int* processed_steps,
    int* structural_barrier,
    int timescale_count, int max_relations, int column_count, int max_depth,
    int local_buffer_size,
    float support_threshold, float activation_threshold,
    float persistence_decay, long long persistence_min_observations,
    float persistence_utility_threshold,
    float relation_prior, float base_decay, float context_decay,
    float predictor_decay, float consolidated_predictor_decay,
    float utility_decay, float trace_mean_decay, float context_prior,
    int consolidated_lifecycle,
    int max_per_column, int max_columns, int max_active, int event_sparse,
    int utility_gated_persistence, long long starting_causal_step, int learn,
    int stop_at_structural_barrier, float surprise_birth_bits,
    long long birth_cooldown, long long starting_organism_step,
    long long last_birth_step, int juvenile_lifecycle, int dormant_lifecycle,
    int adapt_interval, int consolidated_adapt_interval,
    long long min_adapt_samples, long long seed, float perturb_step,
    float minimum_sigma, int lifecycle_sweep_interval,
    long long consolidation_min_age, long long consolidation_min_fires,
    float consolidation_utility_threshold, long long dormancy_staleness,
    float prune_utility
) {{
    cg::grid_group grid = cg::this_grid();
    int lane = threadIdx.x;
    int column = blockIdx.x;
    int global_thread = column * blockDim.x + lane;
    int global_threads = gridDim.x * blockDim.x;
    int channel_count = VOCAB + max_relations;
    __shared__ float work[VOCAB];

    for (int step = 0; step < steps; ++step) {{
        int token = (int)tokens[step];
        long long causal_step = starting_causal_step + step;
        long long organism_step = starting_organism_step + step;
        if (column == 0 && lane == 0 && learn
            && stop_at_structural_barrier && *has_pending
            && -log2f(fmaxf(pending[token], 1.0e-12f))
                >= surprise_birth_bits - 1.0e-5f
            && organism_step - last_birth_step >= birth_cooldown) {{
            *structural_barrier = 1;
        }}
        grid.sync();
        if (*structural_barrier) break;
        int* persistent_current = *persistent_toggle ? persistent_b : persistent_a;
        int* persistent_next = *persistent_toggle ? persistent_a : persistent_b;
        int* persistent_current_count = *persistent_toggle
            ? persistent_count_b : persistent_count_a;
        int* persistent_next_count = *persistent_toggle
            ? persistent_count_a : persistent_count_b;
        if (event_sparse && column == 0 && lane == 0) {{
            for (int depth = 0; depth < max_depth; ++depth) depth_counts[depth] = 0;
            *persistent_next_count = 0;
            *sparse_generation += 1;
            if (*sparse_generation <= 0) *sparse_generation = 1;
        }}
        if (column == 0 && lane == 0) {{
            observed_probabilities[step] = *has_pending
                ? pending[token] : __int_as_float(0x7fffffff);
            base_observed_probabilities[step] = *has_pending
                ? pending_base[token] : __int_as_float(0x7fffffff);
        }}
        grid.sync();
        int generation = *sparse_generation;
        if (event_sparse) {{
            long long sensory_start = consumer_offsets[token];
            long long sensory_stop = consumer_offsets[token + 1];
            for (long long edge = sensory_start + global_thread;
                 edge < sensory_stop; edge += global_threads) {{
                int target = consumer_ids[edge];
                if (atomicExch(scheduled_stamps + target, generation)
                    != generation) {{
                    int depth = consumer_depths[edge] - 1;
                    int position = atomicAdd(depth_counts + depth, 1);
                    depth_queues[depth * max_relations + position] = target;
                }}
            }}
            int persistent_count = *persistent_current_count;
            for (int index = global_thread; index < persistent_count;
                 index += global_threads) {{
                int target = persistent_current[index];
                if (atomicExch(scheduled_stamps + target, generation)
                    != generation) {{
                    int depth = (int)depths[target] - 1;
                    int position = atomicAdd(depth_counts + depth, 1);
                    depth_queues[depth * max_relations + position] = target;
                }}
            }}
        }}
        if (learn && *has_pending) {{
            for (int learning_slot = global_thread;
                 learning_slot < max_relations;
                 learning_slot += global_threads) {{
                int state = pending_local_state[learning_slot];
                if (state >= 0) {{
                    float* row = (state ? state_predictor_counts
                        : predictor_counts) + learning_slot * VOCAB;
                    float total = relation_prior * VOCAB;
                    for (int value = 0; value < VOCAB; ++value) total += row[value];
                    float probability =
                        (row[token] + relation_prior) / total;
                    float gain = log2f(fmaxf(probability, 1.0e-12f) /
                        fmaxf(pending_base[token], 1.0e-12f));
                    if (state) {{
                        state_utility_bits[learning_slot] = utility_decay *
                            state_utility_bits[learning_slot]
                            + (1.0f - utility_decay) * gain;
                        state_observations[learning_slot] += 1;
                    }} else {{
                        utility_bits[learning_slot] = utility_decay *
                            utility_bits[learning_slot]
                            + (1.0f - utility_decay) * gain;
                    }}
                    float decay = lifecycle[learning_slot] == consolidated_lifecycle
                        ? consolidated_predictor_decay : predictor_decay;
                    total = relation_prior * VOCAB;
                    for (int value = 0; value < VOCAB; ++value) {{
                        row[value] *= decay;
                        if (value == token) row[value] += 1.0f;
                        total += row[value];
                    }}
                    float* log_row = (state ? log_state_predictor_probability
                        : log_predictor_probability) + learning_slot * VOCAB;
                    for (int value = 0; value < VOCAB; ++value) {{
                        log_row[value] = logf((row[value] + relation_prior) / total);
                    }}
                }}
            }}
        }}
        grid.sync();
        if (learn && column == 0 && lane < VOCAB) {{
            int old_previous = *previous_token;
            base_counts[lane] *= base_decay;
            if (lane == token) base_counts[lane] += 1.0f;
            if (old_previous >= 0) {{
                int index = old_previous * VOCAB + lane;
                context_counts[index] *= context_decay;
                if (lane == token) context_counts[index] += 1.0f;
            }}
        }}
        grid.sync();
        if (column == 0 && lane < VOCAB) {{
            features[lane] = lane == token ? 1.0f : 0.0f;
        }}
        for (int slot = global_thread; slot < max_relations;
             slot += global_threads) {{
            features[VOCAB + slot] = 0.0f;
            raw[slot] = 0.0f;
            state_probe[slot] = 0.0f;
            candidate_weight[slot] = 0.0f;
            candidate_state[slot] = -1;
            evaluated[slot] = 0;
        }}
        if (!event_sparse) {{
            for (int index = global_thread;
                 index < timescale_count * channel_count;
                 index += global_threads) {{
                int scale = index / channel_count;
                traces[index] *= lambdas[scale];
            }}
        }}
        grid.sync();
        if (column == 0 && lane < timescale_count) {{
            int index = lane * channel_count + token;
            if (event_sparse) {{
                long long gap = causal_step - trace_last_steps[token];
                float old_trace = traces[index];
                float value = old_trace * powf(lambdas[lane], (float)gap)
                    + 1.0f;
                traces[index] = value;
                if (learn) {{
                    float mean = silent_mean_advance(
                        trace_means[index], old_trace, lambdas[lane],
                        trace_scales[lane], trace_mean_decay, gap - 1
                    );
                    trace_means[index] = trace_mean_decay * mean
                        + (1.0f - trace_mean_decay)
                        * fminf(1.0f, trace_scales[lane] * value);
                }}
            }} else {{
                traces[index] += 1.0f;
            }}
        }}
        __syncthreads();
        if (event_sparse && column == 0 && lane == 0) {{
            trace_last_steps[token] = causal_step;
        }}
        grid.sync();

        for (int depth = 1; depth <= max_depth; ++depth) {{
            int work_count = event_sparse
                ? depth_counts[depth - 1] : max_relations;
            for (int work = global_thread; work < work_count;
                 work += global_threads) {{
            int slot = event_sparse
                ? depth_queues[(depth - 1) * max_relations + work] : work;
                int current = (int)current_channels[slot];
                if (live[slot] && depths[slot] == depth) {{
                int trace_channel = (int)trace_channels[slot];
                int scale = (int)timescale_indices[slot];
                float a = features[current];
                int trace_index = scale * channel_count + trace_channel;
                float trace_value = traces[trace_index];
                if (event_sparse) {{
                    long long gap = causal_step - trace_last_steps[trace_channel];
                    trace_value *= powf(lambdas[scale], (float)gap);
                }}
                float b = fminf(
                    1.0f, trace_scales[scale] * trace_value
                );
                float response = 0.0f;
                if (a > support_threshold) {{
                    float drive = w_a[slot] * a + w_b[slot] * b;
                    float delta = drive - mu[slot];
                    response = expf(-(delta * delta) /
                        (2.0f * sigma[slot] * sigma[slot]));
                }}
                raw[slot] = response;
                evaluated[slot] = 1;
                if (learn) atomicAdd(evaluated_total, 1ULL);
                current_values[slot] = a;
                trace_values[slot] = b;
                int event_active = response >= activation_threshold;
                if (learn && event_active
                    && lifecycle[slot] == dormant_lifecycle
                    && utility_bits[slot]
                        >= consolidation_utility_threshold) {{
                    lifecycle[slot] = consolidated_lifecycle;
                    dormant_steps[slot] = -1;
                    atomicAdd(reactivation_total, 1ULL);
                }}
                float retained = persistence[slot] * persistence_decay;
                if (event_active) retained = response;
                persistence[slot] = retained;
                if (!event_active && retained >= activation_threshold) {{
                    state_probe[slot] = retained;
                }}
                float persistent_activation = 0.0f;
                if (state_observations[slot] >= persistence_min_observations &&
                    state_utility_bits[slot] >= persistence_utility_threshold) {{
                    float count = (float)state_observations[slot];
                    persistent_activation = retained * count / (count + 16.0f);
                }}
                float activation = fmaxf(response, persistent_activation);
                features[VOCAB + slot] = activation;
                for (int scale_index = 0; scale_index < timescale_count;
                     ++scale_index) {{
                    int relation_channel = VOCAB + slot;
                    int relation_index =
                        scale_index * channel_count + relation_channel;
                    if (event_sparse) {{
                        long long gap = causal_step
                            - trace_last_steps[relation_channel];
                        float old_trace = traces[relation_index];
                        float value = old_trace * powf(
                            lambdas[scale_index], (float)gap
                        ) + activation;
                        traces[relation_index] = value;
                        if (learn) {{
                            float mean = silent_mean_advance(
                                trace_means[relation_index], old_trace,
                                lambdas[scale_index], trace_scales[scale_index],
                                trace_mean_decay, gap - 1
                            );
                            trace_means[relation_index] =
                                trace_mean_decay * mean
                                + (1.0f - trace_mean_decay) * fminf(
                                    1.0f, trace_scales[scale_index] * value
                                );
                        }}
                    }} else {{
                        traces[relation_index] += activation;
                    }}
                }}
                if (event_sparse) {{
                    trace_last_steps[VOCAB + slot] = causal_step;
                }}
                if (event_sparse) {{
                    int useful_state = !utility_gated_persistence
                        || state_observations[slot] < persistence_min_observations
                        || state_utility_bits[slot]
                            >= persistence_utility_threshold;
                    if (retained > support_threshold && useful_state) {{
                        int position = atomicAdd(persistent_next_count, 1);
                        persistent_next[position] = slot;
                    }}
                    if (activation > support_threshold) {{
                        int channel = VOCAB + slot;
                        long long start = consumer_offsets[channel];
                        long long stop = consumer_offsets[channel + 1];
                        for (long long edge = start; edge < stop; ++edge) {{
                            int target = consumer_ids[edge];
                            if (atomicExch(
                                scheduled_stamps + target, generation
                            ) != generation) {{
                                int target_depth = consumer_depths[edge] - 1;
                                int position = atomicAdd(
                                    depth_counts + target_depth, 1
                                );
                                depth_queues[
                                    target_depth * max_relations + position
                                ] = target;
                            }}
                        }}
                    }}
                }}
                if (learn && event_active) {{
                    fires[slot] += 1;
                    last_fire_steps[slot] = organism_step;
                }}
            }}
            }}
            grid.sync();
        }}
        if (learn && !event_sparse) {{
            for (int index = global_thread;
                 index < timescale_count * channel_count;
                 index += global_threads) {{
                int scale = index / channel_count;
                float normalized = fminf(
                    1.0f, fmaxf(0.0f, trace_scales[scale] * traces[index])
                );
                trace_means[index] = trace_mean_decay * trace_means[index]
                    + (1.0f - trace_mean_decay) * normalized;
            }}
        }}
        if (event_sparse && column == 0 && lane == 0) {{
            *persistent_toggle = 1 - *persistent_toggle;
        }}
        grid.sync();

        int evidence_depths = event_sparse ? max_depth : 1;
        for (int evidence_depth = 0; evidence_depth < evidence_depths;
             ++evidence_depth) {{
        int evidence_count = event_sparse
            ? depth_counts[evidence_depth] : max_relations;
        for (int evidence_work = global_thread;
             evidence_work < evidence_count;
             evidence_work += global_threads) {{
        int slot = event_sparse
            ? depth_queues[evidence_depth * max_relations + evidence_work]
            : evidence_work;
        if (learn && evaluated[slot] && (!event_sparse
            || current_values[slot] > support_threshold) && live[slot]
            && lifecycle[slot] != {DORMANT}) {{
            long long position = local_sample_positions[slot];
            long long sample_index =
                ((long long)slot * local_buffer_size + position) * 2;
            float a = current_values[slot];
            float b = trace_values[slot];
            local_samples[sample_index] = a;
            local_samples[sample_index + 1] = b;
            local_sample_positions[slot] = (position + 1) % local_buffer_size;
            local_sample_counts[slot] = min(
                (long long)local_buffer_size, local_sample_counts[slot] + 1
            );
            observations[slot] += 1;
            if (a > support_threshold) {{
                float response = raw[slot];
                long long n = stats_n[slot] + 1;
                float da = a - stats_mean_a[slot];
                float db = b - stats_mean_b[slot];
                float dd = response - stats_mean_d[slot];
                stats_mean_a[slot] += da / (float)n;
                stats_mean_b[slot] += db / (float)n;
                stats_mean_d[slot] += dd / (float)n;
                stats_m2_a[slot] += da * (a - stats_mean_a[slot]);
                stats_m2_b[slot] += db * (b - stats_mean_b[slot]);
                stats_m2_d[slot] += dd * (response - stats_mean_d[slot]);
                stats_cov_da_acc[slot] +=
                    da * (response - stats_mean_d[slot]);
                stats_cov_db_acc[slot] +=
                    db * (response - stats_mean_d[slot]);
                stats_n[slot] = n;
            }}
            int interval = lifecycle[slot] == consolidated_lifecycle
                ? consolidated_adapt_interval : adapt_interval;
            if (observations[slot] % interval == 0
                && local_sample_counts[slot] >= min_adapt_samples) {{
                adapt_slot_counter(
                    slot, observations[slot], seed, perturb_step,
                    local_samples, local_sample_counts[slot],
                    local_sample_positions[slot], local_buffer_size,
                    w_a, w_b, mu, sigma, minimum_sigma
                );
            }}
        }}
        if (evaluated[slot] && live[slot] &&
            lifecycle[slot] != {DORMANT} && observations[slot] >= 2 &&
            fires[slot] >= 1) {{
            int event_active = raw[slot] >= activation_threshold;
            int state_active = !event_active &&
                state_probe[slot] >= activation_threshold;
            float activation = features[VOCAB + slot];
            if (event_active || state_active) {{
                candidate_state[slot] = state_active;
            }}
            if ((event_active || state_active) &&
                activation >= activation_threshold) {{
                float evidence;
                float utility;
                if (state_active) {{
                    float count = (float)state_observations[slot];
                    evidence = count / (count + 16.0f);
                    utility = state_utility_bits[slot];
                }} else {{
                    float count = (float)observations[slot];
                    evidence = count / (count + 8.0f);
                    utility = utility_bits[slot];
                }}
                float utility_factor = fires[slot] < 8 ? 0.10f :
                    (utility > 0.0f ? fminf(4.0f, 0.25f + 4.0f * utility) : 0.0f);
                candidate_weight[slot] = activation * evidence * utility_factor;
                candidate_state[slot] = state_active;
            }}
        }}
        }}
        }}
        grid.sync();
        if (learn && organism_step % lifecycle_sweep_interval == 0) {{
            for (int slot = global_thread; slot < max_relations;
                 slot += global_threads) {{
                if (!live[slot]) continue;
                if (lifecycle[slot] == juvenile_lifecycle
                    && organism_step - born_steps[slot]
                        >= consolidation_min_age
                    && fires[slot] >= consolidation_min_fires
                    && utility_bits[slot]
                        >= consolidation_utility_threshold) {{
                    lifecycle[slot] = consolidated_lifecycle;
                    consolidated_steps[slot] = organism_step;
                    atomicAdd(consolidation_total, 1ULL);
                }}
                if (lifecycle[slot] == consolidated_lifecycle
                    && (organism_step - last_fire_steps[slot]
                            >= dormancy_staleness
                        || utility_bits[slot] < prune_utility)) {{
                    lifecycle[slot] = dormant_lifecycle;
                    dormant_steps[slot] = organism_step;
                    atomicAdd(dormancy_total, 1ULL);
                }}
            }}
        }}
        grid.sync();
        if (lane == 0) {{
          for (int cortical_column = column; cortical_column < column_count;
               cortical_column += gridDim.x) {{
            int base = cortical_column * max_per_column;
            for (int rank = 0; rank < max_per_column; ++rank) {{
                column_winner_ids[base + rank] = -1;
                column_winner_weights[base + rank] = -1.0f / 0.0f;
            }}
            int first = cortical_column * blockDim.x;
            int stop = min(first + blockDim.x, max_relations);
            for (int candidate = first; candidate < stop; ++candidate) {{
                float weight = candidate_weight[candidate];
                if (weight <= 0.0f) continue;
                for (int rank = 0; rank < max_per_column; ++rank) {{
                    if (weight > column_winner_weights[base + rank]) {{
                        for (int shift = max_per_column - 1; shift > rank; --shift) {{
                            column_winner_ids[base + shift] =
                                column_winner_ids[base + shift - 1];
                            column_winner_weights[base + shift] =
                                column_winner_weights[base + shift - 1];
                        }}
                        column_winner_ids[base + rank] = candidate;
                        column_winner_weights[base + rank] = weight;
                        break;
                    }}
                }}
            }}
            float score = 0.0f;
            for (int rank = 0; rank < max_per_column; ++rank) {{
                if (column_winner_ids[base + rank] >= 0) {{
                    score += column_winner_weights[base + rank];
                }}
            }}
            column_scores[cortical_column] = score;
          }}
        }}
        grid.sync();

        if (column == 0 && lane == 0) {{
            for (int index = 0; index < max_active; ++index) {{
                final_winner_ids[index] = -1;
                final_winner_weights[index] = -1.0f / 0.0f;
            }}
            *final_winner_count = 0;
            // Mark selected columns by negating their positive scores.
            for (int rank = 0; rank < min(max_columns, column_count); ++rank) {{
                int best = -1;
                float best_score = -1.0f / 0.0f;
                for (int candidate = 0; candidate < column_count; ++candidate) {{
                    float score = column_scores[candidate];
                    if (score >= 0.0f && score > best_score) {{
                        best = candidate;
                        best_score = score;
                    }}
                }}
                if (best < 0 || best_score <= 0.0f) break;
                column_scores[best] = -best_score;
            }}
            for (int active_column = 0; active_column < column_count;
                 ++active_column) {{
                if (column_scores[active_column] >= 0.0f) continue;
                int base = active_column * max_per_column;
                for (int local = 0; local < max_per_column; ++local) {{
                    int winner = column_winner_ids[base + local];
                    if (winner < 0) continue;
                    float weight = column_winner_weights[base + local];
                    int count = *final_winner_count;
                    if (count >= max_active &&
                        weight <= final_winner_weights[max_active - 1]) continue;
                    int insertion = count < max_active ? count : max_active - 1;
                    while (insertion > 0 &&
                        weight > final_winner_weights[insertion - 1]) {{
                        final_winner_ids[insertion] = final_winner_ids[insertion - 1];
                        final_winner_state[insertion] =
                            final_winner_state[insertion - 1];
                        final_winner_weights[insertion] =
                            final_winner_weights[insertion - 1];
                        --insertion;
                    }}
                    final_winner_ids[insertion] = winner;
                    final_winner_state[insertion] = candidate_state[winner];
                    final_winner_weights[insertion] = weight;
                    if (count < max_active) *final_winner_count = count + 1;
                }}
            }}
            *previous_token = token;
        }}
        grid.sync();

        if (column == 0) {{
            float context_value = context_counts[token * VOCAB + lane]
                + context_prior;
            work[lane] = context_value;
            __syncthreads();
            for (int stride = 128; stride > 0; stride >>= 1) {{
                if (lane < stride) work[lane] += work[lane + stride];
                __syncthreads();
            }}
            float context_total = work[0];
            log_context_probability[token * VOCAB + lane] =
                logf(context_value / context_total);
            pending_base[lane] = context_value / context_total;
            __syncthreads();
            float total_weight = 1.0f;
            float logit = log_context_probability[token * VOCAB + lane];
            for (int index = 0; index < *final_winner_count; ++index) {{
                int winner = final_winner_ids[index];
                const float* distribution = final_winner_state[index]
                    ? log_state_predictor_probability + winner * VOCAB
                    : log_predictor_probability + winner * VOCAB;
                logit += final_winner_weights[index] * distribution[lane];
                total_weight += final_winner_weights[index];
            }}
            logit /= total_weight;
            work[lane] = logit;
            __syncthreads();
            for (int stride = 128; stride > 0; stride >>= 1) {{
                if (lane < stride) work[lane] = fmaxf(work[lane], work[lane + stride]);
                __syncthreads();
            }}
            float probability = expf(logit - work[0]);
            work[lane] = probability;
            __syncthreads();
            for (int stride = 128; stride > 0; stride >>= 1) {{
                if (lane < stride) work[lane] += work[lane + stride];
                __syncthreads();
            }}
            pending[lane] = probability / work[0];
            if (lane == 0) *has_pending = 1;
        }}
        grid.sync();
        if (learn) {{
            for (int learning_slot = global_thread;
                 learning_slot < max_relations;
                 learning_slot += global_threads) {{
                pending_local_state[learning_slot] =
                    candidate_state[learning_slot];
            }}
        }}
        grid.sync();
        if (column == 0 && lane == 0) *processed_steps = step + 1;
        grid.sync();
    }}
}}

extern "C" __global__ void materialize_lazy_traces(
    float* traces, float* trace_means, const long long* trace_last_steps,
    const float* lambdas, const float* trace_scales,
    int timescale_count, int channel_count, long long current_step,
    float trace_mean_decay, int learn
) {{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int element_count = timescale_count * channel_count;
    if (index >= element_count) return;
    int timescale = index / channel_count;
    int channel = index - timescale * channel_count;
    long long silent_steps = current_step - trace_last_steps[channel];
    if (silent_steps <= 0) return;
    float lambda = lambdas[timescale];
    float trace = traces[index];
    if (learn) {{
        trace_means[index] = silent_mean_advance(
            trace_means[index], trace, lambda, trace_scales[timescale],
            trace_mean_decay, silent_steps
        );
    }}
    traces[index] = trace * powf(lambda, (float)silent_steps);
}}
"""


class CudaMultiColumnFrozenEvaluator:
    """Persistent cooperative-grid evaluator for contextual ALM columns."""

    def __init__(
        self,
        learner: OnlineDendriticStreamLearner,
        *,
        resume_dynamic: bool = False,
    ) -> None:
        c = learner.config
        if c.vocab_size != 256 or not c.contextual_residual:
            raise ValueError("multi-column CUDA requires byte contextual residuals")
        if c.column_size != 256:
            raise ValueError("multi-column CUDA currently requires 256-slot columns")
        if c.max_active_relations > _MAX_ACTIVE:
            raise ValueError(f"max_active_relations must be <= {_MAX_ACTIVE}")
        if c.max_active_per_column > _MAX_PER_COLUMN:
            raise ValueError(f"max_active_per_column must be <= {_MAX_PER_COLUMN}")
        cp = _cupy()
        self.cp, self.config = cp, c
        self.column_count = learner.column_count
        self.local_winners_per_column = (
            c.max_active_relations
            if self.column_count == 1
            else c.max_active_per_column
        )
        multiprocessors = cp.cuda.Device(0).attributes["MultiProcessorCount"]
        # The kernel currently uses 64 registers/thread. Two resident blocks
        # per SM leave a conservative cooperative-launch margin while blocks
        # stride over additional columns at ALM-1GB scale.
        self.grid_blocks = min(self.column_count, 2 * multiprocessors)
        module = cp.RawModule(
            code=_SOURCE,
            options=("-std=c++17",),
            enable_cooperative_groups=True,
            name_expressions=(
                "frozen_multicolumn_chunk",
                "materialize_lazy_traces",
            ),
        )
        self.kernel = module.get_function("frozen_multicolumn_chunk")
        self.materialize_kernel = module.get_function("materialize_lazy_traces")
        self.live = cp.asarray(learner._live, dtype=cp.uint8)
        self.depths = cp.asarray(learner._depths, dtype=cp.int64)
        axonal_index = AxonalConsumerIndex.from_learner(learner)
        self.consumer_offsets = cp.asarray(axonal_index.offsets)
        self.consumer_ids = cp.asarray(axonal_index.slot_ids)
        self.consumer_depths = cp.asarray(axonal_index.depths)
        for name, dtype in (
            ("current_channels", cp.int64),
            ("trace_channels", cp.int64),
            ("timescale_indices", cp.int64),
            ("lifecycle", cp.int8),
            ("observations", cp.int64),
            ("fires", cp.int64),
            ("born_steps", cp.int64),
            ("last_fire_steps", cp.int64),
            ("consolidated_steps", cp.int64),
            ("dormant_steps", cp.int64),
            ("utility_bits", cp.float32),
            ("state_observations", cp.int64),
            ("state_utility_bits", cp.float32),
            ("w_a", cp.float32),
            ("w_b", cp.float32),
            ("mu", cp.float32),
            ("sigma", cp.float32),
        ):
            source = getattr(learner, f"_{name}")
            setattr(self, name, cp.asarray(source, dtype=dtype))
        self.predictor = cp.asarray(learner._predictor_counts, dtype=cp.float32)
        self.state_predictor = cp.asarray(
            learner._state_predictor_counts, dtype=cp.float32
        )
        prior = np.float32(c.relation_prior)
        self.log_predictor = cp.log(
            (self.predictor + prior)
            / (self.predictor.sum(axis=1, keepdims=True) + 256 * prior)
        )
        self.log_state_predictor = cp.log(
            (self.state_predictor + prior)
            / (self.state_predictor.sum(axis=1, keepdims=True) + 256 * prior)
        )
        self.base_counts = cp.asarray(learner.base_counts, dtype=cp.float32)
        self.context_counts = cp.asarray(learner.context_counts, dtype=cp.float32)
        context_probability = self.context_counts + np.float32(c.context_prior)
        self.log_context = cp.log(
            context_probability / context_probability.sum(axis=1, keepdims=True)
        )
        self.lambdas = cp.asarray(learner.lambdas, dtype=cp.float32)
        self.trace_scales = cp.asarray(learner.trace_scales, dtype=cp.float32)
        self.local_samples = cp.asarray(learner._local_samples, dtype=cp.float32)
        self.local_sample_counts = cp.asarray(
            learner._local_sample_counts, dtype=cp.int64
        )
        self.local_sample_positions = cp.asarray(
            learner._local_sample_positions, dtype=cp.int64
        )
        for name, dtype in (
            ("stats_n", cp.int64),
            ("stats_mean_a", cp.float32),
            ("stats_mean_b", cp.float32),
            ("stats_mean_d", cp.float32),
            ("stats_m2_a", cp.float32),
            ("stats_m2_b", cp.float32),
            ("stats_m2_d", cp.float32),
            ("stats_cov_da_acc", cp.float32),
            ("stats_cov_db_acc", cp.float32),
        ):
            setattr(
                self,
                name,
                cp.asarray(getattr(learner, f"_{name}"), dtype=dtype),
            )
        self._learned_steps = 0
        self._learned_scored = 0
        self._synced_steps = 0
        self._synced_scored = 0
        self._synced_evaluated = 0
        self._synced_consolidations = 0
        self._synced_dormancies = 0
        self._synced_reactivations = 0
        self._synced_fires = learner._fires.copy()
        self._host_has_pending = False
        self._causal_steps = 0
        self._episode_learn_mode: bool | None = None
        self.evaluated_total = cp.zeros(1, dtype=cp.uint64)
        self.consolidation_total = cp.zeros(1, dtype=cp.uint64)
        self.dormancy_total = cp.zeros(1, dtype=cp.uint64)
        self.reactivation_total = cp.zeros(1, dtype=cp.uint64)
        self._allocate_dynamic()
        self._organism_steps = learner.stats.steps
        if resume_dynamic:
            self._resume_dynamic_from(learner)

    def _allocate_dynamic(self) -> None:
        cp, c = self.cp, self.config
        channels = 256 + c.max_relations
        self.traces = cp.zeros((len(c.timescales), channels), dtype=cp.float32)
        self.trace_means = cp.full(
            (len(c.timescales), channels), 0.05, dtype=cp.float32
        )
        self.trace_last_steps = cp.full(channels, -1, dtype=cp.int64)
        self.persistence = cp.zeros(c.max_relations, dtype=cp.float32)
        self.features = cp.zeros(channels, dtype=cp.float32)
        self.raw = cp.zeros(c.max_relations, dtype=cp.float32)
        self.state_probe = cp.zeros(c.max_relations, dtype=cp.float32)
        self.current_values = cp.zeros(c.max_relations, dtype=cp.float32)
        self.trace_values = cp.zeros(c.max_relations, dtype=cp.float32)
        self.evaluated = cp.zeros(c.max_relations, dtype=cp.uint8)
        self.scheduled_stamps = cp.zeros(c.max_relations, dtype=cp.int32)
        self.depth_queues = cp.empty((c.max_depth, c.max_relations), dtype=cp.int32)
        self.depth_counts = cp.zeros(c.max_depth, dtype=cp.int32)
        self.persistent_a = cp.empty(c.max_relations, dtype=cp.int32)
        self.persistent_b = cp.empty(c.max_relations, dtype=cp.int32)
        self.persistent_count_a = cp.zeros(1, dtype=cp.int32)
        self.persistent_count_b = cp.zeros(1, dtype=cp.int32)
        self.persistent_toggle = cp.zeros(1, dtype=cp.int32)
        self.sparse_generation = cp.zeros(1, dtype=cp.int32)
        self.candidate_weight = cp.zeros(c.max_relations, dtype=cp.float32)
        self.candidate_state = cp.zeros(c.max_relations, dtype=cp.int32)
        self.pending_local_state = cp.full(c.max_relations, -1, dtype=cp.int32)
        shape = (self.column_count, self.local_winners_per_column)
        self.column_winner_ids = cp.full(shape, -1, dtype=cp.int32)
        self.column_winner_weights = cp.zeros(shape, dtype=cp.float32)
        self.column_scores = cp.zeros(self.column_count, dtype=cp.float32)
        self.final_winner_ids = cp.full(c.max_active_relations, -1, dtype=cp.int32)
        self.final_winner_state = cp.zeros(c.max_active_relations, dtype=cp.int32)
        self.final_winner_weights = cp.zeros(c.max_active_relations, dtype=cp.float32)
        self.final_winner_count = cp.zeros(1, dtype=cp.int32)
        self.pending = cp.zeros(256, dtype=cp.float32)
        self.pending_base = cp.zeros(256, dtype=cp.float32)
        self.has_pending = cp.zeros(1, dtype=cp.int32)
        self.previous_token = cp.full(1, -1, dtype=cp.int32)
        self.processed_steps = cp.zeros(1, dtype=cp.int32)
        self.structural_barrier = cp.zeros(1, dtype=cp.int32)

    def _resume_dynamic_from(self, learner: OnlineDendriticStreamLearner) -> None:
        cp, c = self.cp, self.config
        self.traces[:] = cp.asarray(learner.traces, dtype=cp.float32)
        self.trace_means[:] = cp.asarray(learner.trace_means, dtype=cp.float32)
        self.persistence[:] = cp.asarray(learner._persistence, dtype=cp.float32)
        self.previous_token[0] = learner.previous_token
        pending = learner._pending
        if pending is not None:
            self.pending[:] = cp.asarray(pending.distribution, dtype=cp.float32)
            self.pending_base[:] = cp.asarray(
                pending.base_distribution, dtype=cp.float32
            )
            self.features[:] = cp.asarray(pending.features, dtype=cp.float32)
            self.pending_local_state.fill(-1)
            if len(pending.relation_ids):
                self.pending_local_state[pending.relation_ids] = cp.asarray(
                    pending.relation_is_state, dtype=cp.int32
                )
            winner_count = min(len(pending.active_relations), c.max_active_relations)
            if winner_count:
                self.final_winner_ids[:winner_count] = cp.asarray(
                    pending.active_relations[:winner_count], dtype=cp.int32
                )
            self.final_winner_count[0] = winner_count
            self.has_pending[0] = 1
            self._host_has_pending = True
        if c.metabolic_backend == "event":
            useful = np.ones(c.max_relations, dtype=np.bool_)
            if c.utility_gated_persistence:
                useful = np.logical_or(
                    learner._state_observations < c.persistence_min_observations,
                    learner._state_utility_bits >= c.persistence_utility_threshold,
                )
            persistent_ids = np.flatnonzero(
                (learner._persistence > c.support_threshold) & useful
            )
            if len(persistent_ids):
                self.persistent_a[: len(persistent_ids)] = cp.asarray(
                    persistent_ids, dtype=cp.int32
                )
            self.persistent_count_a[0] = len(persistent_ids)

    def reset_episode(self) -> None:
        self._allocate_dynamic()
        self._host_has_pending = False
        self._causal_steps = 0
        self._episode_learn_mode = None

    def _observe_chunk_probabilities(
        self,
        tokens: np.ndarray,
        *,
        learn: bool,
        structural_barriers: bool,
        last_birth_step: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        cp, c = self.cp, self.config
        token_array = np.ascontiguousarray(tokens, dtype=np.uint8).reshape(-1)
        if len(token_array) and c.metabolic_backend == "event":
            if self._episode_learn_mode is None:
                self._episode_learn_mode = learn
            elif self._episode_learn_mode != learn:
                raise ValueError(
                    "event-sparse lazy traces require one learning mode per episode"
                )
        had_pending = self._host_has_pending
        observed = cp.empty(len(token_array), dtype=cp.float32)
        base_observed = cp.empty(len(token_array), dtype=cp.float32)
        self.processed_steps.fill(0)
        self.structural_barrier.fill(0)
        args: tuple[Any, ...] = (
            cp.asarray(token_array),
            np.int32(len(token_array)),
            self.live,
            self.depths,
            self.current_channels,
            self.trace_channels,
            self.timescale_indices,
            self.lifecycle,
            self.consumer_offsets,
            self.consumer_ids,
            self.consumer_depths,
            self.observations,
            self.fires,
            self.born_steps,
            self.last_fire_steps,
            self.consolidated_steps,
            self.dormant_steps,
            self.utility_bits,
            self.state_observations,
            self.state_utility_bits,
            self.base_counts,
            self.context_counts,
            self.predictor,
            self.state_predictor,
            self.log_context,
            self.log_predictor,
            self.log_state_predictor,
            self.lambdas,
            self.trace_scales,
            self.traces,
            self.trace_means,
            self.trace_last_steps,
            self.persistence,
            self.w_a,
            self.w_b,
            self.mu,
            self.sigma,
            self.features,
            self.raw,
            self.state_probe,
            self.current_values,
            self.trace_values,
            self.evaluated,
            self.evaluated_total,
            self.consolidation_total,
            self.dormancy_total,
            self.reactivation_total,
            self.scheduled_stamps,
            self.depth_queues,
            self.depth_counts,
            self.persistent_a,
            self.persistent_b,
            self.persistent_count_a,
            self.persistent_count_b,
            self.persistent_toggle,
            self.sparse_generation,
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
            self.candidate_weight,
            self.candidate_state,
            self.column_winner_ids,
            self.column_winner_weights,
            self.column_scores,
            self.final_winner_ids,
            self.final_winner_state,
            self.final_winner_weights,
            self.final_winner_count,
            self.pending,
            self.has_pending,
            self.previous_token,
            self.pending_local_state,
            self.pending_base,
            observed,
            base_observed,
            self.processed_steps,
            self.structural_barrier,
            np.int32(len(c.timescales)),
            np.int32(c.max_relations),
            np.int32(self.column_count),
            np.int32(c.max_depth),
            np.int32(c.local_buffer_size),
            np.float32(c.support_threshold),
            np.float32(c.activation_threshold),
            np.float32(c.persistence_decay),
            np.int64(c.persistence_min_observations),
            np.float32(c.persistence_utility_threshold),
            np.float32(c.relation_prior),
            np.float32(c.base_decay),
            np.float32(c.context_decay),
            np.float32(c.predictor_decay),
            np.float32(c.consolidated_predictor_decay),
            np.float32(c.utility_decay),
            np.float32(c.trace_mean_decay),
            np.float32(c.context_prior),
            np.int32(CONSOLIDATED),
            np.int32(self.local_winners_per_column),
            np.int32(c.max_active_columns),
            np.int32(c.max_active_relations),
            np.int32(c.metabolic_backend == "event"),
            np.int32(c.utility_gated_persistence),
            np.int64(self._causal_steps),
            np.int32(learn),
            np.int32(structural_barriers),
            np.float32(c.surprise_birth_bits),
            np.int64(c.birth_cooldown),
            np.int64(self._organism_steps),
            np.int64(last_birth_step),
            np.int32(JUVENILE),
            np.int32(DORMANT),
            np.int32(c.adapt_interval),
            np.int32(c.consolidated_adapt_interval),
            np.int64(c.min_adapt_samples),
            np.int64(c.seed),
            np.float32(c.perturb_step),
            np.float32(c.initial_sigma / 4.0),
            np.int32(c.lifecycle_sweep_interval),
            np.int64(c.consolidation_min_age),
            np.int64(c.consolidation_min_fires),
            np.float32(c.consolidation_utility_threshold),
            np.int64(c.dormancy_staleness),
            np.float32(c.prune_utility),
        )
        self.kernel((self.grid_blocks,), (256,), args)
        processed = int(self.processed_steps.get()[0])
        barrier = bool(self.structural_barrier.get()[0])
        if not structural_barriers and (barrier or processed != len(token_array)):
            raise RuntimeError("fixed-anatomy CUDA loop stopped early")
        result = observed[:processed].get()
        base_result = base_observed[:processed].get()
        if processed:
            self._host_has_pending = True
            self._causal_steps += processed
            if learn:
                self._organism_steps += processed
                self._learned_steps += processed
                self._learned_scored += processed - int(not had_pending)
        return result, base_result, barrier

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
        """Learn until a surprise-eligible byte needs host anatomy."""
        return self._observe_chunk_probabilities(
            tokens,
            learn=True,
            structural_barriers=True,
            last_birth_step=last_birth_step,
        )

    def _materialize_event_traces(self) -> None:
        if self.config.metabolic_backend != "event" or self._causal_steps == 0:
            return
        element_count = self.traces.size
        threads = 256
        self.materialize_kernel(
            ((element_count + threads - 1) // threads,),
            (threads,),
            (
                self.traces,
                self.trace_means,
                self.trace_last_steps,
                self.lambdas,
                self.trace_scales,
                np.int32(len(self.config.timescales)),
                np.int32(256 + self.config.max_relations),
                np.int64(self._causal_steps - 1),
                np.float32(self.config.trace_mean_decay),
                np.int32(bool(self._episode_learn_mode)),
            ),
        )
        self.trace_last_steps.fill(self._causal_steps - 1)

    def synchronize_plasticity_to(self, learner: OnlineDendriticStreamLearner) -> None:
        """Copy resident synaptic and transient state back to its CPU organism.

        This synchronizes predictor plasticity, dendritic evidence, and causal
        activity, materializing event-sparse traces at the episode frontier.
        """
        if learner.config.max_relations != self.config.max_relations:
            raise ValueError("CPU and GPU organisms have different anatomy")
        self._materialize_event_traces()
        learner.base_counts[:] = self.base_counts.get()
        learner.context_counts[:] = self.context_counts.get()
        learner._predictor_counts[:] = self.predictor.get()
        learner._state_predictor_counts[:] = self.state_predictor.get()
        learner._utility_bits[:] = self.utility_bits.get()
        learner._state_utility_bits[:] = self.state_utility_bits.get()
        learner._state_observations[:] = self.state_observations.get()
        learner._observations[:] = self.observations.get()
        learner._last_fire_steps[:] = self.last_fire_steps.get()
        learner._lifecycle[:] = self.lifecycle.get()
        learner._consolidated_steps[:] = self.consolidated_steps.get()
        learner._dormant_steps[:] = self.dormant_steps.get()
        learner._w_a[:] = self.w_a.get()
        learner._w_b[:] = self.w_b.get()
        learner._mu[:] = self.mu.get()
        learner._sigma[:] = self.sigma.get()
        learner._local_samples[:] = self.local_samples.get()
        learner._local_sample_counts[:] = self.local_sample_counts.get()
        learner._local_sample_positions[:] = self.local_sample_positions.get()
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
            getattr(learner, f"_{name}")[:] = getattr(self, name).get()
        device_fires = self.fires.get()
        learner._fires[:] = device_fires
        learner.traces[:] = self.traces.get()
        learner.trace_means[:] = self.trace_means.get()
        learner._persistence[:] = self.persistence.get()
        learner.previous_token = int(self.previous_token.get()[0])

        local_state = self.pending_local_state.get()
        relation_ids = np.flatnonzero(local_state >= 0)
        winner_count = int(self.final_winner_count.get()[0])
        winner_ids = self.final_winner_ids.get()[:winner_count]
        normalized = np.clip(learner.trace_scales[:, None] * learner.traces, 0.0, 1.0)
        learner._pending = _PendingPrediction(
            distribution=self.pending.get().astype(np.float64),
            base_distribution=self.pending_base.get().astype(np.float64),
            relation_ids=relation_ids.astype(np.intp),
            relation_is_state=(local_state[relation_ids] != 0),
            active_relations=tuple(int(slot) for slot in winner_ids),
            features=self.features.get().astype(np.float64),
            traces=normalized,
        )
        learner._materialize_slot_views()
        for raw_slot_id in np.flatnonzero(learner._live):
            slot_id = int(raw_slot_id)
            slot = learner.slots[slot_id]
            assert slot is not None
            slot.predictor_counts = learner._predictor_counts[slot_id]
            slot.state_predictor_counts = learner._state_predictor_counts[slot_id]
            slot.observations = int(learner._observations[slot_id])
            slot.fires = int(learner._fires[slot_id])
            slot.last_fire_step = int(learner._last_fire_steps[slot_id])
            slot.persistence = float(learner._persistence[slot_id])
            slot.utility_bits = float(learner._utility_bits[slot_id])
            slot.state_utility_bits = float(learner._state_utility_bits[slot_id])
            slot.state_observations = int(learner._state_observations[slot_id])
            slot.lifecycle = LIFECYCLE_NAMES[int(learner._lifecycle[slot_id])]
            slot.consolidated_step = int(learner._consolidated_steps[slot_id])
            slot.dormant_step = int(learner._dormant_steps[slot_id])
        step_delta = self._learned_steps - self._synced_steps
        scored_delta = self._learned_scored - self._synced_scored
        learner.stats.steps += step_delta
        learner.stats.predictions_scored += scored_delta
        evaluated = int(self.evaluated_total.get()[0])
        learner.stats.branch_evaluations += evaluated - self._synced_evaluated
        learner.stats.relation_firings += int(np.sum(device_fires - self._synced_fires))
        consolidations = int(self.consolidation_total.get()[0])
        dormancies = int(self.dormancy_total.get()[0])
        reactivations = int(self.reactivation_total.get()[0])
        learner.stats.consolidations += consolidations - self._synced_consolidations
        learner.stats.dormancies += dormancies - self._synced_dormancies
        learner.stats.reactivations += reactivations - self._synced_reactivations
        self._synced_steps = self._learned_steps
        self._synced_scored = self._learned_scored
        self._synced_evaluated = evaluated
        self._synced_consolidations = consolidations
        self._synced_dormancies = dormancies
        self._synced_reactivations = reactivations
        self._synced_fires = device_fires.copy()


class CudaDevelopmentalRunner:
    """Run metabolism on CUDA and wake the host for anatomical change."""

    def __init__(
        self,
        learner: OnlineDendriticStreamLearner,
        *,
        structural_wake_bytes: int = 1,
    ) -> None:
        if structural_wake_bytes < 1:
            raise ValueError("structural_wake_bytes must be positive")
        self.learner = learner
        self.structural_wake_bytes = structural_wake_bytes
        self.structural_barriers = 0
        self.host_wake_bytes = 0
        self.anatomy_rebuilds = 0
        self.resident = CudaMultiColumnFrozenEvaluator(
            learner, resume_dynamic=learner._pending is not None
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

            # Surprise is the structural interrupt. Materialize the organism,
            # let the canonical host nursery/birth/prune logic handle a short
            # wake window, then rebuild the resident anatomy once per window.
            self.structural_barriers += 1
            self.synchronize()
            wake_stop = min(len(token_array), offset + self.structural_wake_bytes)
            host_observed, host_base = self.learner.observe_chunk_probabilities(
                token_array[offset:wake_stop], learn=True
            )
            self.host_wake_bytes += wake_stop - offset
            parts.append(host_observed.astype(np.float32, copy=False))
            base_parts.append(host_base.astype(np.float32, copy=False))
            offset = wake_stop
            self.resident = CudaMultiColumnFrozenEvaluator(
                self.learner, resume_dynamic=True
            )
            self.anatomy_rebuilds += 1

        if not parts:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty.copy()
        return np.concatenate(parts), np.concatenate(base_parts)

    def synchronize(self) -> None:
        self.resident.synchronize_plasticity_to(self.learner)


def observe_cuda_developmental_probabilities(
    learner: OnlineDendriticStreamLearner,
    tokens: np.ndarray,
    *,
    structural_wake_bytes: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one CUDA developmental stream and return synchronized state."""
    runner = CudaDevelopmentalRunner(
        learner, structural_wake_bytes=structural_wake_bytes
    )
    observed, base_observed = runner.observe_chunk_probabilities(tokens)
    runner.synchronize()
    return observed, base_observed
