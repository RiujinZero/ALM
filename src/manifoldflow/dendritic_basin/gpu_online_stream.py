"""CUDA-resident frozen inference for a learned dense dendritic organism.

This first backend deliberately supports a single cortical column (up to 256
relations).  One persistent CUDA block advances an entire byte chunk, keeping
eligibility traces, persistence, and the pending distribution on device.  It
is an inference oracle/backend, not a second implementation of plasticity.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.online_stream import (
    DORMANT,
    OnlineDendriticStreamLearner,
)

_MAX_RELATIONS = 256
_MAX_ACTIVE = 32

_CUDA_SOURCE = rf"""
#ifdef ALM_USE_BF16
#include <cuda_bf16.h>
typedef unsigned short trace_storage_t;
__device__ __forceinline__ float trace_load(
    const trace_storage_t* values, int index
) {{
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(values)[index]
    );
}}
__device__ __forceinline__ void trace_store(
    trace_storage_t* values, int index, float value
) {{
    reinterpret_cast<__nv_bfloat16*>(values)[index] = __float2bfloat16(value);
}}
#else
typedef float trace_storage_t;
__device__ __forceinline__ float trace_load(
    const trace_storage_t* values, int index
) {{
    return values[index];
}}
__device__ __forceinline__ void trace_store(
    trace_storage_t* values, int index, float value
) {{
    values[index] = value;
}}
#endif

#define VOCAB 256
#define MAX_REL {_MAX_RELATIONS}
#define MAX_ACTIVE {_MAX_ACTIVE}

extern "C" __global__ void frozen_contextual_chunk(
    const unsigned char* tokens, int steps,
    const long long* live_ids, int live_count,
    const long long* current_channels,
    const long long* trace_channels,
    const long long* timescale_indices,
    const signed char* lifecycle,
    const long long* observations,
    const long long* fires,
    const float* utility_bits,
    const long long* state_observations,
    const float* state_utility_bits,
    const float* predictor_counts,
    const float* state_predictor_counts,
    const float* predictor_totals,
    const float* state_predictor_totals,
    const float* base_counts,
    const float* context_counts,
    const float* log_base_probability,
    const float* log_context_probability,
    const float* log_predictor_probability,
    const float* log_state_predictor_probability,
    const float* lambdas,
    const float* trace_scales,
    trace_storage_t* traces,
    trace_storage_t* persistence,
    const float* w_a,
    const float* w_b,
    const float* mu,
    const float* sigma,
    float* pending,
    int* has_pending,
    int* previous_token,
    float* observed_probabilities,
    int timescale_count,
    int max_relations,
    float support_threshold,
    float activation_threshold,
    float persistence_decay,
    long long persistence_min_observations,
    float persistence_utility_threshold,
    float relation_prior,
    float base_prior,
    float context_prior,
    int contextual_residual,
    int max_active
) {{
    const int lane = threadIdx.x;
    const int channel_count = VOCAB + max_relations;
    const int sequence = blockIdx.x;
    tokens += sequence * steps;
    traces += sequence * timescale_count * channel_count;
    persistence += sequence * max_relations;
    pending += sequence * VOCAB;
    has_pending += sequence;
    previous_token += sequence;
    observed_probabilities += sequence * steps;
    __shared__ float features[VOCAB + MAX_REL];
    __shared__ float raw[MAX_REL];
    __shared__ float state_probe[MAX_REL];
    __shared__ float candidate_weight[MAX_REL];
    __shared__ int candidate_state[MAX_REL];
    __shared__ int winner_ids[MAX_ACTIVE];
    __shared__ int winner_state[MAX_ACTIVE];
    __shared__ float winner_weights[MAX_ACTIVE];
    __shared__ float work[VOCAB];
    __shared__ float basal_total;
    __shared__ float total_weight;
    __shared__ int winner_count;

    for (int step = 0; step < steps; ++step) {{
        const int token = (int)tokens[step];
        if (lane == 0) {{
            observed_probabilities[step] = *has_pending
                ? pending[token] : __int_as_float(0x7fffffff);
        }}
        if (lane < VOCAB) features[lane] = lane == token ? 1.0f : 0.0f;
        for (int slot = lane; slot < max_relations; slot += blockDim.x) {{
            features[VOCAB + slot] = 0.0f;
            raw[slot] = 0.0f;
            state_probe[slot] = 0.0f;
            candidate_weight[slot] = 0.0f;
            candidate_state[slot] = 0;
        }}
        for (int index = lane; index < timescale_count * channel_count;
             index += blockDim.x) {{
            int scale = index / channel_count;
            trace_store(
                traces, index, trace_load(traces, index) * lambdas[scale]
            );
        }}
        __syncthreads();
        if (lane < timescale_count) {{
            int trace_index = lane * channel_count + token;
            trace_store(
                traces, trace_index, trace_load(traces, trace_index) + 1.0f
            );
        }}
        __syncthreads();

        // Birth order is a causal topological order: relation parents always
        // have lower slot IDs.  A single lane preserves that exact ordering;
        // the 256 lanes parallelize trace decay and the predictor softmax.
        if (lane == 0) {{
            for (int position = 0; position < live_count; ++position) {{
                int slot = (int)live_ids[position];
                int current = (int)current_channels[slot];
                int trace_channel = (int)trace_channels[slot];
                int scale = (int)timescale_indices[slot];
                float a = features[current];
                float b = fminf(
                    1.0f,
                    trace_scales[scale] * trace_load(
                        traces, scale * channel_count + trace_channel
                    )
                );
                float response = 0.0f;
                if (a > support_threshold) {{
                    float drive = w_a[slot] * a + w_b[slot] * b;
                    float width = sigma[slot];
                    float delta = drive - mu[slot];
                    response = expf(-(delta * delta) / (2.0f * width * width));
                }}
                raw[slot] = response;
                int event_active = response >= activation_threshold;
                float retained = trace_load(persistence, slot) * persistence_decay;
                if (event_active) retained = response;
                trace_store(persistence, slot, retained);
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
                for (int trace_index = 0; trace_index < timescale_count;
                     ++trace_index) {{
                    int relation_trace =
                        trace_index * channel_count + VOCAB + slot;
                    trace_store(
                        traces,
                        relation_trace,
                        trace_load(traces, relation_trace) + activation
                    );
                }}
            }}

            winner_count = 0;
            total_weight = 1.0f;
            for (int position = 0; position < live_count; ++position) {{
                int slot = (int)live_ids[position];
                if (lifecycle[slot] == {DORMANT} || observations[slot] < 2 ||
                    fires[slot] < 1) continue;
                int event_active = raw[slot] >= activation_threshold;
                int state_active = !event_active &&
                    state_probe[slot] >= activation_threshold;
                if (!event_active && !state_active) continue;
                float activation = features[VOCAB + slot];
                if (activation < activation_threshold) continue;
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
                float weight = activation * evidence * utility_factor;
                if (weight <= 0.0f) continue;
                int insertion = winner_count < max_active
                    ? winner_count : max_active - 1;
                if (winner_count >= max_active &&
                    weight <= winner_weights[max_active - 1]) continue;
                while (insertion > 0 && weight > winner_weights[insertion - 1]) {{
                    winner_ids[insertion] = winner_ids[insertion - 1];
                    winner_state[insertion] = winner_state[insertion - 1];
                    winner_weights[insertion] = winner_weights[insertion - 1];
                    --insertion;
                }}
                winner_ids[insertion] = slot;
                winner_state[insertion] = state_active;
                winner_weights[insertion] = weight;
                if (winner_count < max_active) ++winner_count;
            }}
            for (int index = 0; index < winner_count; ++index) {{
                total_weight += winner_weights[index];
            }}
            if (!contextual_residual) {{
                basal_total = base_prior * VOCAB;
                for (int value = 0; value < VOCAB; ++value) {{
                    basal_total += base_counts[value];
                }}
            }}
            *previous_token = token;
        }}
        __syncthreads();

        float probability = contextual_residual ? 0.0f :
            (base_counts[lane] + base_prior) / basal_total;
        if (contextual_residual) {{
            float logit = log_context_probability[token * VOCAB + lane];
            for (int index = 0; index < winner_count; ++index) {{
                int slot = winner_ids[index];
                const float* log_distribution = winner_state[index]
                    ? log_state_predictor_probability + slot * VOCAB
                    : log_predictor_probability + slot * VOCAB;
                logit += winner_weights[index] *
                    log_distribution[lane];
            }}
            work[lane] = logit / total_weight;
            __syncthreads();
            for (int stride = 128; stride > 0; stride >>= 1) {{
                if (lane < stride) work[lane] = fmaxf(work[lane], work[lane + stride]);
                __syncthreads();
            }}
            float maximum = work[0];
            probability = expf(logit / total_weight - maximum);
            work[lane] = probability;
            __syncthreads();
            for (int stride = 128; stride > 0; stride >>= 1) {{
                if (lane < stride) work[lane] += work[lane + stride];
                __syncthreads();
            }}
            probability /= work[0];
        }} else {{
            for (int index = 0; index < winner_count; ++index) {{
                int slot = winner_ids[index];
                const float* counts = winner_state[index]
                    ? state_predictor_counts + slot * VOCAB
                    : predictor_counts + slot * VOCAB;
                float count_total = winner_state[index]
                    ? state_predictor_totals[slot] : predictor_totals[slot];
                probability += winner_weights[index] *
                    (counts[lane] + relation_prior) / count_total;
            }}
            probability /= total_weight;
        }}
        pending[lane] = probability;
        __syncthreads();
        if (lane == 0) *has_pending = 1;
        __syncthreads();
    }}
}}
"""


def _cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - optional hardware dependency
        raise RuntimeError("install the 'alm-gpu' extra for CUDA inference") from exc
    return cp


class CudaFrozenDendriticEvaluator:
    """Device-resident causal evaluator for a frozen, one-column ALM."""

    def __init__(
        self,
        learner: OnlineDendriticStreamLearner,
        *,
        activity_precision: str = "fp32",
    ) -> None:
        config = learner.config
        if config.vocab_size != 256:
            raise ValueError("CUDA frozen evaluator currently requires byte vocabulary")
        if config.max_relations > _MAX_RELATIONS or learner.column_count != 1:
            raise ValueError(
                "CUDA frozen evaluator currently supports one <=256-slot column"
            )
        if config.max_active_relations > _MAX_ACTIVE:
            raise ValueError(f"max_active_relations must be <= {_MAX_ACTIVE}")
        if config.metabolic_backend != "dense":
            raise ValueError(
                "CUDA frozen evaluator currently requires dense metabolism"
            )
        if activity_precision not in {"fp32", "bf16"}:
            raise ValueError("activity_precision must be 'fp32' or 'bf16'")

        cp = _cupy()
        self.cp = cp
        self.config = config
        self.activity_precision = activity_precision
        module_options = ["-std=c++17"]
        if activity_precision == "bf16":
            module_options.append("-DALM_USE_BF16")
        self._module = cp.RawModule(
            code=_CUDA_SOURCE,
            options=tuple(module_options),
            name_expressions=("frozen_contextual_chunk",),
        )
        self._kernel = self._module.get_function("frozen_contextual_chunk")
        self.live_ids = cp.asarray(np.flatnonzero(learner._live), dtype=cp.int64)
        self.current_channels = cp.asarray(learner._current_channels, dtype=cp.int64)
        self.trace_channels = cp.asarray(learner._trace_channels, dtype=cp.int64)
        self.timescale_indices = cp.asarray(learner._timescale_indices, dtype=cp.int64)
        self.lifecycle = cp.asarray(learner._lifecycle, dtype=cp.int8)
        self.observations = cp.asarray(learner._observations, dtype=cp.int64)
        self.fires = cp.asarray(learner._fires, dtype=cp.int64)
        self.utility_bits = cp.asarray(learner._utility_bits, dtype=cp.float32)
        self.state_observations = cp.asarray(
            learner._state_observations, dtype=cp.int64
        )
        self.state_utility_bits = cp.asarray(
            learner._state_utility_bits, dtype=cp.float32
        )
        self.predictor_counts = cp.asarray(learner._predictor_counts, dtype=cp.float32)
        self.state_predictor_counts = cp.asarray(
            learner._state_predictor_counts, dtype=cp.float32
        )
        prior_total = config.relation_prior * config.vocab_size
        self.predictor_totals = self.predictor_counts.sum(axis=1) + prior_total
        self.state_predictor_totals = (
            self.state_predictor_counts.sum(axis=1) + prior_total
        )
        self.base_counts = cp.asarray(learner.base_counts, dtype=cp.float32)
        self.context_counts = cp.asarray(learner.context_counts, dtype=cp.float32)
        base_probability = self.base_counts + np.float32(config.base_prior)
        base_probability /= base_probability.sum()
        context_probability = self.context_counts + np.float32(config.context_prior)
        context_probability /= context_probability.sum(axis=1, keepdims=True)
        predictor_probability = self.predictor_counts + np.float32(
            config.relation_prior
        )
        predictor_probability /= self.predictor_totals[:, None]
        state_predictor_probability = self.state_predictor_counts + np.float32(
            config.relation_prior
        )
        state_predictor_probability /= self.state_predictor_totals[:, None]
        self.log_base_probability = cp.log(base_probability)
        self.log_context_probability = cp.log(context_probability)
        self.log_predictor_probability = cp.log(predictor_probability)
        self.log_state_predictor_probability = cp.log(state_predictor_probability)
        self.lambdas = cp.asarray(learner.lambdas, dtype=cp.float32)
        self.trace_scales = cp.asarray(learner.trace_scales, dtype=cp.float32)
        self.w_a = cp.asarray(learner._w_a, dtype=cp.float32)
        self.w_b = cp.asarray(learner._w_b, dtype=cp.float32)
        self.mu = cp.asarray(learner._mu, dtype=cp.float32)
        self.sigma = cp.asarray(learner._sigma, dtype=cp.float32)
        self.reset_episode()

    def reset_episode(self) -> None:
        """Clear only transient cortical state, retaining frozen learned tissue."""
        cp = self.cp
        activity_dtype = cp.uint16 if self.activity_precision == "bf16" else cp.float32
        self.traces = cp.zeros(
            (len(self.config.timescales), 256 + self.config.max_relations),
            dtype=activity_dtype,
        )
        self.persistence = cp.zeros(
            self.config.max_relations, dtype=activity_dtype
        )
        self.pending = cp.zeros(256, dtype=cp.float32)
        self.has_pending = cp.zeros(1, dtype=cp.int32)
        self.previous_token = cp.full(1, -1, dtype=cp.int32)

    def observe_chunk_probabilities(self, tokens: np.ndarray) -> np.ndarray:
        """Return each byte's causal probability while retaining device state."""
        cp = self.cp
        token_array = np.ascontiguousarray(tokens, dtype=np.uint8).reshape(-1)
        device_tokens = cp.asarray(token_array)
        observed = cp.empty(len(token_array), dtype=cp.float32)
        self._launch(
            device_tokens,
            observed,
            self.traces,
            self.persistence,
            self.pending,
            self.has_pending,
            self.previous_token,
            1,
            len(token_array),
        )
        return observed.get()

    def observe_batch_probabilities(self, tokens: np.ndarray) -> np.ndarray:
        """Score independent, reset episodes concurrently, one CUDA block each."""
        cp = self.cp
        token_array = np.ascontiguousarray(tokens, dtype=np.uint8)
        if token_array.ndim != 2:
            raise ValueError("batch tokens must have shape (sequences, steps)")
        sequence_count, step_count = token_array.shape
        if sequence_count == 0 or step_count == 0:
            return np.empty(token_array.shape, dtype=np.float32)
        channel_count = 256 + self.config.max_relations
        activity_dtype = cp.uint16 if self.activity_precision == "bf16" else cp.float32
        traces = cp.zeros(
            (sequence_count, len(self.config.timescales), channel_count),
            dtype=activity_dtype,
        )
        persistence = cp.zeros(
            (sequence_count, self.config.max_relations), dtype=activity_dtype
        )
        pending = cp.zeros((sequence_count, 256), dtype=cp.float32)
        has_pending = cp.zeros(sequence_count, dtype=cp.int32)
        previous_token = cp.full(sequence_count, -1, dtype=cp.int32)
        observed = cp.empty(token_array.shape, dtype=cp.float32)
        self._launch(
            cp.asarray(token_array),
            observed,
            traces,
            persistence,
            pending,
            has_pending,
            previous_token,
            sequence_count,
            step_count,
        )
        return observed.get()

    def _launch(
        self,
        device_tokens: Any,
        observed: Any,
        traces: Any,
        persistence: Any,
        pending: Any,
        has_pending: Any,
        previous_token: Any,
        sequence_count: int,
        step_count: int,
    ) -> None:
        """Launch the persistent causal kernel over independent episodes."""
        c = self.config
        self._kernel(
            (sequence_count,),
            (256,),
            (
                device_tokens,
                np.int32(step_count),
                self.live_ids,
                np.int32(len(self.live_ids)),
                self.current_channels,
                self.trace_channels,
                self.timescale_indices,
                self.lifecycle,
                self.observations,
                self.fires,
                self.utility_bits,
                self.state_observations,
                self.state_utility_bits,
                self.predictor_counts,
                self.state_predictor_counts,
                self.predictor_totals,
                self.state_predictor_totals,
                self.base_counts,
                self.context_counts,
                self.log_base_probability,
                self.log_context_probability,
                self.log_predictor_probability,
                self.log_state_predictor_probability,
                self.lambdas,
                self.trace_scales,
                traces,
                persistence,
                self.w_a,
                self.w_b,
                self.mu,
                self.sigma,
                pending,
                has_pending,
                previous_token,
                observed,
                np.int32(len(c.timescales)),
                np.int32(c.max_relations),
                np.float32(c.support_threshold),
                np.float32(c.activation_threshold),
                np.float32(c.persistence_decay),
                np.int64(c.persistence_min_observations),
                np.float32(c.persistence_utility_threshold),
                np.float32(c.relation_prior),
                np.float32(c.base_prior),
                np.float32(c.context_prior),
                np.int32(c.contextual_residual),
                np.int32(c.max_active_relations),
            ),
        )
