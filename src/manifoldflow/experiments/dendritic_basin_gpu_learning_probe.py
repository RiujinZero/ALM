"""Validate FP32 GPU predictor plasticity on learned ALM tissue."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.online_stream import (
    CONSOLIDATED,
    OnlineDendriticStreamLearner,
)

CUDA_SOURCE = r"""
extern "C" __global__ void learn_local_rows(
    const int* slot_ids, const unsigned char* is_state, int relation_count,
    const unsigned char* tokens, int steps, const float* base_probabilities,
    const unsigned char* lifecycle, float* predictor_counts,
    float* state_predictor_counts, float* utility_bits,
    float* state_utility_bits, long long* state_observations,
    float relation_prior, float utility_decay, float predictor_decay,
    float consolidated_predictor_decay, int consolidated_lifecycle
) {
    int relation = blockIdx.x;
    int token_index = threadIdx.x;
    if (relation >= relation_count || token_index >= 256) return;
    int slot = slot_ids[relation];
    bool state = is_state[relation] != 0;
    float* row = (state ? state_predictor_counts : predictor_counts) + slot * 256;
    __shared__ float reduction[256];
    __shared__ float decay;
    for (int step = 0; step < steps; ++step) {
        int token = (int)tokens[step];
        reduction[token_index] = row[token_index];
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (token_index < stride) {
                reduction[token_index] += reduction[token_index + stride];
            }
            __syncthreads();
        }
        if (token_index == 0) {
            float probability = (row[token] + relation_prior) /
                (reduction[0] + relation_prior * 256.0f);
            float safe_base = fmaxf(base_probabilities[step], 1.0e-12f);
            float gain = log2f(fmaxf(probability, 1.0e-12f) / safe_base);
            if (state) {
                state_utility_bits[slot] = utility_decay *
                    state_utility_bits[slot] + (1.0f - utility_decay) * gain;
                state_observations[slot] += 1;
            } else {
                utility_bits[slot] = utility_decay * utility_bits[slot] +
                    (1.0f - utility_decay) * gain;
            }
            decay = lifecycle[slot] == consolidated_lifecycle
                ? consolidated_predictor_decay : predictor_decay;
        }
        __syncthreads();
        row[token_index] *= decay;
        __syncthreads();
        if (token_index == token) row[token_index] += 1.0f;
        __syncthreads();
    }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--relations", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _cpu_update(
    learner: OnlineDendriticStreamLearner,
    ids: np.ndarray,
    is_state: np.ndarray,
    tokens: np.ndarray,
    base: np.ndarray,
) -> None:
    c = learner.config
    for token, base_probability in zip(tokens, base, strict=True):
        for slot, state in zip(ids, is_state, strict=True):
            row = (
                learner._state_predictor_counts[slot]
                if state else learner._predictor_counts[slot]
            )
            probability = (row[token] + c.relation_prior) / (
                row.sum() + c.relation_prior * 256
            )
            gain = np.log2(max(probability, 1e-12) / max(base_probability, 1e-12))
            if state:
                learner._state_utility_bits[slot] = (
                    c.utility_decay * learner._state_utility_bits[slot]
                    + (1.0 - c.utility_decay) * gain
                )
                learner._state_observations[slot] += 1
            else:
                learner._utility_bits[slot] = (
                    c.utility_decay * learner._utility_bits[slot]
                    + (1.0 - c.utility_decay) * gain
                )
            decay = (
                c.consolidated_predictor_decay
                if learner._lifecycle[slot] == CONSOLIDATED
                else c.predictor_decay
            )
            row *= decay
            row[token] += 1.0


def main() -> None:
    args = parse_args()
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - hardware environment
        raise SystemExit("install the 'alm-gpu' extra to run this probe") from exc

    source = OnlineDendriticStreamLearner.load_checkpoint(args.checkpoint)
    live = np.flatnonzero(source._live)
    ids = live[: min(args.relations, len(live))].astype(np.int32)
    if not len(ids):
        raise SystemExit("checkpoint has no live relations")
    rng = np.random.default_rng(args.seed)
    is_state = rng.integers(0, 2, len(ids), dtype=np.uint8)
    tokens = rng.integers(0, 256, args.steps, dtype=np.uint8)
    base = rng.dirichlet(np.ones(256), args.steps)[
        np.arange(args.steps), tokens
    ].astype(np.float32)

    cpu = OnlineDendriticStreamLearner.load_checkpoint(args.checkpoint)
    started = time.perf_counter()
    _cpu_update(cpu, ids, is_state, tokens, base)
    cpu_seconds = time.perf_counter() - started

    predictor = cp.asarray(source._predictor_counts, dtype=cp.float32)
    state_predictor = cp.asarray(source._state_predictor_counts, dtype=cp.float32)
    utility = cp.asarray(source._utility_bits, dtype=cp.float32)
    state_utility = cp.asarray(source._state_utility_bits, dtype=cp.float32)
    state_observations = cp.asarray(source._state_observations, dtype=cp.int64)
    lifecycle = cp.asarray(source._lifecycle, dtype=cp.uint8)
    kernel = cp.RawKernel(CUDA_SOURCE, "learn_local_rows", options=("-std=c++17",))
    device_ids = cp.asarray(ids)
    device_is_state = cp.asarray(is_state)
    device_tokens = cp.asarray(tokens)
    device_base = cp.asarray(base)
    started = time.perf_counter()
    kernel(
        (len(ids),), (256,),
        (
            device_ids, device_is_state, np.int32(len(ids)),
            device_tokens, np.int32(len(tokens)), device_base, lifecycle,
            predictor, state_predictor, utility, state_utility, state_observations,
            np.float32(source.config.relation_prior),
            np.float32(source.config.utility_decay),
            np.float32(source.config.predictor_decay),
            np.float32(source.config.consolidated_predictor_decay),
            np.int32(CONSOLIDATED),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    gpu_seconds = time.perf_counter() - started

    event_ids = ids[is_state == 0]
    state_ids = ids[is_state != 0]
    errors: dict[str, float] = {}
    if len(event_ids):
        errors["event_predictor_counts"] = float(
            np.max(
                np.abs(
                    predictor.get()[event_ids]
                    - cpu._predictor_counts[event_ids]
                )
            )
        )
        errors["event_utility_bits"] = float(
            np.max(
                np.abs(utility.get()[event_ids] - cpu._utility_bits[event_ids])
            )
        )
    if len(state_ids):
        errors["state_predictor_counts"] = float(
            np.max(
                np.abs(
                    state_predictor.get()[state_ids]
                    - cpu._state_predictor_counts[state_ids]
                )
            )
        )
        errors["state_utility_bits"] = float(
            np.max(
                np.abs(
                    state_utility.get()[state_ids]
                    - cpu._state_utility_bits[state_ids]
                )
            )
        )
        errors["state_observations"] = float(
            np.max(
                np.abs(
                    state_observations.get()[state_ids]
                    - cpu._state_observations[state_ids]
                )
            )
        )
    payload = {
        "checkpoint": str(args.checkpoint), "relations": len(ids),
        "steps": args.steps, "branch_outcomes": len(ids) * args.steps,
        "cpu_seconds": cpu_seconds, "gpu_seconds": gpu_seconds,
        "gpu_branch_outcomes_per_second": len(ids) * args.steps / gpu_seconds,
        "maximum_state_error": max(errors.values(), default=0.0),
        "errors": errors,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
