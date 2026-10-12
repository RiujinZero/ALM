"""Measure mixed-precision drift on a learned contextual-residual checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticConfig,
    OnlineDendriticStreamLearner,
)
from manifoldflow.experiments.dendritic_basin_gpu_precision_probe import (
    _module,
    _roundtrip,
)
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
    iter_byte_stream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--train-offset", type=int, default=100_000)
    parser.add_argument("--continuation-bytes", type=int, default=20_000)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _fp32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def _native_round(
    cp: Any, module: Any, values: np.ndarray, format_name: str
) -> np.ndarray:
    shape = values.shape
    rounded = _roundtrip(
        cp,
        module,
        np.asarray(values, dtype=np.float32).reshape(-1),
        format_name,
    )
    return rounded.reshape(shape).astype(np.float64)


def _scaled_rows(
    cp: Any,
    module: Any,
    values: np.ndarray,
    format_name: str,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if source.size == 0:
        return source.astype(np.float64)
    rows = source.reshape(-1, source.shape[-1])
    maximum = 448.0 if format_name == "fp8_e4m3" else 57_344.0
    scales = np.maximum(np.max(np.abs(rows), axis=1, keepdims=True) / maximum, 1e-12)
    rounded = _native_round(cp, module, rows / scales, format_name)
    return (rounded * scales).reshape(source.shape)


def _quantized_state(
    cp: Any,
    module: Any,
    source: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    state = copy.deepcopy(source)

    def counts(values: np.ndarray) -> np.ndarray:
        if policy == "fp32":
            return _fp32(values)
        if policy == "bf16":
            return _native_round(cp, module, values, "bf16")
        return _scaled_rows(cp, module, values, policy)

    state["base_counts"] = counts(state["base_counts"])
    state["context_counts"] = counts(state["context_counts"])
    for slot in state["slots"]:
        if slot is None:
            continue
        slot["predictor_counts"] = counts(slot["predictor_counts"])
        slot["state_predictor_counts"] = counts(slot["state_predictor_counts"])
        # Learning masters and structural thresholds remain FP32. BF16/FP8
        # are inference mirrors for count distributions, not accumulators.
        dendrite = slot["dendrite"]
        for key in ("w_a", "w_b", "mu", "sigma"):
            dendrite[key] = float(np.float32(dendrite[key]))
    for candidate in state["nursery"]:
        candidate["target_counts"] = counts(candidate["target_counts"])
    return state


def _learner_from_state(state: dict[str, Any]) -> OnlineDendriticStreamLearner:
    config_state = dict(state["config"])
    config_state["timescales"] = tuple(config_state["timescales"])
    model = OnlineDendriticStreamLearner(OnlineDendriticConfig(**config_state))
    model.load_state_dict(state)
    return model


def _score_stream(
    learner: OnlineDendriticStreamLearner,
    path: str,
    byte_count: int,
    *,
    offset: int = 0,
    learn: bool,
) -> float:
    bit_sum = 0.0
    scored = 0
    buffer = bytearray()
    for token in iter_byte_stream(path, byte_count, offset, chunk_size=1 << 20):
        buffer.append(token)
        if len(buffer) < 8192:
            continue
        probabilities, _ = learner.observe_chunk_probabilities(
            np.frombuffer(buffer, dtype=np.uint8), learn=learn
        )
        valid = probabilities[np.isfinite(probabilities)]
        bit_sum += float(-np.log2(np.maximum(valid, 1e-12)).sum())
        scored += len(valid)
        buffer.clear()
    if buffer:
        probabilities, _ = learner.observe_chunk_probabilities(
            np.frombuffer(buffer, dtype=np.uint8), learn=learn
        )
        valid = probabilities[np.isfinite(probabilities)]
        bit_sum += float(-np.log2(np.maximum(valid, 1e-12)).sum())
        scored += len(valid)
    return bit_sum / max(scored, 1)


def main() -> None:
    args = parse_args()
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - hardware environment
        raise SystemExit("install the 'alm-gpu' extra to run this probe") from exc

    module = _module(cp)
    source_learner = OnlineDendriticStreamLearner.load_checkpoint(args.checkpoint)
    source = source_learner.state_dict()
    policies = ("fp32", "bf16", "fp8_e4m3", "fp8_e5m2")
    results = []
    for policy in ("fp64", *policies):
        state = (
            source if policy == "fp64" else _quantized_state(cp, module, source, policy)
        )
        evaluator = _learner_from_state(state)
        evaluator.reset_episode()
        eval_bpb = _score_stream(
            evaluator,
            args.valid_path,
            args.eval_bytes,
            learn=False,
        )
        continuation = _learner_from_state(state)
        before_births = continuation.stats.births
        before_prunes = continuation.stats.prunes
        continuation_bpb = _score_stream(
            continuation,
            args.train_path,
            args.continuation_bytes,
            offset=args.train_offset,
            learn=True,
        )
        itemsize = {
            "fp64": 8,
            "fp32": 4,
            "bf16": 2,
            "fp8_e4m3": 1,
            "fp8_e5m2": 1,
        }[policy]
        scale_bytes = 0
        if policy.startswith("fp8"):
            scale_bytes = 4 * (
                source_learner.config.vocab_size
                + 2 * source_learner.config.max_relations
            )
        predictor_values = (
            source_learner.context_counts.size
            + source_learner._predictor_counts.size
            + source_learner._state_predictor_counts.size
        )
        results.append(
            {
                "policy": policy,
                "eval_bpb": eval_bpb,
                "continuation_bpb": continuation_bpb,
                "continuation_births": continuation.stats.births - before_births,
                "continuation_prunes": continuation.stats.prunes - before_prunes,
                "predictor_storage_bytes": predictor_values * itemsize + scale_bytes,
            }
        )

    reference = results[0]
    for result in results:
        result["eval_bpb_delta"] = result["eval_bpb"] - reference["eval_bpb"]
        result["continuation_bpb_delta"] = (
            result["continuation_bpb"] - reference["continuation_bpb"]
        )
        result["structural_counts_match"] = (
            result["continuation_births"] == reference["continuation_births"]
            and result["continuation_prunes"] == reference["continuation_prunes"]
        )
    payload = {
        "checkpoint": str(args.checkpoint),
        "eval_bytes": args.eval_bytes,
        "continuation_bytes": args.continuation_bytes,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
