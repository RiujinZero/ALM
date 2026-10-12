"""Validate and benchmark frozen contextual ALM inference on CUDA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.gpu_online_stream import (
    CudaFrozenDendriticEvaluator,
)
from manifoldflow.dendritic_basin.online_stream import OnlineDendriticStreamLearner
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_VALID_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--sequence-bytes", type=int, default=4096)
    parser.add_argument("--batch-sequences", type=int, default=512)
    parser.add_argument("--oracle-sequences", type=int, default=3)
    parser.add_argument(
        "--activity-precisions", nargs="+", choices=("fp32", "bf16"),
        default=("fp32", "bf16"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _timed_cpu(
    checkpoint: Path, tokens: np.ndarray
) -> tuple[np.ndarray, float]:
    learner = OnlineDendriticStreamLearner.load_checkpoint(checkpoint)
    learner.reset_episode()
    learner.observe_chunk_probabilities(tokens[:64], learn=False)
    learner.reset_episode()
    started = time.perf_counter()
    probabilities, _ = learner.observe_chunk_probabilities(tokens, learn=False)
    return probabilities, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    total_bytes = args.sequence_bytes * args.batch_sequences
    with args.valid_path.open("rb") as stream:
        flat = np.frombuffer(stream.read(total_bytes), dtype=np.uint8).copy()
    if len(flat) != total_bytes:
        raise SystemExit(f"validation stream has only {len(flat):,} requested bytes")
    tokens = flat.reshape(args.batch_sequences, args.sequence_bytes)

    source = OnlineDendriticStreamLearner.load_checkpoint(args.checkpoint)
    oracle_count = min(args.oracle_sequences, args.batch_sequences)
    cpu_seconds = 0.0
    cpu_probabilities = []
    for sequence in range(oracle_count):
        cpu_probability, elapsed = _timed_cpu(args.checkpoint, tokens[sequence])
        cpu_seconds += elapsed
        cpu_probabilities.append(cpu_probability)

    cpu_rate = oracle_count * args.sequence_bytes / cpu_seconds
    results = []
    for precision in args.activity_precisions:
        gpu = CudaFrozenDendriticEvaluator(
            source, activity_precision=precision
        )
        gpu.observe_batch_probabilities(tokens[:1, :64])
        started = time.perf_counter()
        gpu_probabilities = gpu.observe_batch_probabilities(tokens)
        gpu_seconds = time.perf_counter() - started
        maximum_error = 0.0
        error_sum = 0.0
        compared = 0
        for sequence, cpu_probability in enumerate(cpu_probabilities):
            valid = np.isfinite(cpu_probability)
            error = np.abs(
                cpu_probability[valid] - gpu_probabilities[sequence, valid]
            )
            maximum_error = max(maximum_error, float(error.max(initial=0.0)))
            error_sum += float(error.sum())
            compared += len(error)
        gpu_rate = total_bytes / gpu_seconds
        valid_gpu = gpu_probabilities[:, 1:]
        results.append(
            {
                "activity_precision": precision,
                "gpu_seconds": gpu_seconds,
                "gpu_aggregate_bytes_per_second": gpu_rate,
                "gpu_aggregate_speedup": gpu_rate / cpu_rate,
                "probability_max_abs_error": maximum_error,
                "probability_mean_abs_error": error_sum / max(compared, 1),
                "gpu_bits_per_byte": float(
                    -np.log2(np.maximum(valid_gpu, 1e-12)).mean()
                ),
            }
        )
    payload = {
        "checkpoint": str(args.checkpoint),
        "sequence_bytes": args.sequence_bytes,
        "batch_sequences": args.batch_sequences,
        "total_bytes": total_bytes,
        "live_relations": source.live_relation_count,
        "max_relations": source.config.max_relations,
        "cpu_oracle_sequences": oracle_count,
        "cpu_seconds": cpu_seconds,
        "cpu_bytes_per_second": cpu_rate,
        "results": results,
        "semantics": "independent reset episodes sharing frozen learned tissue",
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
