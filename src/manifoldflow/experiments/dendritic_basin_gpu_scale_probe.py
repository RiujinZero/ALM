"""Dry-run cooperative CUDA anatomy at large byte-class ALM geometries."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.gpu_multicolumn_stream import (
    CudaMultiColumnFrozenEvaluator,
)
from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticConfig,
    OnlineDendriticStreamLearner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--byte-classes", nargs="+", type=int,
        default=(64_000_000, 256_000_000, 1_000_000_000),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - hardware environment
        raise SystemExit("install the 'alm-gpu' extra to run this probe") from exc

    rows = []
    for byte_class in args.byte_classes:
        pool_before = cp.get_default_memory_pool().used_bytes()
        config = OnlineDendriticConfig(
            vocab_size=256,
            max_alm_bytes=byte_class,
            column_size=256,
            contextual_residual=True,
        )
        host_started = time.perf_counter()
        learner = OnlineDendriticStreamLearner(config)
        host_seconds = time.perf_counter() - host_started
        gpu_started = time.perf_counter()
        evaluator = CudaMultiColumnFrozenEvaluator(learner)
        gpu_load_seconds = time.perf_counter() - gpu_started
        run_started = time.perf_counter()
        probability = evaluator.observe_chunk_probabilities(
            np.asarray((65, 66), dtype=np.uint8)
        )
        run_seconds = time.perf_counter() - run_started
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        active_gpu_bytes = cp.get_default_memory_pool().used_bytes() - pool_before
        rows.append(
            {
                "alm_class_bytes": byte_class,
                "estimated_max_serialized_bytes": (
                    config.estimated_max_serialized_bytes
                ),
                "relations": config.max_relations,
                "columns": evaluator.column_count,
                "cooperative_blocks": evaluator.grid_blocks,
                "host_allocate_seconds": host_seconds,
                "gpu_load_seconds": gpu_load_seconds,
                "two_byte_seconds": run_seconds,
                "successor_probability": float(probability[1]),
                "gpu_used_bytes": total_bytes - free_bytes,
                "active_evaluator_gpu_bytes": active_gpu_bytes,
            }
        )
        del evaluator, learner
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    payload = {
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "rows": rows,
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
