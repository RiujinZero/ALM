"""Benchmark resident fixed-anatomy CPU columns at ALM byte geometries."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.cpu_column_stream import (
    CpuColumnFrozenEvaluator,
)
from manifoldflow.dendritic_basin.online_stream import OnlineDendriticStreamLearner
from manifoldflow.experiments.dendritic_basin_gpu_filled_scale_probe import (
    _tile_learned_body,
)
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_TRAIN_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--byte-classes",
        nargs="+",
        type=int,
        default=(64_000_000, 1_000_000_000),
    )
    parser.add_argument("--workers", nargs="+", type=int, default=(1, 8, 16, 30))
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--sample-bytes", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--learn", action="store_true", help="benchmark resident local plasticity"
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_bytes <= 0 or args.repeats <= 0:
        raise ValueError("sample bytes and repeats must be positive")
    source = OnlineDendriticStreamLearner.load_checkpoint(args.checkpoint)
    with args.sample_path.open("rb") as stream:
        tokens = np.frombuffer(
            stream.read(args.sample_bytes), dtype=np.uint8
        ).copy()
    if len(tokens) != args.sample_bytes:
        raise SystemExit(f"sample stream has only {len(tokens):,} requested bytes")

    rows = []
    for byte_class in args.byte_classes:
        learner = _tile_learned_body(source, byte_class)
        for requested_workers in args.workers:
            evaluator = CpuColumnFrozenEvaluator(
                learner, workers=requested_workers
            )
            evaluator.observe_chunk_probabilities(tokens[:16], learn=args.learn)
            rates = []
            for _repeat in range(args.repeats):
                evaluator.reset_episode()
                started = time.perf_counter()
                evaluator.observe_chunk_probabilities(tokens, learn=args.learn)
                seconds = time.perf_counter() - started
                rates.append(len(tokens) / seconds)
            rows.append(
                {
                    "alm_class_bytes": byte_class,
                    "relations": learner.config.max_relations,
                    "requested_workers": requested_workers,
                    "actual_workers": evaluator.workers,
                    "rates_bytes_per_second": rates,
                    "median_bytes_per_second": statistics.median(rates),
                    "best_bytes_per_second": max(rates),
                }
            )
            del evaluator
            gc.collect()
        del learner
        gc.collect()

    payload = {
        "checkpoint": str(args.checkpoint),
        "sample_path": str(args.sample_path),
        "sample_bytes": len(tokens),
        "repeats": args.repeats,
        "source_live_relations": source.live_relation_count,
        "semantics": (
            "fixed anatomy with resident local plasticity"
            if args.learn
            else "frozen learned motif tiled into cortical columns"
        ),
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
