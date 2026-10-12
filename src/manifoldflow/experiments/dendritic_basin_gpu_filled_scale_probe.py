"""Benchmark a learned dendritic motif tiled to large ALM byte classes."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.gpu_multicolumn_stream import (
    CudaMultiColumnFrozenEvaluator,
)
from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticStreamLearner,
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
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--sample-bytes", type=int, default=8192)
    parser.add_argument(
        "--modes", nargs="+", choices=("inference", "learning"),
        default=("inference", "learning"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _tile_learned_body(
    source: OnlineDendriticStreamLearner, byte_class: int
) -> OnlineDendriticStreamLearner:
    """Copy a complete learned motif into independent cortical columns."""
    live_ids = np.flatnonzero(source._live)
    if not len(live_ids):
        raise ValueError("source checkpoint has no live dendritic relations")
    rank = np.full(source.config.max_relations, -1, dtype=np.int64)
    rank[live_ids] = np.arange(len(live_ids))
    config = replace(
        source.config,
        max_alm_bytes=byte_class,
        metabolic_backend="event",
        utility_gated_persistence=True,
        surprise_birth_bits=1e9,
        adapt_interval=1_000_000_000,
        consolidated_adapt_interval=1_000_000_000,
    )
    learner = OnlineDendriticStreamLearner(config)
    target_ids = np.arange(config.max_relations)
    source_ids = live_ids[target_ids % len(live_ids)]
    groups = target_ids // len(live_ids)
    for name in (
        "depths",
        "timescale_indices",
        "lifecycle",
        "observations",
        "fires",
        "utility_bits",
        "state_observations",
        "state_utility_bits",
        "w_a",
        "w_b",
        "mu",
        "sigma",
    ):
        getattr(learner, f"_{name}")[:] = getattr(source, f"_{name}")[source_ids]
    for name in ("current_channels", "trace_channels"):
        channels = getattr(source, f"_{name}")[source_ids].copy()
        recursive = channels >= source.config.vocab_size
        parent_ranks = rank[channels[recursive] - source.config.vocab_size]
        if np.any(parent_ranks < 0):
            raise ValueError("live motif refers to a non-live parent relation")
        channels[recursive] = (
            source.config.vocab_size
            + groups[recursive] * len(live_ids)
            + parent_ranks
        )
        getattr(learner, f"_{name}")[:] = channels
    learner._live[:] = True
    learner._predictor_counts[:] = source._predictor_counts[source_ids]
    learner._state_predictor_counts[:] = source._state_predictor_counts[source_ids]
    learner.base_counts[:] = source.base_counts
    learner.context_counts[:] = source.context_counts
    return learner


def _measure(
    cp: Any,
    learner: OnlineDendriticStreamLearner,
    tokens: np.ndarray,
    *,
    learn: bool,
) -> dict[str, Any]:
    pool_before = cp.get_default_memory_pool().used_bytes()
    evaluator = CudaMultiColumnFrozenEvaluator(learner)
    evaluator.observe_chunk_probabilities(tokens[:64], learn=learn)
    evaluator.reset_episode()
    evaluator.evaluated_total.fill(0)
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    evaluator.observe_chunk_probabilities(tokens, learn=learn)
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - started
    evaluated = int(evaluator.evaluated_total.get()[0]) if learn else None
    result = {
        "mode": "learning" if learn else "inference",
        "seconds": seconds,
        "bytes_per_second": len(tokens) / seconds,
        "active_evaluator_gpu_bytes": (
            cp.get_default_memory_pool().used_bytes() - pool_before
        ),
    }
    if evaluated is not None:
        result["evaluated_relations_per_byte"] = evaluated / len(tokens)
        result["evaluated_relation_fraction"] = (
            evaluated / (len(tokens) * learner.config.max_relations)
        )
    del evaluator
    return result


def main() -> None:
    args = parse_args()
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - hardware environment
        raise SystemExit("install the 'alm-gpu' extra to run this probe") from exc

    source = OnlineDendriticStreamLearner.load_checkpoint(args.checkpoint)
    with args.sample_path.open("rb") as stream:
        tokens = np.frombuffer(stream.read(args.sample_bytes), dtype=np.uint8).copy()
    if len(tokens) != args.sample_bytes:
        raise SystemExit(f"sample stream has only {len(tokens):,} requested bytes")

    rows = []
    for byte_class in args.byte_classes:
        learner = _tile_learned_body(source, byte_class)
        measurements = [
            _measure(cp, learner, tokens, learn=mode == "learning")
            for mode in args.modes
        ]
        rows.append(
            {
                "alm_class_bytes": byte_class,
                "relations": learner.config.max_relations,
                "columns": learner.column_count,
                "measurements": measurements,
            }
        )
        del learner
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    payload = {
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "checkpoint": str(args.checkpoint),
        "sample_path": str(args.sample_path),
        "sample_bytes": len(tokens),
        "source_live_relations": source.live_relation_count,
        "semantics": "learned motif tiled into independent cortical columns",
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
