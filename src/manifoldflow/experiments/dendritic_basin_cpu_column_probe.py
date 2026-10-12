"""Measure CPU-worker locality for filled cortical-column ALM geometries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.column_partition import (
    partition_cortical_columns,
)
from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticConfig,
    OnlineDendriticStreamLearner,
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
    parser.add_argument(
        "--workers", nargs="+", type=int, default=(1, 8, 16, 30)
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("contiguous", "affinity"),
        default=("contiguous", "affinity"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _tiled_channels(
    source: OnlineDendriticStreamLearner, relation_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_ids = np.arange(relation_count) % source.config.max_relations
    groups = np.arange(relation_count) // source.config.max_relations
    tiled = []
    for name in ("_current_channels", "_trace_channels"):
        channels = getattr(source, name)[source_ids].copy()
        recursive = channels >= source.config.vocab_size
        channels[recursive] = (
            source.config.vocab_size
            + groups[recursive] * source.config.max_relations
            + channels[recursive]
            - source.config.vocab_size
        )
        tiled.append(channels)
    return tiled[0], tiled[1], np.ones(relation_count, dtype=np.bool_)


def main() -> None:
    args = parse_args()
    source = OnlineDendriticStreamLearner.load_checkpoint(args.checkpoint)
    if source.live_relation_count != source.config.max_relations:
        raise SystemExit("column probe currently requires a fully live source motif")
    rows = []
    for byte_class in args.byte_classes:
        config = OnlineDendriticConfig(
            vocab_size=source.config.vocab_size,
            timescales=source.config.timescales,
            max_alm_bytes=byte_class,
            column_size=source.config.column_size,
            contextual_residual=source.config.contextual_residual,
        )
        current, trace, live = _tiled_channels(source, config.max_relations)
        for workers in args.workers:
            for strategy in args.strategies:
                partition = partition_cortical_columns(
                    current,
                    trace,
                    live,
                    vocab_size=config.vocab_size,
                    column_size=config.column_size,
                    worker_count=workers,
                    strategy=strategy,
                )
                rows.append(
                    {
                        "alm_class_bytes": byte_class,
                        "relations": config.max_relations,
                        "columns": len(partition.column_to_worker),
                        "workers": workers,
                        "strategy": strategy,
                        "recursive_edges": partition.recursive_edges,
                        "cross_worker_edges": partition.cross_worker_edges,
                        "edge_locality": partition.edge_locality,
                        "load_imbalance": partition.load_imbalance,
                        "minimum_worker_relations": int(
                            partition.worker_loads.min()
                        ),
                        "maximum_worker_relations": int(
                            partition.worker_loads.max()
                        ),
                    }
                )
    payload = {
        "checkpoint": str(args.checkpoint),
        "source_live_relations": source.live_relation_count,
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
