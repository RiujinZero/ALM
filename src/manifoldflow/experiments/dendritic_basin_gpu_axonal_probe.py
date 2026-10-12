"""Measure axonal queue compression at a requested ALM byte geometry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.axonal_index import AxonalConsumerIndex
from manifoldflow.dendritic_basin.gpu_axonal_scheduler import (
    CudaAxonalDepthScheduler,
)
from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticConfig,
    OnlineDendriticStreamLearner,
)
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_TRAIN_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alm-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--active-relation-channels", type=int, default=64)
    parser.add_argument("--persistent-relations", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--sample-bytes", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _tiled_anatomy(
    source: OnlineDendriticStreamLearner, target_relations: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_relations = source.config.max_relations
    source_ids = np.arange(target_relations) % source_relations
    groups = np.arange(target_relations) // source_relations
    channels = source._current_channels[source_ids].copy()
    relation = channels >= source.config.vocab_size
    channels[relation] = (
        source.config.vocab_size
        + groups[relation] * source_relations
        + channels[relation]
        - source.config.vocab_size
    )
    return (
        channels,
        source._depths[source_ids].copy(),
        np.ones(target_relations, dtype=np.bool_),
    )


def main() -> None:
    args = parse_args()
    source = OnlineDendriticStreamLearner.load_checkpoint(args.checkpoint)
    config = OnlineDendriticConfig(
        vocab_size=256,
        max_alm_bytes=args.alm_bytes,
        contextual_residual=True,
    )
    channels, depths, live = _tiled_anatomy(source, config.max_relations)
    index = AxonalConsumerIndex.from_arrays(
        channels,
        depths,
        live,
        vocab_size=256,
        max_depth=config.max_depth,
    )
    relation_fanout = np.diff(index.offsets)[256:]
    available = np.flatnonzero(relation_fanout > 0)
    active_relations = available[: args.active_relation_channels]
    sensory_fanout = np.diff(index.offsets)[:256]
    sensory_channel = int(np.argmax(sensory_fanout))
    active_channels = np.concatenate(
        (np.asarray((sensory_channel,), dtype=np.int32), 256 + active_relations)
    )
    persistent = np.arange(
        min(args.persistent_relations, config.max_relations), dtype=np.int32
    )

    started = time.perf_counter()
    expected = index.depth_queues(active_channels, persistent)
    cpu_seconds = time.perf_counter() - started
    scheduler = CudaAxonalDepthScheduler(index)
    scheduler.schedule(active_channels, persistent)
    timings = []
    actual = ()
    for _ in range(args.repeats):
        started = time.perf_counter()
        actual = scheduler.schedule(active_channels, persistent)
        timings.append(time.perf_counter() - started)
    for gpu_depth, cpu_depth in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(gpu_depth, cpu_depth)
    scheduled = sum(len(queue) for queue in actual)
    expected_sensory_fanout = None
    expected_scheduled_fraction = None
    if args.sample_path.exists():
        with args.sample_path.open("rb") as stream:
            sample = np.frombuffer(stream.read(args.sample_bytes), dtype=np.uint8)
        frequencies = np.bincount(sample, minlength=256).astype(np.float64)
        frequencies /= max(frequencies.sum(), 1.0)
        expected_sensory_fanout = float(frequencies @ sensory_fanout)
        expected_scheduled_fraction = min(
            1.0,
            (expected_sensory_fanout + len(persistent)) / config.max_relations,
        )
    payload = {
        "checkpoint": str(args.checkpoint),
        "alm_class_bytes": args.alm_bytes,
        "relations": config.max_relations,
        "channels": len(index.offsets) - 1,
        "axonal_index_bytes": index.storage_bytes,
        "active_channels": len(active_channels),
        "sensory_channel": sensory_channel,
        "sensory_fanout": int(sensory_fanout[sensory_channel]),
        "persistent_relations": len(persistent),
        "scheduled_relations": scheduled,
        "scheduled_fraction": scheduled / config.max_relations,
        "slot_polls_avoided": config.max_relations - scheduled,
        "cpu_queue_seconds": cpu_seconds,
        "gpu_queue_seconds_best": min(timings),
        "depth_counts": [len(queue) for queue in actual],
        "corpus_expected_sensory_fanout": expected_sensory_fanout,
        "corpus_expected_scheduled_fraction_upper_bound": (
            expected_scheduled_fraction
        ),
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
