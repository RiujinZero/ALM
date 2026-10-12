"""Benchmark packed cortical columns with sparsely scheduled episodic organs."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _enable_relational_access,
    _new_arm,
    _observe,
    _predict_encoding_only,
)
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _clear_fast_state,
)
from manifoldflow.experiments.dendritic_basin_phase111_resident_columns import (
    _migrate_sequence_checkpoint,
)


def _event_offsets(
    probability: np.ndarray,
    *,
    minimum_interval: int,
    surprise_change_bits: float,
) -> np.ndarray:
    """Return local threshold crossings separated by a refractory interval."""
    surprise = -np.log2(np.maximum(probability, 1e-300))
    change = np.abs(np.diff(surprise, prepend=surprise[0]))
    candidates = np.flatnonzero(
        np.isfinite(probability) & (change >= surprise_change_bits)
    )
    retained = []
    last = -minimum_interval
    for offset in candidates:
        value = int(offset)
        if value - last < minimum_interval:
            continue
        retained.append(value)
        last = value
    return np.asarray(retained, dtype=np.int64)


def _develop_event_summary(arm: object, stream: bytes) -> None:
    """Let one sparse sensory episode grow long-range tissue."""
    _clear_fast_state(arm)
    for byte_id in stream:
        distribution = _predict_encoding_only(arm)
        _observe(arm, byte_id, float(distribution[byte_id]), external=True)


def _column(
    cortex_checkpoint: str,
    overlay_checkpoint: str,
    data_path: str,
    start: int,
    train_bytes: int,
    event_window_bytes: int,
    minimum_event_interval: int,
    surprise_change_bits: float,
) -> dict[str, float | int]:
    with open(cortex_checkpoint, "rb") as handle:
        cortex_payload = pickle.load(handle)
    cortex = cortex_payload["learner"]
    _migrate_sequence_checkpoint(cortex)
    with open(overlay_checkpoint, "rb") as handle:
        overlay_payload = pickle.load(handle)
    arm = _new_arm(overlay_payload)
    _enable_relational_access(arm, frames_per_capsule=12)
    arm.relational_access.learning_mode = "branches"
    arm.relational_access.eligibility_mode = "parallel"
    arm.relational_access.consume_on_confirmation = True
    arm.relational_access.minimum_confirmation_age = 1
    arm.relational_access.branch_survival_mode = "surprise"

    with open(data_path, "rb") as handle:
        handle.seek(start)
        raw = handle.read(train_bytes)
    values = np.frombuffer(raw, dtype=np.uint8)
    cortex.reset_episode()
    started = time.perf_counter()
    probability = cortex.observe_chunk_probabilities(values, learn=True)
    cortical_seconds = time.perf_counter() - started
    offsets = _event_offsets(
        probability,
        minimum_interval=minimum_event_interval,
        surprise_change_bits=surprise_change_bits,
    )
    event_bytes = 0
    event_started = time.perf_counter()
    for offset in offsets:
        begin = max(0, int(offset) - event_window_bytes // 4)
        end = min(len(raw), begin + event_window_bytes)
        summary = raw[begin:end]
        event_bytes += len(summary)
        _develop_event_summary(arm, summary)
    event_seconds = time.perf_counter() - event_started
    finite = probability[np.isfinite(probability)]
    return {
        "train_bytes": len(values),
        "cortical_seconds": cortical_seconds,
        "event_seconds": event_seconds,
        "total_seconds": cortical_seconds + event_seconds,
        "train_bpb": float(np.mean(-np.log2(np.maximum(finite, 1e-300)))),
        "events": len(offsets),
        "event_bytes": event_bytes,
        "field_encodings": int(arm.field.encodings),
        "access_branches": int(
            sum(len(row) for row in arm.relational_access.branches.values())
        ),
        "access_confirmations": int(arm.access_confirmations),
    }


def run(
    cortex_checkpoint: Path,
    overlay_checkpoint: Path,
    data: Path,
    *,
    columns: int,
    train_bytes_per_column: int,
    event_window_bytes: int,
    minimum_event_interval: int,
    surprise_change_bits: float,
) -> dict[str, object]:
    if columns < 1 or train_bytes_per_column < 1:
        raise ValueError("column and byte counts must be positive")
    if event_window_bytes < 8 or minimum_event_interval < event_window_bytes:
        raise ValueError(
            "event interval must cover an event window of at least 8 bytes"
        )
    if surprise_change_bits < 0.0:
        raise ValueError("surprise change threshold must be non-negative")
    with cortex_checkpoint.open("rb") as handle:
        cortex_payload = pickle.load(handle)
    cortex = cortex_payload["learner"]
    _migrate_sequence_checkpoint(cortex)
    cortical_body = int(cortex.embodied_bytes)
    overlay_body = overlay_checkpoint.stat().st_size
    body_per_column = cortical_body + overlay_body
    required = columns * train_bytes_per_column
    if required > data.stat().st_size:
        raise ValueError("dataset is shorter than requested disjoint shards")

    wall_started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=columns) as pool:
        futures = {
            pool.submit(
                _column,
                str(cortex_checkpoint),
                str(overlay_checkpoint),
                str(data),
                column * train_bytes_per_column,
                train_bytes_per_column,
                event_window_bytes,
                minimum_event_interval,
                surprise_change_bits,
            ): column
            for column in range(columns)
        }
        for future in as_completed(futures):
            row = future.result()
            row["column"] = futures[future]
            rows.append(row)
    wall_seconds = time.perf_counter() - wall_started
    rows.sort(key=lambda row: int(row["column"]))
    total_bytes = sum(int(row["train_bytes"]) for row in rows)
    maximum_metabolic_seconds = max(float(row["total_seconds"]) for row in rows)
    return {
        "task": "phase112_dual_rate_resident_columns",
        "columns": columns,
        "cortical_body_bytes_per_column": cortical_body,
        "overlay_body_bytes_per_column": overlay_body,
        "aggregate_body_bytes": columns * body_per_column,
        "aggregate_body_gb": columns * body_per_column / 1_000_000_000.0,
        "train_bytes": total_bytes,
        "train_bytes_per_column": train_bytes_per_column,
        "event_window_bytes": event_window_bytes,
        "minimum_event_interval": minimum_event_interval,
        "surprise_change_bits": surprise_change_bits,
        "events": sum(int(row["events"]) for row in rows),
        "event_bytes": sum(int(row["event_bytes"]) for row in rows),
        "field_encodings": sum(int(row["field_encodings"]) for row in rows),
        "access_branches": sum(int(row["access_branches"]) for row in rows),
        "access_confirmations": sum(
            int(row["access_confirmations"]) for row in rows
        ),
        "steady_train_bytes_per_second": total_bytes
        / max(maximum_metabolic_seconds, 1e-12),
        "wall_bytes_per_second_including_load": total_bytes
        / max(wall_seconds, 1e-12),
        "mean_train_bpb": float(np.mean([row["train_bpb"] for row in rows])),
        "maximum_column_seconds": maximum_metabolic_seconds,
        "wall_seconds_including_load": wall_seconds,
        "columns_detail": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cortex_checkpoint", type=Path)
    parser.add_argument("overlay_checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--train-bytes-per-column", type=int, default=256_000)
    parser.add_argument("--event-window-bytes", type=int, default=64)
    parser.add_argument("--minimum-event-interval", type=int, default=256_000)
    parser.add_argument("--surprise-change-bits", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.cortex_checkpoint,
        args.overlay_checkpoint,
        args.data,
        columns=args.columns,
        train_bytes_per_column=args.train_bytes_per_column,
        event_window_bytes=args.event_window_bytes,
        minimum_event_interval=args.minimum_event_interval,
        surprise_change_bits=args.surprise_change_bits,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
