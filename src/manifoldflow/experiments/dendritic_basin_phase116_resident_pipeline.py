"""Measure an actual resident cortex-to-shared-organ developmental pipeline."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

try:
    from numba import set_num_threads
except ImportError:  # pragma: no cover - minimal installations use Python kernels

    def set_num_threads(_count: int) -> None:
        return None

from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _enable_relational_access,
    _new_arm,
)
from manifoldflow.experiments.dendritic_basin_phase111_resident_columns import (
    _migrate_sequence_checkpoint,
)
from manifoldflow.experiments.dendritic_basin_phase112_dual_rate_columns import (
    _develop_event_summary,
)


def _resident_column(
    column: int,
    checkpoint: str,
    data_path: str,
    start: int,
    rounds: int,
    chunk_bytes: int,
    event_window_bytes: int,
    minimum_event_interval: int,
    surprise_change_bits: float,
    ready_queue: object,
    output_queue: object,
    start_event: object,
) -> None:
    with open(checkpoint, "rb") as handle:
        payload = pickle.load(handle)
    cortex = payload["learner"]
    _migrate_sequence_checkpoint(cortex)
    with open(data_path, "rb") as handle:
        handle.seek(start)
        raw = handle.read(rounds * chunk_bytes)
    ready_queue.put(("cortex", column))
    start_event.wait()
    cortex.reset_episode()
    last_event = -minimum_event_interval
    previous_surprise: float | None = None
    for round_index in range(rounds):
        begin = round_index * chunk_bytes
        chunk = raw[begin : begin + chunk_bytes]
        values = np.frombuffer(chunk, dtype=np.uint8)
        started = time.perf_counter()
        probability = cortex.observe_chunk_probabilities(values, learn=True)
        cortical_seconds = time.perf_counter() - started
        surprise = -np.log2(np.maximum(probability, 1e-300))
        if previous_surprise is None:
            change = np.abs(np.diff(surprise, prepend=surprise[0]))
        else:
            change = np.abs(np.diff(surprise, prepend=previous_surprise))
        candidates = np.flatnonzero(
            np.isfinite(probability) & (change >= surprise_change_bits)
        )
        retained = []
        global_begin = round_index * chunk_bytes
        for offset in candidates:
            global_offset = global_begin + int(offset)
            if global_offset - last_event < minimum_event_interval:
                continue
            retained.append(int(offset))
            last_event = global_offset
        offsets = np.asarray(retained, dtype=np.int64)
        if len(surprise):
            previous_surprise = float(surprise[-1])
        summaries = []
        for offset in offsets:
            event_begin = max(0, int(offset) - event_window_bytes // 4)
            event_end = min(len(chunk), event_begin + event_window_bytes)
            summaries.append(chunk[event_begin:event_end])
        finite = probability[np.isfinite(probability)]
        output_queue.put(
            {
                "column": column,
                "round": round_index,
                "train_bytes": len(values),
                "scored_bytes": len(finite),
                "bits": float(
                    np.sum(-np.log2(np.maximum(finite, 1e-300)))
                ),
                "cortical_seconds": cortical_seconds,
                "summaries": summaries,
            }
        )


def _slow_lobe(
    lobe: int,
    overlay_checkpoint: str,
    event_window_bytes: int,
    slow_threads: int,
    ready_queue: object,
    input_queue: object,
    result_queue: object,
    start_event: object,
) -> None:
    set_num_threads(slow_threads)
    with open(overlay_checkpoint, "rb") as handle:
        overlay_payload = pickle.load(handle)
    arm = _new_arm(overlay_payload)
    _enable_relational_access(arm, frames_per_capsule=event_window_bytes)
    arm.relational_access.learning_mode = "branches"
    arm.relational_access.eligibility_mode = "parallel"
    arm.relational_access.consume_on_confirmation = True
    arm.relational_access.minimum_confirmation_age = 1
    arm.relational_access.branch_survival_mode = "surprise"
    ready_queue.put(("slow", lobe))
    start_event.wait()
    event_seconds = 0.0
    events = 0
    event_bytes = 0
    while True:
        summaries = input_queue.get()
        if summaries is None:
            break
        event_started = time.perf_counter()
        for summary in summaries:
            events += 1
            event_bytes += len(summary)
            _develop_event_summary(arm, summary)
        event_seconds += time.perf_counter() - event_started
    result_queue.put(
        {
            "lobe": lobe,
            "event_seconds": event_seconds,
            "events": events,
            "event_bytes": event_bytes,
            "field_encodings": int(arm.field.encodings),
            "access_branches": int(
                sum(len(row) for row in arm.relational_access.branches.values())
            ),
            "access_confirmations": int(arm.access_confirmations),
        }
    )


def run(
    cortex_checkpoint: Path,
    overlay_checkpoint: Path,
    data: Path,
    *,
    columns: int,
    slow_lobes: int,
    slow_threads: int,
    rounds: int,
    chunk_bytes: int,
    event_window_bytes: int,
    minimum_event_interval: int,
    surprise_change_bits: float,
) -> dict[str, object]:
    if (
        columns < 1
        or slow_lobes < 1
        or slow_threads < 1
        or rounds < 2
        or chunk_bytes < 1
    ):
        raise ValueError("columns and bytes must be positive; rounds must exceed one")
    if event_window_bytes < 8 or minimum_event_interval < event_window_bytes:
        raise ValueError(
            "event interval must cover an event window of at least 8 bytes"
        )
    required = columns * rounds * chunk_bytes
    if required > data.stat().st_size:
        raise ValueError("dataset is shorter than the requested resident streams")

    with cortex_checkpoint.open("rb") as handle:
        cortex_payload = pickle.load(handle)
    cortex = cortex_payload["learner"]
    _migrate_sequence_checkpoint(cortex)
    cortical_body = int(cortex.embodied_bytes)
    overlay_body = overlay_checkpoint.stat().st_size
    context = mp.get_context("spawn")
    ready_queue = context.Queue()
    output_queue = context.Queue()
    slow_result_queue = context.Queue()
    start_event = context.Event()
    slow_queues = [context.Queue() for _ in range(slow_lobes)]
    processes = []
    stream_bytes = rounds * chunk_bytes
    for column in range(columns):
        process = context.Process(
            target=_resident_column,
            args=(
                column,
                str(cortex_checkpoint),
                str(data),
                column * stream_bytes,
                rounds,
                chunk_bytes,
                event_window_bytes,
                minimum_event_interval,
                surprise_change_bits,
                ready_queue,
                output_queue,
                start_event,
            ),
        )
        process.start()
        processes.append(process)
    for lobe in range(slow_lobes):
        process = context.Process(
            target=_slow_lobe,
            args=(
                lobe,
                str(overlay_checkpoint),
                event_window_bytes,
                slow_threads,
                ready_queue,
                slow_queues[lobe],
                slow_result_queue,
                start_event,
            ),
        )
        process.start()
        processes.append(process)
    ready = {tuple(ready_queue.get()) for _ in range(columns + slow_lobes)}
    expected_ready = {
        *(("cortex", column) for column in range(columns)),
        *(("slow", lobe) for lobe in range(slow_lobes)),
    }
    if ready != expected_ready:
        raise RuntimeError("resident cortical or slow-organ workers did not initialize")

    started = time.perf_counter()
    start_event.set()
    expected_rows = columns * rounds
    received = 0
    pending: dict[tuple[int, int], dict[str, object]] = {}
    ordered_rows = []
    next_round = 0
    next_column = 0
    while received < expected_rows:
        row = output_queue.get()
        received += 1
        key = int(row["round"]), int(row["column"])
        pending[key] = row
        while (next_round, next_column) in pending:
            current = pending.pop((next_round, next_column))
            summaries = current.pop("summaries")
            slow_queues[next_column % slow_lobes].put(summaries)
            ordered_rows.append(current)
            next_column += 1
            if next_column == columns:
                next_column = 0
                next_round += 1
    for queue in slow_queues:
        queue.put(None)
    lobe_rows = [slow_result_queue.get() for _ in range(slow_lobes)]
    lobe_rows.sort(key=lambda row: int(row["lobe"]))
    metabolic_elapsed = time.perf_counter() - started
    shutdown_started = time.perf_counter()
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"resident cortical worker exited with code {process.exitcode}"
            )
    shutdown_seconds = time.perf_counter() - shutdown_started
    total_elapsed = time.perf_counter() - started
    total_bytes = sum(int(row["train_bytes"]) for row in ordered_rows)
    total_scored = sum(int(row["scored_bytes"]) for row in ordered_rows)
    total_bits = sum(float(row["bits"]) for row in ordered_rows)
    per_column_seconds = [
        sum(
            float(row["cortical_seconds"])
            for row in ordered_rows
            if int(row["column"]) == column
        )
        for column in range(columns)
    ]
    return {
        "task": "phase116_resident_pipeline",
        "columns": columns,
        "slow_lobes": slow_lobes,
        "slow_threads_per_lobe": slow_threads,
        "rounds": rounds,
        "chunk_bytes": chunk_bytes,
        "cortical_body_bytes_per_column": cortical_body,
        "shared_overlay_body_bytes": overlay_body,
        "aggregate_body_bytes": columns * cortical_body
        + slow_lobes * overlay_body,
        "aggregate_body_gb": (
            columns * cortical_body + slow_lobes * overlay_body
        )
        / 1_000_000_000.0,
        "train_bytes": total_bytes,
        "scored_bytes": total_scored,
        "train_bpb": total_bits / max(total_scored, 1),
        "events": sum(int(row["events"]) for row in lobe_rows),
        "event_bytes": sum(int(row["event_bytes"]) for row in lobe_rows),
        "event_cpu_seconds": sum(
            float(row["event_seconds"]) for row in lobe_rows
        ),
        "maximum_lobe_event_seconds": max(
            float(row["event_seconds"]) for row in lobe_rows
        ),
        "field_encodings": sum(int(row["field_encodings"]) for row in lobe_rows),
        "access_branches": sum(int(row["access_branches"]) for row in lobe_rows),
        "access_confirmations": sum(
            int(row["access_confirmations"]) for row in lobe_rows
        ),
        "actual_pipeline_seconds": metabolic_elapsed,
        "actual_pipeline_bytes_per_second": total_bytes
        / max(metabolic_elapsed, 1e-12),
        "worker_shutdown_seconds": shutdown_seconds,
        "pipeline_plus_shutdown_seconds": total_elapsed,
        "pipeline_plus_shutdown_bytes_per_second": total_bytes
        / max(total_elapsed, 1e-12),
        "maximum_column_cortical_seconds": max(per_column_seconds),
        "columns_detail": per_column_seconds,
        "lobes_detail": lobe_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cortex_checkpoint", type=Path)
    parser.add_argument("overlay_checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--columns", type=int, default=11)
    parser.add_argument("--slow-lobes", type=int, default=1)
    parser.add_argument("--slow-threads", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--chunk-bytes", type=int, default=256_000)
    parser.add_argument("--event-window-bytes", type=int, default=24)
    parser.add_argument("--minimum-event-interval", type=int, default=256_000)
    parser.add_argument("--surprise-change-bits", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.cortex_checkpoint,
        args.overlay_checkpoint,
        args.data,
        columns=args.columns,
        slow_lobes=args.slow_lobes,
        slow_threads=args.slow_threads,
        rounds=args.rounds,
        chunk_bytes=args.chunk_bytes,
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
