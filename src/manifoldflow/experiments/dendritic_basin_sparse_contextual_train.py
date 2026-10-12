"""Resumable raw-byte training for the sparse contextual opponent organ."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    ContextualOpponentAttractorLearner,
)
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--train-bytes", type=int, default=0, help="0 means full file")
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--chunk-bytes", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-every-bytes", type=int, default=10_000_000)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _summary(
    learner: ContextualOpponentAttractorLearner,
    *,
    offset: int,
    target: int,
    bit_sum: float,
    count: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    rate = offset / max(elapsed_seconds, 1e-12)
    return {
        "train_bytes_processed": offset,
        "train_bytes_target": target,
        "fraction_complete": offset / max(target, 1),
        "train_bits_per_byte": bit_sum / max(count, 1),
        "train_bytes_per_second": rate,
        "eta_seconds": (target - offset) / max(rate, 1e-12),
        "embodied_bytes": learner.embodied_bytes,
        "assembly_births": learner.assembly_births,
        "context_births": learner.context_births,
        "assembly_nursery": learner.nursery_count - len(learner.context_nursery),
        "context_nursery": len(learner.context_nursery),
        "basal_transitions": learner.live_transition_count,
        "context_branches": learner.live_context_count,
        "context_synapses": learner.live_context_synapse_count,
    }


def _save(
    path: Path,
    learner: ContextualOpponentAttractorLearner,
    *,
    offset: int,
    target: int,
    bit_sum: float,
    count: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    state = {
        "learner": learner,
        "offset": offset,
        "target": target,
        "bit_sum": bit_sum,
        "count": count,
        "elapsed_seconds": elapsed_seconds,
    }
    _atomic_pickle(path, state)
    summary = _summary(
        learner,
        offset=offset,
        target=target,
        bit_sum=bit_sum,
        count=count,
        elapsed_seconds=elapsed_seconds,
    )
    _atomic_json(path.with_suffix(path.suffix + ".progress.json"), summary)
    return summary


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_path)
    available = train_path.stat().st_size
    requested = available if args.train_bytes == 0 else args.train_bytes
    target = min(requested, available)
    if args.resume:
        with args.checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        learner = state["learner"]
        offset = int(state["offset"])
        bit_sum = float(state["bit_sum"])
        count = int(state["count"])
        prior_elapsed = float(state["elapsed_seconds"])
        if int(state["target"]) != target:
            raise ValueError("resume target differs from checkpoint target")
    else:
        learner = ContextualOpponentAttractorLearner()
        offset = 0
        bit_sum = 0.0
        count = 0
        prior_elapsed = 0.0

    started = time.perf_counter()
    next_checkpoint = (
        (offset // args.checkpoint_every_bytes) + 1
    ) * args.checkpoint_every_bytes
    with train_path.open("rb") as handle:
        handle.seek(offset)
        while offset < target:
            raw = handle.read(min(args.chunk_bytes, target - offset))
            if not raw:
                break
            values = np.frombuffer(raw, dtype=np.uint8)
            probability = learner.observe_chunk_probabilities(values, learn=True)
            finite = probability[np.isfinite(probability)]
            bit_sum += float(np.sum(-np.log2(np.maximum(finite, 1e-300))))
            count += len(finite)
            offset += len(raw)
            elapsed = prior_elapsed + time.perf_counter() - started
            summary = _summary(
                learner,
                offset=offset,
                target=target,
                bit_sum=bit_sum,
                count=count,
                elapsed_seconds=elapsed,
            )
            print(
                f"[train] {offset:,}/{target:,} | "
                f"{summary['train_bytes_per_second']:,.0f} B/s | "
                f"{summary['train_bits_per_byte']:.4f} bpb | "
                f"body {summary['embodied_bytes']:,} B | "
                f"ETA {summary['eta_seconds'] / 3600:.2f}h",
                flush=True,
            )
            if offset >= next_checkpoint:
                _save(
                    args.checkpoint,
                    learner,
                    offset=offset,
                    target=target,
                    bit_sum=bit_sum,
                    count=count,
                    elapsed_seconds=elapsed,
                )
                next_checkpoint += args.checkpoint_every_bytes

    elapsed = prior_elapsed + time.perf_counter() - started
    summary = _save(
        args.checkpoint,
        learner,
        offset=offset,
        target=target,
        bit_sum=bit_sum,
        count=count,
        elapsed_seconds=elapsed,
    )
    learner.reset_episode()
    eval_start = time.perf_counter()
    with Path(args.valid_path).open("rb") as handle:
        valid = np.frombuffer(handle.read(args.eval_bytes), dtype=np.uint8)
    probability = learner.observe_chunk_probabilities(valid, learn=False)
    finite = probability[np.isfinite(probability)]
    eval_seconds = time.perf_counter() - eval_start
    summary.update(
        {
            "eval_bytes": len(valid),
            "eval_bits_per_byte": float(
                np.sum(-np.log2(np.maximum(finite, 1e-300))) / max(len(finite), 1)
            ),
            "eval_bytes_per_second": len(valid) / max(eval_seconds, 1e-12),
        }
    )
    report_path = args.report or args.checkpoint.with_suffix(".report.json")
    _atomic_json(report_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
