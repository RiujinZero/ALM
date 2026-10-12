"""Screen a bounded transient basin-role-filler index on raw TinyStories bytes."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.routed_event_memory import (
    GoalConditionedTemporalBasisMotifOpponentLearner,
)
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
)
from manifoldflow.experiments.dendritic_basin_sparse_contextual_train import (
    _atomic_pickle,
)


def _bpb(probability: np.ndarray) -> float:
    finite = probability[np.isfinite(probability)]
    return float(np.mean(-np.log2(np.maximum(finite, 1e-300))))


def _run_condition(
    cells: int,
    surprise_bits: float,
    train_bytes: int,
    eval_bytes: int,
    train_path: str,
    valid_path: str,
    checkpoint_dir: str | None,
) -> tuple[tuple[int, float], dict[str, object]]:
    train = np.fromfile(train_path, dtype=np.uint8, count=train_bytes)
    valid = np.fromfile(valid_path, dtype=np.uint8, count=eval_bytes)
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        goal_apical_weight=8.0,
        max_goal_assemblies=256,
        max_alm_bytes=64_000_000,
        temporal_bins=(1, 2, 4, 8, 16, 32, 64),
        max_active_temporal_traces=64,
        route_confirmations=3,
        max_event_motifs=2,
        event_confirmations=2,
        boundary_surprise_bits=3.0,
        minimum_motif_events=3,
        maximum_motif_events=16,
        goal_permission_trial_limit=0,
        goal_third_component="temporal_disagreement",
        goal_route_context="restorative_basin",
        goal_permission_context="direction",
        maximum_goal_route_bonus=0.5,
        maximum_goal_route_byte_weight=0.02,
        goal_route_projection_policy="aligned_flow",
        goal_route_timing="learned_hazard",
        goal_route_alignment="restoration_signature",
        goal_flow_polarity="regulatory_basin",
        predictive_goal_basins=16,
        predictive_goal_birth_similarity=0.75,
        predictive_goal_hysteresis=0.5,
        predictive_goal_weight=1.0,
        binding_index_cells=cells,
        binding_role_assemblies=32,
        binding_cells_per_association=2,
        binding_activity_decay=0.98,
        binding_minimum_activity=0.05,
        binding_role_birth_similarity=0.8,
        binding_surprise_bits=surprise_bits,
        goal_flow_mode_gate="learned_utility",
        goal_flow_mode_trials=128,
        goal_surprise_scope="all_bytes",
    )
    started = time.perf_counter()
    train_probability = learner.observe_chunk_probabilities(train, learn=True)
    train_seconds = time.perf_counter() - started
    training_counters = {
        name: int(getattr(learner, name))
        for name in (
            "binding_role_observations",
            "binding_formation_events",
            "binding_recall_attempts",
            "binding_reinstatements",
        )
    }
    learner.reset_episode()
    started = time.perf_counter()
    eval_probability = learner.observe_chunk_probabilities(valid, learn=False)
    eval_seconds = time.perf_counter() - started
    role_organ = learner.binding_role_organ
    index = learner.binding_index
    condition = {
        "train_bpb": _bpb(train_probability),
        "eval_bpb": _bpb(eval_probability),
        "embodied_bytes": learner.embodied_bytes,
        "serialized_bytes": learner.serialized_size_bytes(),
        "binding_transient_bytes": 0 if index is None else index.transient_bytes,
        "binding_live_cells": 0 if index is None else index.live_cells,
        "binding_role_live_assemblies": (
            0 if role_organ is None else role_organ.live_basins
        ),
        "training": training_counters,
        "evaluation": {
            name: int(getattr(learner, name)) - value
            for name, value in training_counters.items()
        },
        "train_bytes_per_second": len(train) / max(train_seconds, 1e-12),
        "eval_bytes_per_second": len(valid) / max(eval_seconds, 1e-12),
    }
    learner.reset_episode()
    if checkpoint_dir is not None:
        checkpoint = (
            Path(checkpoint_dir) / f"cells_{cells}_surprise_{surprise_bits:g}.pkl"
        )
        _atomic_pickle(checkpoint, {"learner": learner, "report": condition})
        condition["checkpoint"] = str(checkpoint)
    return (cells, surprise_bits), condition


def run_probe(
    *,
    cells: tuple[int, ...] = (0, 256, 512),
    surprise_bits: tuple[float, ...] = (2.0,),
    train_bytes: int = 100_000,
    eval_bytes: int = 100_000,
    workers: int = 1,
    train_path: str = DEFAULT_TRAIN_PATH,
    valid_path: str = DEFAULT_VALID_PATH,
    checkpoint_dir: str | None = None,
) -> dict[str, object]:
    arguments = [
        (
            capacity,
            threshold,
            train_bytes,
            eval_bytes,
            train_path,
            valid_path,
            checkpoint_dir,
        )
        for capacity in cells
        for threshold in surprise_bits
    ]
    started = time.perf_counter()
    if workers == 1:
        outcomes = [_run_condition(*argument) for argument in arguments]
    else:
        outcomes = []
        with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as executor:
            pending = [
                executor.submit(_run_condition, *argument) for argument in arguments
            ]
            for future in as_completed(pending):
                outcomes.append(future.result())
    wall_seconds = time.perf_counter() - started
    return {
        "task": "phase72_transient_binding_index_screen",
        "train_bytes": train_bytes,
        "eval_bytes": eval_bytes,
        "workers": min(workers, len(arguments)),
        "wall_seconds": wall_seconds,
        "aggregate_bytes_per_second": (
            len(arguments) * (train_bytes + eval_bytes) / max(wall_seconds, 1e-12)
        ),
        "conditions": {
            f"cells={key[0]},surprise={key[1]:g}": value
            for key, value in sorted(outcomes)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(0, 256, 512))
    parser.add_argument("--surprise-bits", type=float, nargs="+", default=(2.0,))
    parser.add_argument("--train-bytes", type=int, default=100_000)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = run_probe(
        cells=tuple(args.cells),
        surprise_bits=tuple(args.surprise_bits),
        train_bytes=args.train_bytes,
        eval_bytes=args.eval_bytes,
        workers=args.workers,
        train_path=args.train_path,
        valid_path=args.valid_path,
        checkpoint_dir=(
            None if args.checkpoint_dir is None else str(args.checkpoint_dir)
        ),
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
