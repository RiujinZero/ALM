"""Screen bounded homeostatic goal gain on a small raw-byte corpus slice."""

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
    gain: float,
    basin_width: float,
    max_goal_assemblies: int,
    train_bytes: int,
    eval_bytes: int,
    train_path: str,
    valid_path: str,
    checkpoint_dir: str | None,
    goal_permission_trials: int,
    goal_third_component: str,
    goal_route_context: str,
    goal_permission_context: str,
    maximum_goal_route_bonus: float,
    goal_reserve_activation: bool,
    maximum_goal_route_byte_weight: float,
    goal_route_projection_policy: str,
    goal_route_timing: str,
    goal_route_alignment: str,
    goal_flow_polarity: str,
    goal_lifecycle: str,
    goal_failure_patience: int,
    goal_target_alignment_weight: float,
    predictive_goal_basins: int,
    predictive_goal_birth_similarity: float,
    predictive_goal_hysteresis: float,
    predictive_goal_weight: float,
    goal_flow_mode_gate: str,
    goal_flow_mode_trials: int,
    goal_surprise_scope: str,
) -> tuple[tuple[float, float], dict[str, object]]:
    train = np.fromfile(train_path, dtype=np.uint8, count=train_bytes)
    valid = np.fromfile(valid_path, dtype=np.uint8, count=eval_bytes)
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        goal_apical_weight=gain,
        goal_basin_width=basin_width,
        max_goal_assemblies=max_goal_assemblies,
        max_alm_bytes=64_000_000,
        temporal_bins=(1, 2, 4, 8, 16, 32, 64),
        max_active_temporal_traces=64,
        route_confirmations=3,
        max_event_motifs=2,
        event_confirmations=2,
        boundary_surprise_bits=3.0,
        minimum_motif_events=3,
        maximum_motif_events=16,
        goal_permission_trial_limit=goal_permission_trials,
        goal_third_component=goal_third_component,
        goal_route_context=goal_route_context,
        goal_permission_context=goal_permission_context,
        maximum_goal_route_bonus=maximum_goal_route_bonus,
        goal_reserve_activation=goal_reserve_activation,
        maximum_goal_route_byte_weight=maximum_goal_route_byte_weight,
        goal_route_projection_policy=goal_route_projection_policy,
        goal_route_timing=goal_route_timing,
        goal_route_alignment=goal_route_alignment,
        goal_flow_polarity=goal_flow_polarity,
        goal_lifecycle=goal_lifecycle,
        goal_failure_patience=goal_failure_patience,
        goal_target_alignment_weight=goal_target_alignment_weight,
        predictive_goal_basins=predictive_goal_basins,
        predictive_goal_birth_similarity=predictive_goal_birth_similarity,
        predictive_goal_hysteresis=predictive_goal_hysteresis,
        predictive_goal_weight=predictive_goal_weight,
        goal_flow_mode_gate=goal_flow_mode_gate,
        goal_flow_mode_trials=goal_flow_mode_trials,
        goal_surprise_scope=goal_surprise_scope,
    )
    started = time.perf_counter()
    train_probability = learner.observe_chunk_probabilities(train, learn=True)
    train_seconds = time.perf_counter() - started
    train_goal_route_telemetry = {
        name: getattr(learner, name)
        for name in (
            "goal_route_context_calls",
            "goal_route_proposals",
            "goal_route_denials",
            "goal_route_switches",
            "goal_route_flow_uses",
        )
    }
    learner.reset_episode()
    started = time.perf_counter()
    eval_probability = learner.observe_chunk_probabilities(valid, learn=False)
    eval_seconds = time.perf_counter() - started
    condition = {
        "train_bpb": _bpb(train_probability),
        "eval_bpb": _bpb(eval_probability),
        "embodied_bytes": learner.embodied_bytes,
        "goal_embodied_bytes": learner.goal_organ.embodied_bytes,
        "goal_transient_bytes": learner.goal_transient_bytes,
        "goal_values": len(learner.goal_organ.restoration_sum),
        "goal_selections": learner.goal_selections,
        "goal_births": learner.goal_organ.goal_births,
        "goal_dissolutions": learner.goal_organ.goal_dissolutions,
        "regulatory_mode_births": learner.goal_organ.mode_births,
        "regulatory_mode_dissolutions": learner.goal_organ.mode_dissolutions,
        "goal_route_paths": len(getattr(learner, "goal_route_counts", {})),
        "goal_route_synapses": sum(
            len(row) for row in getattr(learner, "goal_route_counts", {}).values()
        ),
        "goal_route_nursery_paths": len(getattr(learner, "goal_route_nursery", {})),
        "goal_route_credit_paths": len(learner.goal_route_restoration_sum),
        "goal_route_credit_scalars": 3 * len(learner.goal_route_restoration_sum),
        "maximum_goal_route_bonus": learner.maximum_goal_route_bonus,
        "goal_selection_paths": len(learner.goal_selection_utility),
        "positive_goal_selection_paths": sum(
            utility > learner.minimum_goal_selection_utility
            for utility in learner.goal_selection_utility.values()
        ),
        "goal_selection_evidence": sum(learner.goal_selection_evidence.values()),
        "goal_permission_trials": goal_permission_trials,
        "goal_permission_paths": len(learner.goal_permission_restoration_sum),
        "positive_goal_permission_paths": sum(
            restoration > 0.0
            for restoration in learner.goal_permission_restoration_sum.values()
        ),
        "goal_permission_evidence": sum(
            learner.goal_permission_restoration_evidence.values()
        ),
        "goal_flow_mode_gate": learner.goal_flow_mode_gate,
        "goal_lifecycle": learner.goal_lifecycle,
        "goal_failure_patience": learner.goal_failure_patience,
        "goal_target_alignment_weight": learner.goal_target_alignment_weight,
        "predictive_goal_basins": learner.predictive_goal_basins,
        "predictive_goal_birth_similarity": (learner.predictive_goal_birth_similarity),
        "predictive_goal_hysteresis": learner.predictive_goal_hysteresis,
        "predictive_goal_weight": learner.predictive_goal_weight,
        "predictive_goal_live_basins": (
            0
            if learner.predictive_goal_organ is None
            else learner.predictive_goal_organ.live_basins
        ),
        "predictive_goal_basin_switches": (
            0
            if learner.predictive_goal_organ is None
            else learner.predictive_goal_organ.basin_switches
        ),
        "goal_flow_mode_trials": learner.goal_flow_mode_trials,
        "goal_flow_mode_utility_bits": learner.goal_flow_mode_utility.tolist(),
        "goal_flow_mode_evidence": learner.goal_flow_mode_evidence.tolist(),
        "goal_flow_mode_candidates": learner.goal_flow_mode_candidates.tolist(),
        "goal_flow_mode_applications": learner.goal_flow_mode_applications.tolist(),
        "train_goal_route_telemetry": train_goal_route_telemetry,
        "eval_goal_route_telemetry": {
            name: getattr(learner, name) - value
            for name, value in train_goal_route_telemetry.items()
        },
        "train_goal_route_switches": train_goal_route_telemetry["goal_route_switches"],
        "eval_goal_route_switches": (
            learner.goal_route_switches
            - train_goal_route_telemetry["goal_route_switches"]
        ),
        "train_bytes_per_second": len(train) / max(train_seconds, 1e-12),
        "eval_bytes_per_second": len(valid) / max(eval_seconds, 1e-12),
    }
    learner.reset_episode()
    if checkpoint_dir is not None:
        checkpoint = Path(checkpoint_dir) / f"gain_{gain:g}_width_{basin_width:g}.pkl"
        _atomic_pickle(checkpoint, {"learner": learner, "report": condition})
        condition["checkpoint"] = str(checkpoint)
    return (gain, basin_width), condition


def run_probe(
    *,
    train_bytes: int = 100_000,
    eval_bytes: int = 100_000,
    gains: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1),
    basin_widths: tuple[float, ...] = (0.125,),
    max_goal_assemblies: int = 256,
    workers: int = 1,
    train_path: str = DEFAULT_TRAIN_PATH,
    valid_path: str = DEFAULT_VALID_PATH,
    checkpoint_dir: str | None = None,
    goal_permission_trials: int = 0,
    goal_third_component: str = "capacity_pressure",
    goal_route_context: str = "deficit",
    goal_permission_context: str = "direction",
    maximum_goal_route_bonus: float = 0.5,
    goal_reserve_activation: bool = False,
    maximum_goal_route_byte_weight: float = 0.0,
    goal_route_projection_policy: str = "switched_winner",
    goal_route_timing: str = "linear_age",
    goal_route_alignment: str = "target_basin_mass",
    goal_flow_polarity: str = "selected_goal",
    goal_lifecycle: str = "homeostatic",
    goal_failure_patience: int = 4,
    goal_target_alignment_weight: float = 1.0,
    predictive_goal_basins: int = 0,
    predictive_goal_birth_similarity: float = 0.75,
    predictive_goal_hysteresis: float = 0.05,
    predictive_goal_weight: float = 0.25,
    goal_flow_mode_gate: str = "all",
    goal_flow_mode_trials: int = 32,
    goal_surprise_scope: str = "alphabetic",
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    conditions: dict[str, object] = {}
    wall_started = time.perf_counter()
    arguments = [
        (
            gain,
            basin_width,
            max_goal_assemblies,
            train_bytes,
            eval_bytes,
            train_path,
            valid_path,
            checkpoint_dir,
            goal_permission_trials,
            goal_third_component,
            goal_route_context,
            goal_permission_context,
            maximum_goal_route_bonus,
            goal_reserve_activation,
            maximum_goal_route_byte_weight,
            goal_route_projection_policy,
            goal_route_timing,
            goal_route_alignment,
            goal_flow_polarity,
            goal_lifecycle,
            goal_failure_patience,
            goal_target_alignment_weight,
            predictive_goal_basins,
            predictive_goal_birth_similarity,
            predictive_goal_hysteresis,
            predictive_goal_weight,
            goal_flow_mode_gate,
            goal_flow_mode_trials,
            goal_surprise_scope,
        )
        for gain in gains
        for basin_width in basin_widths
    ]
    if workers == 1:
        outcomes = [_run_condition(*argument) for argument in arguments]
    else:
        outcomes = []
        with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as executor:
            pending = {
                executor.submit(_run_condition, *argument): argument[:2]
                for argument in arguments
            }
            for future in as_completed(pending):
                outcomes.append(future.result())
    for (gain, basin_width), condition in sorted(outcomes):
        conditions[f"gain={gain:g},width={basin_width:g}"] = condition
    wall_seconds = time.perf_counter() - wall_started
    return {
        "task": "raw_byte_goal_apical_gain_screen",
        "train_bytes": train_bytes,
        "eval_bytes": eval_bytes,
        "workers": min(workers, len(arguments)),
        "wall_seconds": wall_seconds,
        "aggregate_bytes_per_second": (
            len(arguments) * (train_bytes + eval_bytes) / max(wall_seconds, 1e-12)
        ),
        "max_goal_assemblies": max_goal_assemblies,
        "basin_widths": basin_widths,
        "viability_components": [
            "externally_scored_normalized_motif_surprise",
            "recent_motif_repetition",
            goal_third_component,
        ],
        "goal_route_context": goal_route_context,
        "goal_permission_context": goal_permission_context,
        "maximum_goal_route_bonus": maximum_goal_route_bonus,
        "goal_reserve_activation": goal_reserve_activation,
        "maximum_goal_route_byte_weight": maximum_goal_route_byte_weight,
        "goal_route_projection_policy": goal_route_projection_policy,
        "goal_route_timing": goal_route_timing,
        "goal_route_alignment": goal_route_alignment,
        "goal_flow_polarity": goal_flow_polarity,
        "goal_lifecycle": goal_lifecycle,
        "goal_failure_patience": goal_failure_patience,
        "goal_target_alignment_weight": goal_target_alignment_weight,
        "predictive_goal_basins": predictive_goal_basins,
        "predictive_goal_birth_similarity": predictive_goal_birth_similarity,
        "predictive_goal_hysteresis": predictive_goal_hysteresis,
        "predictive_goal_weight": predictive_goal_weight,
        "goal_flow_mode_gate": goal_flow_mode_gate,
        "goal_flow_mode_trials": goal_flow_mode_trials,
        "goal_surprise_scope": goal_surprise_scope,
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bytes", type=int, default=100_000)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument(
        "--gains", type=float, nargs="+", default=(0.0, 0.02, 0.05, 0.1)
    )
    parser.add_argument("--basin-widths", type=float, nargs="+", default=(0.125,))
    parser.add_argument("--max-goal-assemblies", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--goal-permission-trials", type=int, default=0)
    parser.add_argument(
        "--goal-third-component",
        choices=("capacity_pressure", "temporal_disagreement"),
        default="capacity_pressure",
    )
    parser.add_argument(
        "--goal-route-context",
        choices=("deficit", "restorative_basin"),
        default="deficit",
    )
    parser.add_argument(
        "--goal-permission-context",
        choices=("direction", "goal_basin"),
        default="direction",
    )
    parser.add_argument("--maximum-goal-route-bonus", type=float, default=0.5)
    parser.add_argument("--goal-reserve-activation", action="store_true")
    parser.add_argument("--maximum-goal-route-byte-weight", type=float, default=0.0)
    parser.add_argument(
        "--goal-route-projection-policy",
        choices=("switched_winner", "aligned_flow"),
        default="switched_winner",
    )
    parser.add_argument(
        "--goal-route-timing",
        choices=("linear_age", "learned_hazard"),
        default="linear_age",
    )
    parser.add_argument(
        "--goal-route-alignment",
        choices=("target_basin_mass", "restoration_signature"),
        default="target_basin_mass",
    )
    parser.add_argument(
        "--goal-flow-polarity",
        choices=(
            "selected_goal",
            "regulatory_mode",
            "regulatory_goal",
            "regulatory_basin",
        ),
        default="selected_goal",
    )
    parser.add_argument(
        "--goal-lifecycle",
        choices=("homeostatic", "target_or_failure"),
        default="homeostatic",
    )
    parser.add_argument("--goal-failure-patience", type=int, default=4)
    parser.add_argument("--goal-target-alignment-weight", type=float, default=1.0)
    parser.add_argument("--predictive-goal-basins", type=int, default=0)
    parser.add_argument("--predictive-goal-birth-similarity", type=float, default=0.75)
    parser.add_argument("--predictive-goal-hysteresis", type=float, default=0.05)
    parser.add_argument("--predictive-goal-weight", type=float, default=0.25)
    parser.add_argument(
        "--goal-flow-mode-gate",
        choices=("all", "learned_utility"),
        default="all",
    )
    parser.add_argument("--goal-flow-mode-trials", type=int, default=32)
    parser.add_argument(
        "--goal-surprise-scope",
        choices=("alphabetic", "all_bytes"),
        default="alphabetic",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = run_probe(
        train_bytes=args.train_bytes,
        eval_bytes=args.eval_bytes,
        gains=tuple(args.gains),
        basin_widths=tuple(args.basin_widths),
        max_goal_assemblies=args.max_goal_assemblies,
        workers=args.workers,
        train_path=args.train_path,
        valid_path=args.valid_path,
        checkpoint_dir=(
            None if args.checkpoint_dir is None else str(args.checkpoint_dir)
        ),
        goal_permission_trials=args.goal_permission_trials,
        goal_third_component=args.goal_third_component,
        goal_route_context=args.goal_route_context,
        goal_permission_context=args.goal_permission_context,
        maximum_goal_route_bonus=args.maximum_goal_route_bonus,
        goal_reserve_activation=args.goal_reserve_activation,
        maximum_goal_route_byte_weight=args.maximum_goal_route_byte_weight,
        goal_route_projection_policy=args.goal_route_projection_policy,
        goal_route_timing=args.goal_route_timing,
        goal_route_alignment=args.goal_route_alignment,
        goal_flow_polarity=args.goal_flow_polarity,
        goal_lifecycle=args.goal_lifecycle,
        goal_failure_patience=args.goal_failure_patience,
        goal_target_alignment_weight=args.goal_target_alignment_weight,
        predictive_goal_basins=args.predictive_goal_basins,
        predictive_goal_birth_similarity=args.predictive_goal_birth_similarity,
        predictive_goal_hysteresis=args.predictive_goal_hysteresis,
        predictive_goal_weight=args.predictive_goal_weight,
        goal_flow_mode_gate=args.goal_flow_mode_gate,
        goal_flow_mode_trials=args.goal_flow_mode_trials,
        goal_surprise_scope=args.goal_surprise_scope,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
