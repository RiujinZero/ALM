"""Train sparse opponent organs in parallel territories, then consolidate."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    ContextualOpponentAttractorLearner,
    EpisodicOpponentAttractorLearner,
    PhraseEventOpponentLearner,
    PredictiveMotifOpponentLearner,
    SequenceCortexOpponentLearner,
)
from manifoldflow.dendritic_basin.routed_event_memory import (
    GoalConditionedTemporalBasisMotifOpponentLearner,
    RoutedMotifOpponentLearner,
    TemporalBasisMotifOpponentLearner,
)
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
)
from manifoldflow.experiments.dendritic_basin_sparse_contextual_train import (
    _atomic_json,
    _atomic_pickle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--secondary-valid-path")
    parser.add_argument("--secondary-eval-bytes", type=int, default=100_000)
    parser.add_argument("--secondary-eval-label", default="secondary")
    parser.add_argument("--train-bytes", type=int, default=30_000_000)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument(
        "--territory-bytes",
        type=int,
        default=0,
        help="Bound each developmental territory; 0 makes one territory per worker",
    )
    parser.add_argument("--chunk-bytes", type=int, default=1_000_000)
    parser.add_argument("--episodic", action="store_true")
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--phrase-event", action="store_true")
    parser.add_argument("--motif-event", action="store_true")
    parser.add_argument("--routed-motif-event", action="store_true")
    parser.add_argument("--temporal-basis-event", action="store_true")
    parser.add_argument("--goal-conditioned-event", action="store_true")
    parser.add_argument("--goal-apical-weight", type=float, default=0.5)
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
    parser.add_argument("--binding-index-cells", type=int, default=0)
    parser.add_argument("--binding-role-assemblies", type=int, default=32)
    parser.add_argument("--binding-cells-per-association", type=int, default=2)
    parser.add_argument("--binding-activity-decay", type=float, default=0.98)
    parser.add_argument("--binding-minimum-activity", type=float, default=0.05)
    parser.add_argument("--binding-role-birth-similarity", type=float, default=0.8)
    parser.add_argument("--binding-surprise-bits", type=float, default=2.0)
    parser.add_argument("--binding-role-trial-limit", type=int, default=16)
    parser.add_argument("--binding-minimum-role-utility", type=float, default=0.0)
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
    parser.add_argument("--max-goal-assemblies", type=int, default=256)
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
    parser.add_argument("--event-words", type=int, default=2)
    parser.add_argument("--event-confirmations", type=int, default=2)
    parser.add_argument("--max-sequence-nursery-paths", type=int, default=50_000)
    parser.add_argument("--max-event-nursery-paths", type=int, default=50_000)
    parser.add_argument("--apical-strength", type=float, default=8.0)
    parser.add_argument("--maximum-apical-weight", type=float, default=0.5)
    parser.add_argument("--minimum-prefix-bytes", type=int, default=1)
    parser.add_argument("--boundary-surprise-bits", type=float, default=3.0)
    parser.add_argument("--minimum-motif-events", type=int, default=3)
    parser.add_argument("--maximum-motif-events", type=int, default=16)
    parser.add_argument("--route-lags", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--route-confirmations", type=int, default=3)
    parser.add_argument("--max-active-temporal-traces", type=int, default=64)
    parser.add_argument("--max-alm-bytes", type=int, default=64_000_000)
    parser.add_argument("--byte-budget-check-interval", type=int, default=5_000_000)
    parser.add_argument("--shed-nurseries", action="store_true")
    parser.add_argument(
        "--minimum-event-observations",
        type=int,
        default=0,
        help="sleep-prune mature event branches below this global count",
    )
    parser.add_argument("--consolidate-episodes", action="store_true")
    parser.add_argument("--local-prune-every-bytes", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _territory(
    train_path: str,
    start: int,
    length: int,
    chunk_bytes: int,
    episodic: bool,
    sequence: bool,
    phrase_event: bool,
    motif_event: bool,
    routed_motif_event: bool,
    temporal_basis_event: bool,
    goal_conditioned_event: bool,
    event_words: int,
    event_confirmations: int,
    max_sequence_nursery_paths: int,
    max_event_nursery_paths: int,
    apical_strength: float,
    maximum_apical_weight: float,
    minimum_prefix_bytes: int,
    boundary_surprise_bits: float,
    minimum_motif_events: int,
    maximum_motif_events: int,
    route_lags: tuple[int, ...],
    route_confirmations: int,
    max_active_temporal_traces: int,
    goal_apical_weight: float,
    max_goal_assemblies: int,
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
    binding_index_cells: int,
    binding_role_assemblies: int,
    binding_cells_per_association: int,
    binding_activity_decay: float,
    binding_minimum_activity: float,
    binding_role_birth_similarity: float,
    binding_surprise_bits: float,
    binding_role_trial_limit: int,
    binding_minimum_role_utility: float,
    goal_flow_mode_gate: str,
    goal_flow_mode_trials: int,
    goal_surprise_scope: str,
    max_alm_bytes: int,
    byte_budget_check_interval: int,
    local_prune_every_bytes: int,
) -> tuple[ContextualOpponentAttractorLearner, float, int, int]:
    learner = (
        EpisodicOpponentAttractorLearner()
        if episodic
        else (
            GoalConditionedTemporalBasisMotifOpponentLearner(
                max_event_motifs=event_words,
                event_confirmations=event_confirmations,
                max_sequence_nursery_paths=max_sequence_nursery_paths,
                max_event_nursery_paths=max_event_nursery_paths,
                apical_strength=apical_strength,
                maximum_apical_weight=maximum_apical_weight,
                minimum_prefix_events=minimum_prefix_bytes,
                boundary_surprise_bits=boundary_surprise_bits,
                minimum_motif_events=minimum_motif_events,
                maximum_motif_events=maximum_motif_events,
                temporal_bins=route_lags,
                max_active_temporal_traces=max_active_temporal_traces,
                route_confirmations=route_confirmations,
                goal_apical_weight=goal_apical_weight,
                max_goal_assemblies=max_goal_assemblies,
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
                binding_index_cells=binding_index_cells,
                binding_role_assemblies=binding_role_assemblies,
                binding_cells_per_association=binding_cells_per_association,
                binding_activity_decay=binding_activity_decay,
                binding_minimum_activity=binding_minimum_activity,
                binding_role_birth_similarity=binding_role_birth_similarity,
                binding_surprise_bits=binding_surprise_bits,
                binding_role_trial_limit=binding_role_trial_limit,
                binding_minimum_role_utility=binding_minimum_role_utility,
                goal_flow_mode_gate=goal_flow_mode_gate,
                goal_flow_mode_trials=goal_flow_mode_trials,
                goal_surprise_scope=goal_surprise_scope,
                max_alm_bytes=max_alm_bytes,
                byte_budget_check_interval=byte_budget_check_interval,
            )
            if goal_conditioned_event
            else TemporalBasisMotifOpponentLearner(
                max_event_motifs=event_words,
                event_confirmations=event_confirmations,
                max_sequence_nursery_paths=max_sequence_nursery_paths,
                max_event_nursery_paths=max_event_nursery_paths,
                apical_strength=apical_strength,
                maximum_apical_weight=maximum_apical_weight,
                minimum_prefix_events=minimum_prefix_bytes,
                boundary_surprise_bits=boundary_surprise_bits,
                minimum_motif_events=minimum_motif_events,
                maximum_motif_events=maximum_motif_events,
                temporal_bins=route_lags,
                max_active_temporal_traces=max_active_temporal_traces,
                route_confirmations=route_confirmations,
                max_alm_bytes=max_alm_bytes,
                byte_budget_check_interval=byte_budget_check_interval,
            )
            if temporal_basis_event
            else RoutedMotifOpponentLearner(
                max_event_motifs=event_words,
                event_confirmations=event_confirmations,
                max_sequence_nursery_paths=max_sequence_nursery_paths,
                max_event_nursery_paths=max_event_nursery_paths,
                apical_strength=apical_strength,
                maximum_apical_weight=maximum_apical_weight,
                minimum_prefix_events=minimum_prefix_bytes,
                boundary_surprise_bits=boundary_surprise_bits,
                minimum_motif_events=minimum_motif_events,
                maximum_motif_events=maximum_motif_events,
                route_lags=route_lags,
                route_confirmations=route_confirmations,
                max_alm_bytes=max_alm_bytes,
                byte_budget_check_interval=byte_budget_check_interval,
            )
            if routed_motif_event
            else PredictiveMotifOpponentLearner(
                max_event_motifs=event_words,
                event_confirmations=event_confirmations,
                max_sequence_nursery_paths=max_sequence_nursery_paths,
                max_event_nursery_paths=max_event_nursery_paths,
                apical_strength=apical_strength,
                maximum_apical_weight=maximum_apical_weight,
                minimum_prefix_events=minimum_prefix_bytes,
                boundary_surprise_bits=boundary_surprise_bits,
                minimum_motif_events=minimum_motif_events,
                maximum_motif_events=maximum_motif_events,
            )
            if motif_event
            else PhraseEventOpponentLearner(
                max_event_words=event_words,
                event_confirmations=event_confirmations,
                max_sequence_nursery_paths=max_sequence_nursery_paths,
                max_event_nursery_paths=max_event_nursery_paths,
                apical_strength=apical_strength,
                maximum_apical_weight=maximum_apical_weight,
                minimum_prefix_bytes=minimum_prefix_bytes,
            )
            if phrase_event
            else (
                SequenceCortexOpponentLearner(
                    max_sequence_nursery_paths=max_sequence_nursery_paths
                )
                if sequence
                else ContextualOpponentAttractorLearner()
            )
        )
    )
    bit_sum = 0.0
    count = 0
    locally_pruned = 0
    bytes_since_prune = 0
    remaining = length
    with open(train_path, "rb") as handle:
        handle.seek(start)
        while remaining:
            raw = handle.read(min(chunk_bytes, remaining))
            if not raw:
                break
            probability = learner.observe_chunk_probabilities(
                np.frombuffer(raw, dtype=np.uint8), learn=True
            )
            finite = probability[np.isfinite(probability)]
            bit_sum += float(np.sum(-np.log2(np.maximum(finite, 1e-300))))
            count += len(finite)
            remaining -= len(raw)
            bytes_since_prune += len(raw)
            if (
                isinstance(learner, EpisodicOpponentAttractorLearner)
                and local_prune_every_bytes > 0
                and bytes_since_prune >= local_prune_every_bytes
            ):
                locally_pruned += learner.consolidate_episodes(minimum_utility_bits=0.0)
                bytes_since_prune = 0
    learner.reset_episode()
    return learner, bit_sum, count, locally_pruned


def _partitions(
    total: int, workers: int, territory_bytes: int = 0
) -> list[tuple[int, int]]:
    if territory_bytes > 0:
        return [
            (offset, min(territory_bytes, total - offset))
            for offset in range(0, total, territory_bytes)
        ]
    workers = min(workers, total)
    base, remainder = divmod(total, workers)
    result = []
    offset = 0
    for index in range(workers):
        length = base + (1 if index < remainder else 0)
        result.append((offset, length))
        offset += length
    return result


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.train_bytes < 1 or args.chunk_bytes < 1:
        raise ValueError("workers, train-bytes, and chunk-bytes must be positive")
    if (
        sum(
            (
                args.episodic,
                args.sequence,
                args.phrase_event,
                args.motif_event,
                args.routed_motif_event,
                args.temporal_basis_event,
                args.goal_conditioned_event,
            )
        )
        > 1
    ):
        raise ValueError("select only one cortex architecture")
    routed_without_goal_merge = args.routed_motif_event or args.temporal_basis_event
    if routed_without_goal_merge and args.workers != 1:
        raise ValueError(
            "bounded routed motif consolidation is not yet parallel-safe; "
            "use --workers 1"
        )
    available = Path(args.train_path).stat().st_size
    target = min(args.train_bytes, available)
    partitions = _partitions(target, args.workers, args.territory_bytes)
    worker_count = min(args.workers, len(partitions))
    started = time.perf_counter()
    learner: ContextualOpponentAttractorLearner | None = None
    bit_sum = 0.0
    count = 0
    locally_pruned = 0
    consolidation_seconds = 0.0
    completed_bytes = 0
    completed_territories = 0
    progress_interval = max(1, len(partitions) // 100)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        pending = {
            executor.submit(
                _territory,
                args.train_path,
                start,
                length,
                args.chunk_bytes,
                args.episodic,
                args.sequence,
                args.phrase_event,
                args.motif_event,
                args.routed_motif_event,
                args.temporal_basis_event,
                args.goal_conditioned_event,
                args.event_words,
                args.event_confirmations,
                args.max_sequence_nursery_paths,
                args.max_event_nursery_paths,
                args.apical_strength,
                args.maximum_apical_weight,
                args.minimum_prefix_bytes,
                args.boundary_surprise_bits,
                args.minimum_motif_events,
                args.maximum_motif_events,
                tuple(args.route_lags),
                args.route_confirmations,
                args.max_active_temporal_traces,
                args.goal_apical_weight,
                args.max_goal_assemblies,
                args.goal_permission_trials,
                args.goal_third_component,
                args.goal_route_context,
                args.goal_permission_context,
                args.maximum_goal_route_bonus,
                args.goal_reserve_activation,
                args.maximum_goal_route_byte_weight,
                args.goal_route_projection_policy,
                args.goal_route_timing,
                args.goal_route_alignment,
                args.goal_flow_polarity,
                args.goal_lifecycle,
                args.goal_failure_patience,
                args.goal_target_alignment_weight,
                args.predictive_goal_basins,
                args.predictive_goal_birth_similarity,
                args.predictive_goal_hysteresis,
                args.predictive_goal_weight,
                args.binding_index_cells,
                args.binding_role_assemblies,
                args.binding_cells_per_association,
                args.binding_activity_decay,
                args.binding_minimum_activity,
                args.binding_role_birth_similarity,
                args.binding_surprise_bits,
                args.binding_role_trial_limit,
                args.binding_minimum_role_utility,
                args.goal_flow_mode_gate,
                args.goal_flow_mode_trials,
                args.goal_surprise_scope,
                args.max_alm_bytes,
                args.byte_budget_check_interval,
                args.local_prune_every_bytes,
            ): length
            for start, length in partitions
        }
        for future in as_completed(pending):
            territory_length = pending.pop(future)
            column, column_bits, column_count, column_pruned = future.result()
            if learner is None:
                learner = column
            else:
                merge_started = time.perf_counter()
                learner.consolidate(column)
                consolidation_seconds += time.perf_counter() - merge_started
            bit_sum += column_bits
            count += column_count
            locally_pruned += column_pruned
            completed_bytes += territory_length
            completed_territories += 1
            if (
                completed_territories % progress_interval == 0
                or completed_territories == len(partitions)
            ):
                elapsed = time.perf_counter() - started
                print(
                    f"[territories] {completed_territories:,}/{len(partitions):,} | "
                    f"{completed_bytes:,}/{target:,} bytes | "
                    f"{completed_bytes / max(elapsed, 1e-12):,.0f} B/s | "
                    f"merged episodes "
                    f"{getattr(learner, 'live_episode_count', 0):,}",
                    flush=True,
                )
            del column
    if learner is None:
        raise RuntimeError("no cortical territories completed")
    trained_seconds = time.perf_counter() - started - consolidation_seconds

    pruned_episodes = 0
    if args.consolidate_episodes:
        if not isinstance(learner, EpisodicOpponentAttractorLearner):
            raise ValueError("--consolidate-episodes requires --episodic")
        merge_started = time.perf_counter()
        pruned_episodes = learner.consolidate_episodes()
        consolidation_seconds += time.perf_counter() - merge_started

    shed_sequence_paths = 0
    shed_event_paths = 0
    if args.shed_nurseries and isinstance(learner, SequenceCortexOpponentLearner):
        merge_started = time.perf_counter()
        shed_sequence_paths = learner.shed_sequence_nursery()
        if isinstance(learner, PhraseEventOpponentLearner):
            shed_event_paths = learner.shed_event_nursery()
        consolidation_seconds += time.perf_counter() - merge_started

    pruned_event_branches = 0
    pruned_event_synapses = 0
    if args.minimum_event_observations:
        if not isinstance(learner, PhraseEventOpponentLearner):
            raise ValueError("event pruning requires an event cortex")
        merge_started = time.perf_counter()
        pruned_event_branches, pruned_event_synapses = (
            learner.consolidate_event_anatomy(
                minimum_observations=args.minimum_event_observations
            )
        )
        consolidation_seconds += time.perf_counter() - merge_started

    route_nursery_shed = 0
    if isinstance(learner, RoutedMotifOpponentLearner):
        merge_started = time.perf_counter()
        route_nursery_shed = learner.shed_route_nursery()
        learner.enforce_byte_budget(target_fraction=1.0)
        consolidation_seconds += time.perf_counter() - merge_started

    goal_route_counter_names = (
        "goal_route_context_calls",
        "goal_route_proposals",
        "goal_route_denials",
        "goal_route_switches",
        "goal_route_flow_uses",
    )
    train_goal_route_telemetry = (
        {name: getattr(learner, name) for name in goal_route_counter_names}
        if isinstance(learner, GoalConditionedTemporalBasisMotifOpponentLearner)
        else {}
    )
    with Path(args.valid_path).open("rb") as handle:
        valid = np.frombuffer(handle.read(args.eval_bytes), dtype=np.uint8)
    eval_started = time.perf_counter()
    probability = learner.observe_chunk_probabilities(valid, learn=False)
    eval_seconds = time.perf_counter() - eval_started
    finite = probability[np.isfinite(probability)]
    learner.reset_episode()
    if isinstance(learner, RoutedMotifOpponentLearner):
        learner.enforce_byte_budget(target_fraction=1.0)
    report = {
        "workers": worker_count,
        "territories": len(partitions),
        "territory_bytes": args.territory_bytes,
        "train_bytes": target,
        "train_bits_per_byte": bit_sum / max(count, 1),
        "train_seconds": trained_seconds,
        "train_bytes_per_second": target / max(trained_seconds, 1e-12),
        "consolidation_seconds": consolidation_seconds,
        "eval_bytes": len(valid),
        "eval_bits_per_byte": float(
            np.sum(-np.log2(np.maximum(finite, 1e-300))) / max(len(finite), 1)
        ),
        "eval_bytes_per_second": len(valid) / max(eval_seconds, 1e-12),
        "embodied_bytes": learner.embodied_bytes,
        "assembly_births": learner.assembly_births,
        "context_births": learner.context_births,
        "basal_transitions": learner.live_transition_count,
        "context_branches": learner.live_context_count,
        "context_synapses": learner.live_context_synapse_count,
        "pruned_episodes": pruned_episodes,
        "locally_pruned_episodes": locally_pruned,
        "shed_sequence_paths": shed_sequence_paths,
        "shed_event_paths": shed_event_paths,
        "pruned_event_branches": pruned_event_branches,
        "pruned_event_synapses": pruned_event_synapses,
        "route_nursery_shed": route_nursery_shed,
    }
    if args.secondary_valid_path is not None:
        learner.reset_episode()
        with Path(args.secondary_valid_path).open("rb") as handle:
            secondary = np.frombuffer(
                handle.read(args.secondary_eval_bytes), dtype=np.uint8
            )
        secondary_started = time.perf_counter()
        secondary_probability = learner.observe_chunk_probabilities(
            secondary, learn=False
        )
        secondary_seconds = time.perf_counter() - secondary_started
        secondary_finite = secondary_probability[np.isfinite(secondary_probability)]
        prefix = args.secondary_eval_label
        report.update(
            {
                f"{prefix}_eval_path": args.secondary_valid_path,
                f"{prefix}_eval_bytes": len(secondary),
                f"{prefix}_eval_bits_per_byte": float(
                    np.sum(-np.log2(np.maximum(secondary_finite, 1e-300)))
                    / max(len(secondary_finite), 1)
                ),
                f"{prefix}_eval_bytes_per_second": len(secondary)
                / max(secondary_seconds, 1e-12),
            }
        )
        learner.reset_episode()
    if isinstance(learner, EpisodicOpponentAttractorLearner):
        report.update(
            {
                "content_assemblies": learner.live_content_count,
                "episodic_branches": learner.live_episode_count,
                "episodic_synapses": learner.live_episode_synapse_count,
                "utility_gated_branches": sum(
                    utility > learner.minimum_utility_bits
                    for utility in learner.episodic_utility.values()
                ),
            }
        )
    if isinstance(learner, SequenceCortexOpponentLearner):
        report.update(
            {
                "sequence_assemblies": learner.live_sequence_count,
                "sequence_synapses": learner.live_sequence_synapse_count,
                "sequence_nursery": learner.sequence_nursery_count,
                "sequence_depth": learner.max_context_bytes,
                "sequence_nursery_evictions": learner.sequence_nursery_evictions,
            }
        )
    if isinstance(learner, PhraseEventOpponentLearner):
        report.update(
            {
                "word_assemblies": learner.live_word_count,
                "event_assemblies": learner.live_event_count,
                "event_synapses": learner.live_event_synapse_count,
                "event_nursery": learner.event_nursery_count,
                "event_depth": learner.max_event_words,
                "event_nursery_evictions": learner.event_nursery_evictions,
            }
        )
    if isinstance(learner, PredictiveMotifOpponentLearner):
        report.update(
            {
                "motif_assemblies": learner.live_motif_count,
                "learned_boundaries": learner.learned_boundaries,
                "boundary_surprise_bits": learner.boundary_surprise_bits,
            }
        )
    if isinstance(learner, RoutedMotifOpponentLearner):
        report.update(
            {
                "alm_class_bytes": learner.max_alm_bytes,
                "serialized_alm_bytes": learner.serialized_size_bytes(),
                "byte_occupancy": learner.byte_occupancy,
                "route_lags": learner.route_lags,
                "route_assemblies": learner.live_route_count,
                "route_synapses": learner.live_route_synapse_count,
                "route_nursery": learner.route_nursery_count,
                "route_births": learner.route_births,
                "route_nursery_evictions": learner.route_nursery_evictions,
                "route_capacity_rejections": learner.route_capacity_rejections,
                "budget_checks": learner.budget_checks,
                "budget_evictions": learner.budget_evictions,
                "budget_mature_replacements": learner.budget_mature_replacements,
                "budget_nursery_losses": learner.budget_nursery_losses,
                "budget_evictions_by_tissue": learner.budget_evictions_by_tissue,
                "competition_history": learner.competition_history,
                "motif_nursery": len(learner.motif_nursery),
                "motif_nursery_evictions": learner.motif_nursery_evictions,
                "motif_recycles": learner.motif_recycles,
                "growth_frozen": learner.growth_frozen,
            }
        )
    if isinstance(learner, TemporalBasisMotifOpponentLearner):
        report.update(
            {
                "temporal_bins": learner.temporal_bins,
                "max_active_temporal_traces": (learner.max_active_temporal_traces),
                "live_temporal_traces": learner.live_temporal_trace_count,
                "temporal_trace_evictions": learner.temporal_trace_evictions,
            }
        )
    if isinstance(learner, GoalConditionedTemporalBasisMotifOpponentLearner):
        report.update(
            {
                "goal_apical_weight": learner.goal_apical_weight,
                "goal_third_component": learner.goal_third_component,
                "goal_route_context": learner.goal_route_context,
                "goal_permission_context": learner.goal_permission_context,
                "goal_reserve_activation": learner.goal_reserve_activation,
                "maximum_goal_route_byte_weight": (
                    learner.maximum_goal_route_byte_weight
                ),
                "goal_route_projection_policy": (learner.goal_route_projection_policy),
                "goal_route_timing": learner.goal_route_timing,
                "goal_route_alignment": learner.goal_route_alignment,
                "goal_flow_polarity": learner.goal_flow_polarity,
                "goal_lifecycle": learner.goal_lifecycle,
                "goal_failure_patience": learner.goal_failure_patience,
                "goal_target_alignment_weight": (learner.goal_target_alignment_weight),
                "predictive_goal_basins": learner.predictive_goal_basins,
                "predictive_goal_birth_similarity": (
                    learner.predictive_goal_birth_similarity
                ),
                "predictive_goal_hysteresis": learner.predictive_goal_hysteresis,
                "predictive_goal_weight": learner.predictive_goal_weight,
                "binding_index_cells": learner.binding_index_cells,
                "binding_role_assemblies": learner.binding_role_assemblies,
                "binding_surprise_bits": learner.binding_surprise_bits,
                "binding_role_trial_limit": learner.binding_role_trial_limit,
                "binding_minimum_role_utility": (learner.binding_minimum_role_utility),
                "binding_role_live_assemblies": (
                    0
                    if learner.binding_role_organ is None
                    else learner.binding_role_organ.live_basins
                ),
                "binding_transient_bytes": (
                    0
                    if learner.binding_index is None
                    else learner.binding_index.transient_bytes
                ),
                "binding_role_utility_bits": (
                    learner.binding_role_utility_sum.tolist()
                ),
                "binding_role_utility_evidence": (
                    learner.binding_role_utility_evidence.tolist()
                ),
                "binding_role_trials": learner.binding_role_trials.tolist(),
                "binding_formation_events": learner.binding_formation_events,
                "binding_recall_attempts": learner.binding_recall_attempts,
                "binding_reinstatements": learner.binding_reinstatements,
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
                "goal_flow_mode_gate": learner.goal_flow_mode_gate,
                "goal_flow_mode_trials": learner.goal_flow_mode_trials,
                "goal_flow_mode_utility_bits": (
                    learner.goal_flow_mode_utility.tolist()
                ),
                "goal_flow_mode_evidence": learner.goal_flow_mode_evidence.tolist(),
                "goal_flow_mode_candidates": (
                    learner.goal_flow_mode_candidates.tolist()
                ),
                "goal_flow_mode_applications": (
                    learner.goal_flow_mode_applications.tolist()
                ),
                "goal_surprise_scope": learner.goal_surprise_scope,
                "goal_values": len(learner.goal_organ.restoration_sum),
                "goal_selections": learner.goal_selections,
                "goal_births": learner.goal_organ.goal_births,
                "goal_dissolutions": learner.goal_organ.goal_dissolutions,
                "goal_embodied_bytes": learner.goal_organ.embodied_bytes,
                "goal_transient_bytes": learner.goal_transient_bytes,
                "regulatory_mode_births": learner.goal_organ.mode_births,
                "regulatory_mode_dissolutions": (learner.goal_organ.mode_dissolutions),
                "goal_route_credit_paths": len(learner.goal_route_restoration_sum),
                "goal_route_credit_scalars": 3
                * len(learner.goal_route_restoration_sum),
                "maximum_goal_route_bonus": learner.maximum_goal_route_bonus,
                "goal_selection_paths": len(learner.goal_selection_utility),
                "positive_goal_selection_paths": sum(
                    utility > learner.minimum_goal_selection_utility
                    for utility in learner.goal_selection_utility.values()
                ),
                "goal_selection_evidence": sum(
                    learner.goal_selection_evidence.values()
                ),
                "goal_permission_trial_limit": learner.goal_permission_trial_limit,
                "goal_permission_paths": len(learner.goal_permission_restoration_sum),
                "positive_goal_permission_paths": sum(
                    restoration > 0.0
                    for restoration in (
                        learner.goal_permission_restoration_sum.values()
                    )
                ),
                "goal_permission_evidence": sum(
                    learner.goal_permission_restoration_evidence.values()
                ),
                "train_goal_route_telemetry": train_goal_route_telemetry,
                "eval_goal_route_telemetry": {
                    name: getattr(learner, name) - value
                    for name, value in train_goal_route_telemetry.items()
                },
                "train_goal_route_switches": train_goal_route_telemetry[
                    "goal_route_switches"
                ],
                "eval_goal_route_switches": (
                    learner.goal_route_switches
                    - train_goal_route_telemetry["goal_route_switches"]
                ),
            }
        )
    if args.checkpoint is not None:
        _atomic_pickle(args.checkpoint, {"learner": learner, "report": report})
    if args.report is not None:
        _atomic_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
