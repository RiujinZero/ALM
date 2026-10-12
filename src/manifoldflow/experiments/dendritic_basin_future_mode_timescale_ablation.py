"""Two-timescale categorical selection over frozen future-mode candidates.

The preceding fast-state selector narrowed surprise-grown futures but did not
beat a global mode-frequency prior after a synthetic interruption.  This
development-only assay adds a separately accumulated slow conversation state
and crosses fast-only, slow-only, and fast-plus-slow categorical readouts while
holding trainable readout allocation, candidate population, update count, and
shortlist policy fixed.

The slow state is a causal exponential aggregate of observed role-bound turn
tokens.  It is projected using training conversations only and never sees a
future target.  Successor responsibilities arrive after each selection and
train the one-layer readouts with local delta rules.  No native ALM checkpoint,
free-byte generation, autograd, backpropagation, or BPTT is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.categorical_future_mode_selector import (
    CategoricalFutureModeSelector,
    CategoricalFutureModeSelectorConfig,
)
from manifoldflow.dendritic_basin.future_turn_trajectory import FutureTurnTrajectory
from manifoldflow.dendritic_basin.surprise_grown_future_transition import (
    SurpriseGrownFutureTransition,
)
from manifoldflow.experiments import (
    dendritic_basin_future_mode_selector_ablation as selector_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_recurrent_codec_ablation as recurrent_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_surprise_future_transition_ablation as surprise_assay,
)

IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/dendritic_basin_future_mode_timescale_ablation.py",
    "tests/test_dendritic_basin_future_mode_timescale_ablation.py",
    *selector_assay.IMPLEMENTATION_PATHS,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(IMPLEMENTATION_PATHS):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return result


def training_slow_states(
    examples: Sequence[surprise_assay.TransitionExample],
    *,
    conversations: int,
    token_dimensions: int,
    decay: float,
) -> np.ndarray:
    """Causally aggregate the observed current token within each conversation."""

    if not 0.0 < decay <= 1.0:
        raise ValueError("slow decay must be in (0, 1]")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.conversation_index].append(index)
    if set(grouped) != set(range(conversations)):
        raise ValueError("each conversation must contribute a transition")
    states = np.zeros((len(examples), token_dimensions), dtype=np.float64)
    for conversation in range(conversations):
        slow = np.zeros(token_dimensions, dtype=np.float64)
        indexes = grouped[conversation]
        if indexes != sorted(
            indexes, key=lambda index: examples[index].transition_index
        ):
            raise AssertionError("transition chronology is not stable")
        for index in indexes:
            current_token = examples[index].context_state[:token_dimensions]
            slow = decay * slow + current_token
            states[index] = slow
    return states


def _slow_state_for_prefix(
    model: FutureTurnTrajectory,
    turns: Sequence[recurrent_assay.EncodedTurn],
    *,
    decay: float,
) -> np.ndarray:
    slow = np.zeros(model.token_dim, dtype=np.float64)
    for turn in turns:
        slow = decay * slow + model._token(turn)
    return slow


def development_examples_with_slow_state(
    model: FutureTurnTrajectory,
    codec: codec_assay.FutureTurnCodec,
    pairs: Sequence[codec_assay.DevelopmentPair],
    *,
    mode: str,
    slow_decay: float,
) -> list[
    tuple[
        codec_assay.DevelopmentPair,
        str,
        surprise_assay.TransitionExample,
        np.ndarray,
    ]
]:
    """Build the same causal development examples plus a slow prefix aggregate."""

    rows = []
    for pair_index, pair in enumerate(pairs):
        for side_index, (side_name, side) in enumerate(
            (("a", pair.side_a), ("b", pair.side_b))
        ):
            context = (
                side.original_context
                if mode == "original_unanswered_prefix"
                else side.interrupted_context
            )
            if context is None:
                raise ValueError("requested development context is unavailable")
            observed = recurrent_assay.observed_turns_from_transcript(
                context, generation_context=True
            )
            turns = tuple(
                recurrent_assay._encoded_event(
                    codec, turn, source_label=pair.source_label
                )
                for turn in observed
            )
            target = recurrent_assay._encoded_event(
                codec,
                recurrent_assay.ObservedTurn(
                    codec_assay.ASSISTANT_ROLE, side.full_response
                ),
                source_label=pair.source_label,
            )
            slow = _slow_state_for_prefix(model, turns, decay=slow_decay)
            model.reset()
            try:
                for turn in turns:
                    model.observe(turn)
                snapshot = model.snapshot()
                basal = model.predicted_tape().slots[0].vector
            finally:
                model.reset()
            rows.append(
                (
                    pair,
                    side_name,
                    surprise_assay.TransitionExample(
                        conversation_index=pair_index,
                        transition_index=side_index,
                        episode_id=pair.pair_id,
                        context_state=snapshot.state,
                        basal_prediction=basal,
                        target=target.vector,
                        target_turn=target,
                    ),
                    slow,
                )
            )
    return rows


def _selector(
    context_dimensions: int, args: argparse.Namespace, *, seed: int
) -> CategoricalFutureModeSelector:
    return CategoricalFutureModeSelector(
        CategoricalFutureModeSelectorConfig(
            context_dim=context_dimensions,
            mode_capacity=args.max_residual_modes,
            random_features=args.timescale_random_features,
            include_direct_context=False,
            feature_scale=args.timescale_feature_scale,
            learning_rate=args.timescale_learning_rate,
            weight_decay=args.timescale_weight_decay,
            initial_readout_scale=args.timescale_initial_readout_scale,
            probability_floor=args.selector_probability_floor,
            seed=seed,
        )
    )


def train_timescale_selectors(
    examples: Sequence[surprise_assay.TransitionExample],
    fast_states: np.ndarray,
    slow_states: np.ndarray,
    learner: SurpriseGrownFutureTransition,
    *,
    conversations: int,
    args: argparse.Namespace,
) -> tuple[dict[str, CategoricalFutureModeSelector], np.ndarray, dict[str, Any]]:
    """Fit matched fast, slow, joint, and shuffled-joint categorical readouts."""

    combined = np.concatenate((fast_states, slow_states), axis=1)
    selectors = {
        "fast_only": _selector(fast_states.shape[1], args, seed=args.timescale_seed),
        "slow_only": _selector(slow_states.shape[1], args, seed=args.timescale_seed),
        "fast_plus_slow": _selector(combined.shape[1], args, seed=args.timescale_seed),
        "successor_shuffled": _selector(
            combined.shape[1], args, seed=args.timescale_seed
        ),
        "frozen_random": _selector(
            combined.shape[1], args, seed=args.timescale_random_control_seed
        ),
    }
    order = surprise_assay._training_order(
        examples, conversations=conversations, seed=args.timescale_order_seed
    )
    permutation, shuffle_audit = surprise_assay.successor_target_permutation(
        examples, seed=args.timescale_shuffle_seed
    )
    actual_scorer = copy.deepcopy(learner)
    shuffled_scorer = copy.deepcopy(learner)
    global_counts = np.full(args.max_residual_modes + 1, args.global_prior)
    totals = {
        name: {"loss": 0.0, "events": 0}
        for name in (
            "fast_only",
            "slow_only",
            "fast_plus_slow",
            "global_frequency",
            "affinity",
            "successor_shuffled",
            "frozen_random",
        )
    }
    inputs = {
        "fast_only": fast_states,
        "slow_only": slow_states,
        "fast_plus_slow": combined,
        "successor_shuffled": combined,
        "frozen_random": combined,
    }
    for _pass in range(args.timescale_training_passes):
        for raw_index in order:
            index = int(raw_index)
            example = examples[index]
            actual = selector_assay.responsibility_surface(
                actual_scorer,
                fast_states[index],
                example.basal_prediction,
                example.target,
            )
            donor = examples[int(permutation[index])]
            shuffled_target = selector_assay.responsibility_surface(
                shuffled_scorer,
                fast_states[index],
                example.basal_prediction,
                donor.target,
            )
            predictions = {
                "fast_only": selectors["fast_only"]
                .predict(fast_states[index], actual.mode_ids)
                .probabilities,
                "slow_only": selectors["slow_only"]
                .predict(slow_states[index], actual.mode_ids)
                .probabilities,
                "fast_plus_slow": selectors["fast_plus_slow"]
                .predict(combined[index], actual.mode_ids)
                .probabilities,
                "global_frequency": selector_assay._global_selection(
                    global_counts, actual.mode_ids
                ).probabilities,
                "affinity": actual.affinity_probabilities,
                "successor_shuffled": selectors["successor_shuffled"]
                .predict(combined[index], actual.mode_ids)
                .probabilities,
                "frozen_random": selectors["frozen_random"]
                .predict(combined[index], actual.mode_ids)
                .probabilities,
            }
            for name, prediction in predictions.items():
                totals[name]["loss"] += selector_assay._cross_entropy_bits(
                    actual.target_probabilities, prediction
                )
                totals[name]["events"] += 1
            for name in ("fast_only", "slow_only", "fast_plus_slow"):
                selectors[name].update(
                    inputs[name][index],
                    actual.mode_ids,
                    actual.target_probabilities,
                )
            selectors["successor_shuffled"].update(
                combined[index],
                shuffled_target.mode_ids,
                shuffled_target.target_probabilities,
            )
            classes = np.asarray(
                (0, *(mode_id + 1 for mode_id in actual.mode_ids)),
                dtype=np.int64,
            )
            global_counts[classes] += actual.target_probabilities
    trainable = {name: selector.trainable_bytes for name, selector in selectors.items()}
    if len(set(trainable.values())) != 1:
        raise AssertionError("timescale readout trainable allocation differs")
    return (
        selectors,
        global_counts,
        {
            "target_timing": "categorical prediction is scored before successor update",
            "training_passes": args.timescale_training_passes,
            "order_sha256": _sha256(np.asarray(order, dtype="<i8").tobytes()),
            "prequential_categorical_log_loss_bits": {
                name: values["loss"] / values["events"]
                for name, values in totals.items()
            },
            "events": next(iter(totals.values()))["events"],
            "successor_shuffle": shuffle_audit,
            "trainable_bytes": trainable,
            "trainable_allocation_matched": True,
            "total_allocated_bytes": {
                name: selector.allocated_bytes for name, selector in selectors.items()
            },
            "fingerprints": {
                name: selector.fingerprint() for name, selector in selectors.items()
            },
            "global_counts_sha256": surprise_assay._array_sha256(global_counts),
        },
    )


def _pair_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row["partition"] == partition]
    pair_ids = tuple(dict.fromkeys(row["pair_id"] for row in selected))
    arms = tuple(selected[0]["arms"])
    metrics = tuple(selected[0]["arms"][arms[0]])

    def pair_means(arm: str, metric: str) -> list[float]:
        return [
            float(
                np.mean(
                    [
                        row["arms"][arm][metric]
                        for row in selected
                        if row["pair_id"] == pair_id
                    ]
                )
            )
            for pair_id in pair_ids
        ]

    arm_summary = {
        arm: {
            metric: surprise_assay._mean_ci(
                pair_means(arm, metric),
                seed=seed + arm_index * 100 + metric_index,
                resamples=resamples,
            )
            for metric_index, metric in enumerate(metrics)
        }
        for arm_index, arm in enumerate(arms)
    }
    contrasts: dict[str, Any] = {}
    categorical_controls = (
        "fast_only",
        "slow_only",
        "global_frequency",
        "affinity",
        "successor_shuffled",
        "frozen_random",
    )
    for index, control in enumerate(categorical_controls):
        joint = pair_means("fast_plus_slow", "categorical_log_loss_bits")
        other = pair_means(control, "categorical_log_loss_bits")
        contrasts[f"fast_plus_slow_minus_{control}_categorical_log_loss_bits"] = (
            surprise_assay._mean_ci(
                [left - right for left, right in zip(joint, other, strict=True)],
                seed=seed + 10_000 + index,
                resamples=resamples,
            )
        )
    for index, metric in enumerate(("mse", "proper_bits_per_coordinate")):
        joint = pair_means("fast_plus_slow", metric)
        basal = pair_means("basal", metric)
        contrasts[f"fast_plus_slow_minus_basal_{metric}"] = surprise_assay._mean_ci(
            [left - right for left, right in zip(joint, basal, strict=True)],
            seed=seed + 20_000 + index,
            resamples=resamples,
        )
    return {
        "pairs": len(pair_ids),
        "sides": len(selected),
        "arms": arm_summary,
        "paired_contrasts": contrasts,
    }


def evaluate_timescales(
    basal: FutureTurnTrajectory,
    codec: codec_assay.FutureTurnCodec,
    pairs: Sequence[codec_assay.DevelopmentPair],
    fast_projector: surprise_assay.ContextProjector,
    slow_projector: surprise_assay.ContextProjector,
    learner: SurpriseGrownFutureTransition,
    selectors: Mapping[str, CategoricalFutureModeSelector],
    global_counts: np.ndarray,
    proper_variance: np.ndarray,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fingerprints_before = {
        "basal": basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(learner),
        **{name: selector.fingerprint() for name, selector in selectors.items()},
    }
    context_results: dict[str, Any] = {}
    for mode_index, mode in enumerate(
        ("original_unanswered_prefix", "long_interruption_crossover")
    ):
        examples = development_examples_with_slow_state(
            basal, codec, pairs, mode=mode, slow_decay=args.slow_decay
        )
        scorer = copy.deepcopy(learner)
        rows = []
        for pair, side_name, example, slow_raw in examples:
            fast = fast_projector.transform(example.context_state)
            slow = slow_projector.transform(slow_raw)
            joint = np.concatenate((fast, slow))
            surface = selector_assay.responsibility_surface(
                scorer, fast, example.basal_prediction, example.target
            )
            distributions = {
                "fast_only": selectors["fast_only"].predict(fast, surface.mode_ids),
                "slow_only": selectors["slow_only"].predict(slow, surface.mode_ids),
                "fast_plus_slow": selectors["fast_plus_slow"].predict(
                    joint, surface.mode_ids
                ),
                "global_frequency": selector_assay._global_selection(
                    global_counts, surface.mode_ids
                ),
                "affinity": selector_assay._selection_from_probabilities(
                    surface.mode_ids, surface.affinity_probabilities
                ),
                "successor_shuffled": selectors["successor_shuffled"].predict(
                    joint, surface.mode_ids
                ),
                "frozen_random": selectors["frozen_random"].predict(
                    joint, surface.mode_ids
                ),
            }
            basal_metrics = {
                "categorical_log_loss_bits": -math.log2(
                    max(float(surface.target_probabilities[0]), 1e-12)
                ),
                "shortlist_size": 1.0,
                "shortlist_fraction": 1.0 / len(surface.target_probabilities),
                "posterior_mass_captured": float(surface.target_probabilities[0]),
                "posterior_argmax_recalled": float(
                    int(np.argmax(surface.target_probabilities)) == 0
                ),
                "mse": float(
                    np.mean(np.square(example.target - example.basal_prediction))
                ),
                "cosine": selector_assay._cosine(
                    example.target, example.basal_prediction
                ),
                "proper_bits_per_coordinate": (
                    surprise_assay.gaussian_bits_per_coordinate(
                        example.target,
                        (example.basal_prediction,),
                        (1.0,),
                        proper_variance,
                    )
                ),
            }
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "side": side_name,
                    "candidate_modes": len(surface.mode_ids),
                    "arms": {
                        "basal": basal_metrics,
                        **{
                            name: selector_assay.selection_metrics(
                                example.target,
                                surface,
                                distribution,
                                shortlist_mass=args.shortlist_mass,
                                shortlist_maximum=args.shortlist_maximum,
                                proper_variance=proper_variance,
                            )
                            for name, distribution in distributions.items()
                        },
                    },
                }
            )
        context_results[mode] = {
            "summaries": {
                partition: _pair_summary(
                    rows,
                    partition=partition,
                    seed=args.bootstrap_seed
                    + mode_index * 100_000
                    + (0 if partition == "selection" else 50_000),
                    resamples=args.bootstrap_resamples,
                )
                for partition in ("selection", "evaluation")
            },
            "pair_rows": rows,
        }
    fingerprints_after = {
        "basal": basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(learner),
        **{name: selector.fingerprint() for name, selector in selectors.items()},
    }
    if fingerprints_before != fingerprints_after or basal.observed_count != 0:
        raise AssertionError("evaluation changed frozen state")
    return {
        "context_modes": context_results,
        "fitted_fingerprints_before": fingerprints_before,
        "fitted_fingerprints_after": fingerprints_after,
        "all_predictors_frozen_and_unchanged": True,
    }


def development_gate(
    evaluation: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    summary = evaluation["context_modes"]["long_interruption_crossover"]["summaries"][
        "evaluation"
    ]
    arms = summary["arms"]
    contrasts = summary["paired_contrasts"]
    fast = contrasts["fast_plus_slow_minus_fast_only_categorical_log_loss_bits"]
    global_frequency = contrasts[
        "fast_plus_slow_minus_global_frequency_categorical_log_loss_bits"
    ]
    shuffled = contrasts[
        "fast_plus_slow_minus_successor_shuffled_categorical_log_loss_bits"
    ]
    mse = contrasts["fast_plus_slow_minus_basal_mse"]
    proper = contrasts["fast_plus_slow_minus_basal_proper_bits_per_coordinate"]
    joint = arms["fast_plus_slow"]
    components = {
        "joint_beats_fast_only": fast["ci95_high"] < 0.0,
        "joint_beats_global_frequency": global_frequency["ci95_high"] < 0.0,
        "joint_beats_successor_shuffled": shuffled["ci95_high"] < 0.0,
        "shortlist_size_at_most_locked_maximum": (
            joint["shortlist_size"]["mean"] <= args.shortlist_maximum
        ),
        "posterior_mass_at_least_locked_minimum": (
            joint["posterior_mass_captured"]["mean"] >= args.minimum_posterior_mass
        ),
        "argmax_recall_at_least_locked_minimum": (
            joint["posterior_argmax_recalled"]["mean"] >= args.minimum_argmax_recall
        ),
        "shortlisted_prediction_beats_basal": (
            mse["ci95_high"] < 0.0 or proper["ci95_high"] < 0.0
        ),
    }
    return {
        "scope": "long_interruption_crossover/evaluation_partition",
        "fast_plus_slow_minus_fast_only_categorical_log_loss_bits": fast,
        "fast_plus_slow_minus_global_frequency_categorical_log_loss_bits": (
            global_frequency
        ),
        "fast_plus_slow_minus_successor_shuffled_categorical_log_loss_bits": shuffled,
        "mean_shortlist_size": joint["shortlist_size"]["mean"],
        "mean_posterior_mass_captured": joint["posterior_mass_captured"]["mean"],
        "posterior_argmax_recall": joint["posterior_argmax_recalled"]["mean"],
        "fast_plus_slow_minus_basal_mse": mse,
        "fast_plus_slow_minus_basal_proper_bits_per_coordinate": proper,
        "components": components,
        "slow_state_contribution_demonstrated": (
            components["joint_beats_fast_only"]
            and components["joint_beats_global_frequency"]
            and components["joint_beats_successor_shuffled"]
        ),
        "advances_to_expensive_competition": all(components.values()),
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if not args.latent_only:
        raise ValueError("this assay is latent-only")
    manifest, verified, conversations, pairs = (
        surprise_assay.load_declared_training_and_development(args)
    )
    assistant_turns = [
        turn for conversation in conversations for turn in conversation.assistant_turns
    ]
    codec = codec_assay.FutureTurnCodec.fit(
        assistant_turns,
        raw_dimensions=args.raw_dimensions,
        content_dimensions=args.content_dimensions,
        active_content_features=args.active_content_features,
    )
    sequences = recurrent_assay.encode_training_sequences(conversations, codec)
    crossfit = surprise_assay.cross_fitted_basal_transitions(
        conversations,
        sequences,
        content_dimensions=codec.content_dimensions,
        folds=args.crossfit_folds,
        fold_seed=args.crossfit_seed,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    fast_matrix = np.stack([item.context_state for item in crossfit.examples])
    fast_projector = surprise_assay.ContextProjector.fit(
        fast_matrix,
        dimensions=args.context_dimensions,
        scale_floor=args.context_scale_floor,
        clip=args.context_clip,
    )
    fast_states = fast_projector.transform(fast_matrix)
    token_dimensions = crossfit.examples[0].context_state.size // 2
    slow_matrix = training_slow_states(
        crossfit.examples,
        conversations=len(conversations),
        token_dimensions=token_dimensions,
        decay=args.slow_decay,
    )
    slow_projector = surprise_assay.ContextProjector.fit(
        slow_matrix,
        dimensions=args.slow_dimensions,
        scale_floor=args.context_scale_floor,
        clip=args.context_clip,
    )
    slow_states = slow_projector.transform(slow_matrix)
    ordered, _unused, overlay = surprise_assay.train_surprise_arms(
        crossfit.examples,
        fast_projector,
        conversations=len(conversations),
        args=args,
    )
    selectors, global_counts, selector_training = train_timescale_selectors(
        crossfit.examples,
        fast_states,
        slow_states,
        ordered,
        conversations=len(conversations),
        args=args,
    )
    full_basal = recurrent_assay.fit_recurrent_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    assistant_examples = [
        item
        for item in crossfit.examples
        if item.target_turn.role_id == codec_assay.ASSISTANT_ROLE
    ]
    residuals = np.stack(
        [item.target - item.basal_prediction for item in assistant_examples]
    )
    proper_variance = np.maximum(
        np.mean(np.square(residuals), axis=0), args.proper_variance_floor
    )
    evaluation = evaluate_timescales(
        full_basal,
        codec,
        pairs,
        fast_projector,
        slow_projector,
        ordered,
        selectors,
        global_counts,
        proper_variance,
        args=args,
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit exceeded")
    result = {
        "schema_version": 1,
        "experiment": "future_mode_selector_two_timescale_latent_development",
        "status": "complete_development",
        "claim_scope": {
            "schema4_train_and_development_only": True,
            "native_alm_loaded": False,
            "free_byte_generation_run": False,
            "surprise_population_frozen_before_selector_training": True,
            "target_arrives_after_causal_selection": True,
            "backpropagation_used": False,
            "backpropagation_through_time_used": False,
            "fixed_random_features_and_local_delta_readout": True,
            "goal_or_preference_learning": False,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "dataset": {
            "path": str(args.dataset_dir),
            "manifest_schema_version": int(manifest["schema_version"]),
            "addressed_splits": ["train", "dev"],
        },
        "config": {
            "fast_dimensions": args.context_dimensions,
            "slow_dimensions": args.slow_dimensions,
            "slow_decay": args.slow_decay,
            "random_features_per_arm": args.timescale_random_features,
            "training_passes": args.timescale_training_passes,
            "learning_rate": args.timescale_learning_rate,
            "feature_scale": args.timescale_feature_scale,
            "weight_decay": args.timescale_weight_decay,
            "initial_readout_scale": args.timescale_initial_readout_scale,
            "selector_seed": args.timescale_seed,
            "order_seed": args.timescale_order_seed,
            "shuffle_seed": args.timescale_shuffle_seed,
            "random_control_seed": args.timescale_random_control_seed,
            "shortlist_mass": args.shortlist_mass,
            "shortlist_maximum": args.shortlist_maximum,
            "minimum_posterior_mass": args.minimum_posterior_mass,
            "minimum_argmax_recall": args.minimum_argmax_recall,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "training_transitions": len(crossfit.examples),
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
        },
        "cross_fitted_basal": crossfit.audit,
        "fast_projector": {
            "fingerprint": fast_projector.fingerprint(),
            "development_rows_used": 0,
        },
        "slow_projector": {
            "definition": "slow <- decay * slow + current role-bound observed token",
            "fingerprint": slow_projector.fingerprint(),
            "training_state_sha256": surprise_assay._array_sha256(slow_matrix),
            "development_rows_used": 0,
        },
        "surprise_population_training": overlay,
        "selector_training": selector_training,
        "evaluation": evaluation,
        "development_gate": development_gate(evaluation, args),
        "wall_seconds": elapsed,
    }
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = selector_assay._parser()
    parser.description = __doc__
    parser.add_argument("--slow-dimensions", type=int, default=24)
    parser.add_argument("--slow-decay", type=float, default=0.98)
    parser.add_argument("--timescale-random-features", type=int, default=96)
    parser.add_argument("--timescale-feature-scale", type=float, default=0.75)
    parser.add_argument("--timescale-training-passes", type=int, default=1)
    parser.add_argument("--timescale-learning-rate", type=float, default=0.08)
    parser.add_argument("--timescale-weight-decay", type=float, default=1e-5)
    parser.add_argument("--timescale-initial-readout-scale", type=float, default=0.01)
    parser.add_argument("--timescale-seed", type=int, default=261_201)
    parser.add_argument("--timescale-order-seed", type=int, default=261_202)
    parser.add_argument("--timescale-shuffle-seed", type=int, default=261_203)
    parser.add_argument("--timescale-random-control-seed", type=int, default=261_204)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_ablation(args)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
