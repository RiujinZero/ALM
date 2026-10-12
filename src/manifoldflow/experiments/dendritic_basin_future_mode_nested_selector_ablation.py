"""Nested additive-logit ablation for slow future-mode selection.

A fast categorical selector is trained and frozen first.  Separate zero-logit
residual experts then receive either the slow conversation aggregate or the
same fast state.  Their logits are added to the frozen fast distribution, so
the nested model is exactly the fast model before residual learning.  The
fast-residual arm is capacity and update matched to distinguish useful slow
information from a generic second readout.

Successor responsibilities arrive only after each causal selection.  Every
trainable readout uses a one-layer local delta rule with fixed random features.
No native ALM checkpoint, free-byte generation, autograd, backpropagation, or
BPTT is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
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
    dendritic_basin_future_mode_timescale_ablation as timescale_assay,
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
    "src/manifoldflow/experiments/dendritic_basin_future_mode_nested_selector_ablation.py",
    "tests/test_dendritic_basin_future_mode_nested_selector_ablation.py",
    *timescale_assay.IMPLEMENTATION_PATHS,
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


def _selector(
    context_dimensions: int,
    args: argparse.Namespace,
    *,
    seed: int,
    residual: bool,
) -> CategoricalFutureModeSelector:
    return CategoricalFutureModeSelector(
        CategoricalFutureModeSelectorConfig(
            context_dim=context_dimensions,
            mode_capacity=args.max_residual_modes,
            random_features=args.nested_random_features,
            include_direct_context=False,
            feature_scale=args.nested_feature_scale,
            learning_rate=(
                args.nested_residual_learning_rate
                if residual
                else args.base_learning_rate
            ),
            weight_decay=args.nested_weight_decay,
            initial_readout_scale=(
                0.0 if residual else args.base_initial_readout_scale
            ),
            probability_floor=args.selector_probability_floor,
            seed=seed,
        )
    )


def train_nested_selectors(
    examples: Sequence[surprise_assay.TransitionExample],
    fast_states: np.ndarray,
    slow_states: np.ndarray,
    learner: SurpriseGrownFutureTransition,
    *,
    conversations: int,
    args: argparse.Namespace,
) -> tuple[
    CategoricalFutureModeSelector,
    dict[str, CategoricalFutureModeSelector],
    np.ndarray,
    dict[str, Any],
]:
    """Train a fast base, freeze it, then train matched residual experts."""

    base = _selector(fast_states.shape[1], args, seed=args.base_seed, residual=False)
    order = surprise_assay._training_order(
        examples, conversations=conversations, seed=args.nested_order_seed
    )
    permutation, shuffle_audit = surprise_assay.successor_target_permutation(
        examples, seed=args.nested_shuffle_seed
    )
    global_counts = np.full(args.max_residual_modes + 1, args.global_prior)
    base_scorer = copy.deepcopy(learner)
    base_loss = 0.0
    for _pass in range(args.base_training_passes):
        for raw_index in order:
            index = int(raw_index)
            example = examples[index]
            surface = selector_assay.responsibility_surface(
                base_scorer,
                fast_states[index],
                example.basal_prediction,
                example.target,
            )
            score = base.update(
                fast_states[index],
                surface.mode_ids,
                surface.target_probabilities,
            )
            base_loss += score["log_loss_bits"]
            classes = np.asarray(
                (0, *(mode_id + 1 for mode_id in surface.mode_ids)),
                dtype=np.int64,
            )
            global_counts[classes] += surface.target_probabilities
    base_frozen_fingerprint = base.fingerprint()

    residuals = {
        "slow": _selector(
            slow_states.shape[1], args, seed=args.residual_seed, residual=True
        ),
        "fast_capacity": _selector(
            fast_states.shape[1], args, seed=args.residual_seed, residual=True
        ),
        "successor_shuffled_slow": _selector(
            slow_states.shape[1], args, seed=args.residual_seed, residual=True
        ),
        "frozen_random_slow": _selector(
            slow_states.shape[1],
            args,
            seed=args.residual_random_control_seed,
            residual=False,
        ),
    }
    actual_scorer = copy.deepcopy(learner)
    shuffled_scorer = copy.deepcopy(learner)
    totals = {
        name: 0.0
        for name in (
            "fast_base",
            "nested_fast_slow",
            "nested_fast_fast",
            "nested_successor_shuffled_slow",
            "nested_frozen_random_slow",
            "global_frequency",
            "affinity",
        )
    }
    events = 0
    for _pass in range(args.residual_training_passes):
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
            prior = base.predict(fast_states[index], actual.mode_ids).probabilities
            predictions = {
                "fast_base": prior,
                "nested_fast_slow": residuals["slow"]
                .predict_with_prior(slow_states[index], actual.mode_ids, prior)
                .probabilities,
                "nested_fast_fast": residuals["fast_capacity"]
                .predict_with_prior(fast_states[index], actual.mode_ids, prior)
                .probabilities,
                "nested_successor_shuffled_slow": residuals["successor_shuffled_slow"]
                .predict_with_prior(slow_states[index], actual.mode_ids, prior)
                .probabilities,
                "nested_frozen_random_slow": residuals["frozen_random_slow"]
                .predict_with_prior(slow_states[index], actual.mode_ids, prior)
                .probabilities,
                "global_frequency": selector_assay._global_selection(
                    global_counts, actual.mode_ids
                ).probabilities,
                "affinity": actual.affinity_probabilities,
            }
            for name, prediction in predictions.items():
                totals[name] += selector_assay._cross_entropy_bits(
                    actual.target_probabilities, prediction
                )
            residuals["slow"].update_with_prior(
                slow_states[index],
                actual.mode_ids,
                prior,
                actual.target_probabilities,
            )
            residuals["fast_capacity"].update_with_prior(
                fast_states[index],
                actual.mode_ids,
                prior,
                actual.target_probabilities,
            )
            residuals["successor_shuffled_slow"].update_with_prior(
                slow_states[index],
                shuffled_target.mode_ids,
                prior,
                shuffled_target.target_probabilities,
            )
            events += 1
    if base.fingerprint() != base_frozen_fingerprint:
        raise AssertionError("residual training modified the frozen fast selector")
    trained_residual_bytes = {
        name: residuals[name].trainable_bytes
        for name in ("slow", "fast_capacity", "successor_shuffled_slow")
    }
    if len(set(trained_residual_bytes.values())) != 1:
        raise AssertionError("residual expert trainable allocations differ")
    return (
        base,
        residuals,
        global_counts,
        {
            "base_stage": {
                "passes": args.base_training_passes,
                "events": len(order) * args.base_training_passes,
                "prequential_log_loss_bits": base_loss
                / (len(order) * args.base_training_passes),
                "frozen_fingerprint": base_frozen_fingerprint,
            },
            "residual_stage": {
                "passes": args.residual_training_passes,
                "events": events,
                "prequential_log_loss_bits": {
                    name: value / events for name, value in totals.items()
                },
                "zero_logit_initialization": True,
                "fast_prior_frozen": True,
                "trainable_bytes": trained_residual_bytes,
                "trainable_allocation_matched": True,
                "fingerprints": {
                    name: selector.fingerprint() for name, selector in residuals.items()
                },
            },
            "order_sha256": _sha256(np.asarray(order, dtype="<i8").tobytes()),
            "successor_shuffle": shuffle_audit,
            "global_counts_sha256": surprise_assay._array_sha256(global_counts),
        },
    )


def _summarize(
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
    controls = (
        "fast_base",
        "nested_fast_fast",
        "global_frequency",
        "affinity",
        "nested_successor_shuffled_slow",
        "nested_frozen_random_slow",
    )
    for index, control in enumerate(controls):
        nested = pair_means("nested_fast_slow", "categorical_log_loss_bits")
        other = pair_means(control, "categorical_log_loss_bits")
        key = f"nested_fast_slow_minus_{control}_categorical_log_loss_bits"
        contrasts[key] = surprise_assay._mean_ci(
            [left - right for left, right in zip(nested, other, strict=True)],
            seed=seed + 10_000 + index,
            resamples=resamples,
        )
    for index, metric in enumerate(("mse", "proper_bits_per_coordinate")):
        nested = pair_means("nested_fast_slow", metric)
        basal = pair_means("basal", metric)
        contrasts[f"nested_fast_slow_minus_basal_{metric}"] = surprise_assay._mean_ci(
            [left - right for left, right in zip(nested, basal, strict=True)],
            seed=seed + 20_000 + index,
            resamples=resamples,
        )
    return {
        "pairs": len(pair_ids),
        "sides": len(selected),
        "arms": arm_summary,
        "paired_contrasts": contrasts,
    }


def evaluate_nested(
    basal: FutureTurnTrajectory,
    codec: codec_assay.FutureTurnCodec,
    pairs: Sequence[codec_assay.DevelopmentPair],
    fast_projector: surprise_assay.ContextProjector,
    slow_projector: surprise_assay.ContextProjector,
    learner: SurpriseGrownFutureTransition,
    base: CategoricalFutureModeSelector,
    residuals: Mapping[str, CategoricalFutureModeSelector],
    global_counts: np.ndarray,
    proper_variance: np.ndarray,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    before = {
        "basal": basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(learner),
        "base": base.fingerprint(),
        **{name: selector.fingerprint() for name, selector in residuals.items()},
    }
    context_results = {}
    for mode_index, mode in enumerate(
        ("original_unanswered_prefix", "long_interruption_crossover")
    ):
        examples = timescale_assay.development_examples_with_slow_state(
            basal,
            codec,
            pairs,
            mode=mode,
            slow_decay=args.slow_decay,
        )
        scorer = copy.deepcopy(learner)
        rows = []
        for pair, side_name, example, slow_raw in examples:
            fast = fast_projector.transform(example.context_state)
            slow = slow_projector.transform(slow_raw)
            surface = selector_assay.responsibility_surface(
                scorer, fast, example.basal_prediction, example.target
            )
            prior = base.predict(fast, surface.mode_ids).probabilities
            distributions = {
                "fast_base": selector_assay._selection_from_probabilities(
                    surface.mode_ids, prior
                ),
                "nested_fast_slow": residuals["slow"].predict_with_prior(
                    slow, surface.mode_ids, prior
                ),
                "nested_fast_fast": residuals["fast_capacity"].predict_with_prior(
                    fast, surface.mode_ids, prior
                ),
                "global_frequency": selector_assay._global_selection(
                    global_counts, surface.mode_ids
                ),
                "affinity": selector_assay._selection_from_probabilities(
                    surface.mode_ids, surface.affinity_probabilities
                ),
                "nested_successor_shuffled_slow": residuals[
                    "successor_shuffled_slow"
                ].predict_with_prior(slow, surface.mode_ids, prior),
                "nested_frozen_random_slow": residuals[
                    "frozen_random_slow"
                ].predict_with_prior(slow, surface.mode_ids, prior),
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
                partition: _summarize(
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
    after = {
        "basal": basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(learner),
        "base": base.fingerprint(),
        **{name: selector.fingerprint() for name, selector in residuals.items()},
    }
    if before != after or basal.observed_count != 0:
        raise AssertionError("evaluation modified frozen state")
    return {
        "context_modes": context_results,
        "fitted_fingerprints_before": before,
        "fitted_fingerprints_after": after,
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
    names = {
        "fast_base": "nested_fast_slow_minus_fast_base_categorical_log_loss_bits",
        "fast_capacity": (
            "nested_fast_slow_minus_nested_fast_fast_categorical_log_loss_bits"
        ),
        "global_frequency": (
            "nested_fast_slow_minus_global_frequency_categorical_log_loss_bits"
        ),
        "successor_shuffled": (
            "nested_fast_slow_minus_nested_successor_shuffled_slow_"
            "categorical_log_loss_bits"
        ),
    }
    comparisons = {name: contrasts[key] for name, key in names.items()}
    mse = contrasts["nested_fast_slow_minus_basal_mse"]
    proper = contrasts["nested_fast_slow_minus_basal_proper_bits_per_coordinate"]
    nested = arms["nested_fast_slow"]
    components = {
        "nested_slow_beats_fast_base": comparisons["fast_base"]["ci95_high"] < 0,
        "nested_slow_beats_fast_capacity_control": (
            comparisons["fast_capacity"]["ci95_high"] < 0
        ),
        "nested_slow_beats_global_frequency": (
            comparisons["global_frequency"]["ci95_high"] < 0
        ),
        "nested_slow_beats_successor_shuffled": (
            comparisons["successor_shuffled"]["ci95_high"] < 0
        ),
        "shortlist_size_at_most_locked_maximum": (
            nested["shortlist_size"]["mean"] <= args.shortlist_maximum
        ),
        "posterior_mass_at_least_locked_minimum": (
            nested["posterior_mass_captured"]["mean"] >= args.minimum_posterior_mass
        ),
        "argmax_recall_at_least_locked_minimum": (
            nested["posterior_argmax_recalled"]["mean"] >= args.minimum_argmax_recall
        ),
        "shortlisted_prediction_beats_basal": (
            mse["ci95_high"] < 0 or proper["ci95_high"] < 0
        ),
    }
    return {
        "scope": "long_interruption_crossover/evaluation_partition",
        "categorical_comparisons": comparisons,
        "mean_shortlist_size": nested["shortlist_size"]["mean"],
        "mean_posterior_mass_captured": nested["posterior_mass_captured"]["mean"],
        "posterior_argmax_recall": nested["posterior_argmax_recalled"]["mean"],
        "nested_fast_slow_minus_basal_mse": mse,
        "nested_fast_slow_minus_basal_proper_bits_per_coordinate": proper,
        "components": components,
        "incremental_slow_information_demonstrated": all(
            components[name]
            for name in (
                "nested_slow_beats_fast_base",
                "nested_slow_beats_fast_capacity_control",
                "nested_slow_beats_global_frequency",
                "nested_slow_beats_successor_shuffled",
            )
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
    slow_matrix = timescale_assay.training_slow_states(
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
    base, residuals, global_counts, selector_training = train_nested_selectors(
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
    residual_vectors = np.stack(
        [item.target - item.basal_prediction for item in assistant_examples]
    )
    proper_variance = np.maximum(
        np.mean(np.square(residual_vectors), axis=0), args.proper_variance_floor
    )
    evaluation = evaluate_nested(
        full_basal,
        codec,
        pairs,
        fast_projector,
        slow_projector,
        ordered,
        base,
        residuals,
        global_counts,
        proper_variance,
        args=args,
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit exceeded")
    result = {
        "schema_version": 1,
        "experiment": "future_mode_nested_slow_selector_latent_development",
        "status": "complete_development",
        "claim_scope": {
            "schema4_train_and_development_only": True,
            "native_alm_loaded": False,
            "free_byte_generation_run": False,
            "fast_selector_frozen_before_residual_training": True,
            "zero_residual_logits_exactly_reproduce_fast_prior": True,
            "target_arrives_after_causal_selection": True,
            "backpropagation_used": False,
            "backpropagation_through_time_used": False,
            "fixed_random_features_and_local_delta_readouts": True,
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
            "random_features_per_readout": args.nested_random_features,
            "base_training_passes": args.base_training_passes,
            "residual_training_passes": args.residual_training_passes,
            "base_learning_rate": args.base_learning_rate,
            "residual_learning_rate": args.nested_residual_learning_rate,
            "feature_scale": args.nested_feature_scale,
            "weight_decay": args.nested_weight_decay,
            "base_initial_readout_scale": args.base_initial_readout_scale,
            "residual_initial_readout_scale": 0.0,
            "base_seed": args.base_seed,
            "residual_seed": args.residual_seed,
            "order_seed": args.nested_order_seed,
            "shuffle_seed": args.nested_shuffle_seed,
            "random_control_seed": args.residual_random_control_seed,
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
    parser = timescale_assay._parser()
    parser.description = __doc__
    parser.add_argument("--nested-random-features", type=int, default=96)
    parser.add_argument("--nested-feature-scale", type=float, default=0.75)
    parser.add_argument("--base-training-passes", type=int, default=1)
    parser.add_argument("--residual-training-passes", type=int, default=1)
    parser.add_argument("--base-learning-rate", type=float, default=0.08)
    parser.add_argument(
        "--nested-residual-learning-rate", type=float, default=0.08
    )
    parser.add_argument("--nested-weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-initial-readout-scale", type=float, default=0.01)
    parser.add_argument("--base-seed", type=int, default=261_301)
    parser.add_argument("--residual-seed", type=int, default=261_302)
    parser.add_argument("--nested-order-seed", type=int, default=261_303)
    parser.add_argument("--nested-shuffle-seed", type=int, default=261_304)
    parser.add_argument("--residual-random-control-seed", type=int, default=261_305)
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
