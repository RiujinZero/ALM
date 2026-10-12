"""Development-only categorical gate over surprise-grown future modes.

The frozen surprise population proposes candidate next-turn vectors.  This
assay asks whether a cheap causal distribution learned from compressed
conversation state can shortlist the useful candidates before expensive
competition.  The later observed successor supplies soft posterior
responsibilities only after the categorical prediction has been scored.

No native ALM checkpoint or free-byte decoder is loaded.  The categorical
readout uses a single local delta rule with fixed random features; there is no
autograd, backpropagation, or BPTT.  PCA and ridge dependencies remain batch
estimators, so the assay does not claim a fully online-local system.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.categorical_future_mode_selector import (
    CategoricalFutureModeSelector,
    CategoricalFutureModeSelectorConfig,
    CategoricalSelection,
)
from manifoldflow.dendritic_basin.surprise_grown_future_transition import (
    SurpriseGrownFutureTransition,
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
    "src/manifoldflow/experiments/dendritic_basin_future_mode_selector_ablation.py",
    "tests/test_dendritic_basin_future_mode_selector_ablation.py",
    "src/manifoldflow/dendritic_basin/categorical_future_mode_selector.py",
    "tests/test_dendritic_basin_categorical_future_mode_selector.py",
    *surprise_assay.IMPLEMENTATION_PATHS,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    records: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(IMPLEMENTATION_PATHS):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        records[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return records


@dataclass(frozen=True)
class ResponsibilitySurface:
    """Frozen candidates and hindsight labels for one causal forecast."""

    mode_ids: tuple[int, ...]
    target_probabilities: np.ndarray
    affinity_probabilities: np.ndarray
    means: tuple[np.ndarray, ...]


def responsibility_surface(
    learner: SurpriseGrownFutureTransition,
    context: np.ndarray,
    basal_prediction: np.ndarray,
    target: np.ndarray,
) -> ResponsibilitySurface:
    """Forecast first, then expose target responsibilities without learning."""

    forecast = learner.forecast(context, basal_prediction, remember=True)
    update = learner.resolve(target, learn=False)
    mode_ids = tuple(candidate.mode_id for candidate in forecast.candidates)
    if mode_ids != update.scored_mode_ids:
        raise AssertionError("forecast candidates and scored responsibilities differ")
    target_probabilities = np.asarray(
        (update.unclaimed_responsibility, *update.responsibilities),
        dtype=np.float64,
    )
    affinity_probabilities = np.asarray(
        (forecast.basal_weight, *(item.weight for item in forecast.candidates)),
        dtype=np.float64,
    )
    for values in (target_probabilities, affinity_probabilities):
        values /= values.sum()
        values.setflags(write=False)
    means = (forecast.basal_prediction,) + tuple(
        candidate.prediction for candidate in forecast.candidates
    )
    return ResponsibilitySurface(
        mode_ids,
        target_probabilities,
        affinity_probabilities,
        means,
    )


def _selection_from_probabilities(
    mode_ids: tuple[int, ...], probabilities: np.ndarray
) -> CategoricalSelection:
    values = np.asarray(probabilities, dtype=np.float64)
    values = np.maximum(values, 1e-12)
    values /= values.sum()
    return CategoricalSelection(mode_ids, values)


def _global_selection(
    counts: np.ndarray, mode_ids: tuple[int, ...]
) -> CategoricalSelection:
    classes = np.asarray((0, *(item + 1 for item in mode_ids)), dtype=np.int64)
    return _selection_from_probabilities(mode_ids, counts[classes])


def _cross_entropy_bits(target: np.ndarray, prediction: np.ndarray) -> float:
    return -float(np.sum(target * np.log2(np.maximum(prediction, 1e-12))))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-15 else float(np.dot(left, right) / denominator)


def shortlist_indices(
    probabilities: np.ndarray, *, mass: float, maximum: int
) -> tuple[int, ...]:
    """Take the smallest probability-ranked set reaching mass, capped at K."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("shortlisting requires a nonempty probability vector")
    if not 0.0 < mass <= 1.0 or maximum <= 0:
        raise ValueError("shortlist mass and maximum are outside their ranges")
    order = np.argsort(-values, kind="stable")
    chosen: list[int] = []
    accumulated = 0.0
    for raw_index in order[:maximum]:
        index = int(raw_index)
        chosen.append(index)
        accumulated += float(values[index])
        if accumulated >= mass:
            break
    return tuple(chosen)


def selection_metrics(
    target: np.ndarray,
    surface: ResponsibilitySurface,
    selection: CategoricalSelection,
    *,
    shortlist_mass: float,
    shortlist_maximum: int,
    proper_variance: np.ndarray,
) -> dict[str, float]:
    if selection.mode_ids != surface.mode_ids:
        raise ValueError("selection and responsibility surface are misaligned")
    chosen = shortlist_indices(
        selection.probabilities,
        mass=shortlist_mass,
        maximum=shortlist_maximum,
    )
    weights = selection.probabilities[np.asarray(chosen, dtype=np.int64)]
    weights = weights / weights.sum()
    point = sum(
        (
            weight * surface.means[index]
            for weight, index in zip(weights, chosen, strict=True)
        ),
        np.zeros_like(target),
    )
    chosen_means = tuple(surface.means[index] for index in chosen)
    proper_bits = surprise_assay.gaussian_bits_per_coordinate(
        target,
        chosen_means,
        tuple(float(item) for item in weights),
        proper_variance,
    )
    oracle_class = int(np.argmax(surface.target_probabilities))
    return {
        "categorical_log_loss_bits": _cross_entropy_bits(
            surface.target_probabilities, selection.probabilities
        ),
        "shortlist_size": float(len(chosen)),
        "shortlist_fraction": float(len(chosen) / len(selection.probabilities)),
        "posterior_mass_captured": float(
            surface.target_probabilities[np.asarray(chosen, dtype=np.int64)].sum()
        ),
        "posterior_argmax_recalled": float(oracle_class in chosen),
        "mse": float(np.mean(np.square(target - point))),
        "cosine": _cosine(target, point),
        "proper_bits_per_coordinate": proper_bits,
    }


def _make_selector(
    args: argparse.Namespace, *, seed: int
) -> CategoricalFutureModeSelector:
    return CategoricalFutureModeSelector(
        CategoricalFutureModeSelectorConfig(
            context_dim=args.context_dimensions,
            mode_capacity=args.max_residual_modes,
            random_features=args.selector_random_features,
            feature_scale=args.selector_feature_scale,
            learning_rate=args.selector_learning_rate,
            weight_decay=args.selector_weight_decay,
            initial_readout_scale=args.selector_initial_readout_scale,
            probability_floor=args.selector_probability_floor,
            seed=seed,
        )
    )


def train_selectors(
    examples: Sequence[surprise_assay.TransitionExample],
    projector: surprise_assay.ContextProjector,
    learner: SurpriseGrownFutureTransition,
    *,
    conversations: int,
    args: argparse.Namespace,
) -> tuple[
    CategoricalFutureModeSelector,
    CategoricalFutureModeSelector,
    CategoricalFutureModeSelector,
    np.ndarray,
    dict[str, Any],
]:
    """Train actual and shuffled selectors, scoring every target pre-update."""

    learned = _make_selector(args, seed=args.selector_seed)
    shuffled = _make_selector(args, seed=args.selector_seed)
    frozen_random = _make_selector(args, seed=args.random_control_seed)
    global_counts = np.full(args.max_residual_modes + 1, args.global_prior)
    order = surprise_assay._training_order(
        examples, conversations=conversations, seed=args.selector_order_seed
    )
    permutation, shuffle_audit = surprise_assay.successor_target_permutation(
        examples, seed=args.selector_shuffle_seed
    )
    contexts = projector.transform(np.stack([item.context_state for item in examples]))
    actual_scorer = copy.deepcopy(learner)
    shuffled_scorer = copy.deepcopy(learner)
    totals = {
        name: {"log_loss": 0.0, "events": 0, "multicandidate": 0}
        for name in (
            "learned",
            "affinity",
            "global_frequency",
            "frozen_random",
            "successor_shuffled",
        )
    }
    for _pass in range(args.selector_training_passes):
        for raw_index in order:
            index = int(raw_index)
            example = examples[index]
            donor = examples[int(permutation[index])]
            context = contexts[index]
            actual = responsibility_surface(
                actual_scorer, context, example.basal_prediction, example.target
            )
            shuffled_target = responsibility_surface(
                shuffled_scorer, context, example.basal_prediction, donor.target
            )
            predictions = {
                "learned": learned.predict(context, actual.mode_ids).probabilities,
                "affinity": actual.affinity_probabilities,
                "global_frequency": _global_selection(
                    global_counts, actual.mode_ids
                ).probabilities,
                "frozen_random": frozen_random.predict(
                    context, actual.mode_ids
                ).probabilities,
                "successor_shuffled": shuffled.predict(
                    context, actual.mode_ids
                ).probabilities,
            }
            for name, prediction in predictions.items():
                totals[name]["log_loss"] += _cross_entropy_bits(
                    actual.target_probabilities, prediction
                )
                totals[name]["events"] += 1
                totals[name]["multicandidate"] += len(actual.mode_ids) >= 2
            learned.update(context, actual.mode_ids, actual.target_probabilities)
            shuffled.update(
                context,
                shuffled_target.mode_ids,
                shuffled_target.target_probabilities,
            )
            classes = np.asarray(
                (0, *(item + 1 for item in actual.mode_ids)), dtype=np.int64
            )
            global_counts[classes] += actual.target_probabilities
    training = {
        "target_timing": "each distribution scored before its successor target update",
        "training_passes": args.selector_training_passes,
        "order_sha256": _sha256(np.asarray(order, dtype="<i8").tobytes()),
        "selectors": {
            name: {
                "prequential_categorical_log_loss_bits": (
                    values["log_loss"] / values["events"]
                ),
                "events": values["events"],
                "events_with_two_or_more_modes": values["multicandidate"],
            }
            for name, values in totals.items()
        },
        "successor_shuffle": shuffle_audit,
        "learned_fingerprint": learned.fingerprint(),
        "successor_shuffled_fingerprint": shuffled.fingerprint(),
        "frozen_random_fingerprint": frozen_random.fingerprint(),
        "global_counts_sha256": surprise_assay._array_sha256(global_counts),
        "allocated_bytes_per_selector": learned.allocated_bytes,
    }
    return learned, shuffled, frozen_random, global_counts, training


def _mean_ci(values: Sequence[float], *, seed: int, resamples: int) -> dict[str, Any]:
    return surprise_assay._mean_ci(values, seed=seed, resamples=resamples)


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row["partition"] == partition]
    pair_ids = tuple(dict.fromkeys(row["pair_id"] for row in selected))
    arm_names = tuple(selected[0]["arms"]) if selected else ()
    metric_names = tuple(selected[0]["arms"][arm_names[0]]) if arm_names else ()

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

    arms = {
        arm: {
            metric: _mean_ci(
                pair_means(arm, metric),
                seed=seed + arm_index * 100 + metric_index,
                resamples=resamples,
            )
            for metric_index, metric in enumerate(metric_names)
        }
        for arm_index, arm in enumerate(arm_names)
    }
    controls = ("affinity", "global_frequency", "frozen_random", "successor_shuffled")
    contrasts: dict[str, Any] = {}
    for control_index, control in enumerate(controls):
        for metric_index, metric in enumerate(
            ("categorical_log_loss_bits", "mse", "proper_bits_per_coordinate")
        ):
            learned_values = pair_means("learned", metric)
            control_values = pair_means(control, metric)
            contrasts[f"learned_minus_{control}_{metric}"] = _mean_ci(
                [
                    left - right
                    for left, right in zip(learned_values, control_values, strict=True)
                ],
                seed=seed + 10_000 + control_index * 100 + metric_index,
                resamples=resamples,
            )
    for metric_index, metric in enumerate(("mse", "proper_bits_per_coordinate")):
        learned_values = pair_means("learned", metric)
        basal_values = pair_means("basal", metric)
        contrasts[f"learned_minus_basal_{metric}"] = _mean_ci(
            [
                left - right
                for left, right in zip(learned_values, basal_values, strict=True)
            ],
            seed=seed + 20_000 + metric_index,
            resamples=resamples,
        )
    return {
        "pairs": len(pair_ids),
        "sides": len(selected),
        "arms": arms,
        "paired_contrasts": contrasts,
    }


def evaluate_selectors(
    basal: recurrent_assay.FutureTurnTrajectory,
    codec: codec_assay.FutureTurnCodec,
    pairs: Sequence[codec_assay.DevelopmentPair],
    projector: surprise_assay.ContextProjector,
    learner: SurpriseGrownFutureTransition,
    learned: CategoricalFutureModeSelector,
    shuffled: CategoricalFutureModeSelector,
    frozen_random: CategoricalFutureModeSelector,
    global_counts: np.ndarray,
    *,
    proper_variance: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fingerprints_before = {
        "basal": basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(learner),
        "learned": learned.fingerprint(),
        "shuffled": shuffled.fingerprint(),
        "random": frozen_random.fingerprint(),
    }
    result: dict[str, Any] = {}
    for mode_index, mode in enumerate(
        ("original_unanswered_prefix", "long_interruption_crossover")
    ):
        examples = surprise_assay._development_examples(basal, codec, pairs, mode=mode)
        scorer = copy.deepcopy(learner)
        rows: list[dict[str, Any]] = []
        for pair, side_name, example in examples:
            context = projector.transform(example.context_state)
            surface = responsibility_surface(
                scorer, context, example.basal_prediction, example.target
            )
            selections = {
                "learned": learned.predict(context, surface.mode_ids),
                "affinity": _selection_from_probabilities(
                    surface.mode_ids, surface.affinity_probabilities
                ),
                "global_frequency": _global_selection(global_counts, surface.mode_ids),
                "frozen_random": frozen_random.predict(context, surface.mode_ids),
                "successor_shuffled": shuffled.predict(context, surface.mode_ids),
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
                "cosine": _cosine(example.target, example.basal_prediction),
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
                            name: selection_metrics(
                                example.target,
                                surface,
                                selection,
                                shortlist_mass=args.shortlist_mass,
                                shortlist_maximum=args.shortlist_maximum,
                                proper_variance=proper_variance,
                            )
                            for name, selection in selections.items()
                        },
                    },
                }
            )
        result[mode] = {
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
    fingerprints_after = {
        "basal": basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(learner),
        "learned": learned.fingerprint(),
        "shuffled": shuffled.fingerprint(),
        "random": frozen_random.fingerprint(),
    }
    if fingerprints_before != fingerprints_after or basal.observed_count != 0:
        raise AssertionError("evaluation modified a frozen predictor")
    return {
        "context_modes": result,
        "fitted_fingerprints_before": fingerprints_before,
        "fitted_fingerprints_after": fingerprints_after,
        "all_predictors_frozen_and_unchanged": True,
    }


def _development_gate(
    evaluation: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    summary = evaluation["context_modes"]["long_interruption_crossover"]["summaries"][
        "evaluation"
    ]
    arms = summary["arms"]
    contrasts = summary["paired_contrasts"]
    log_controls = (
        "affinity",
        "global_frequency",
        "frozen_random",
        "successor_shuffled",
    )
    log_wins = {
        control: contrasts[f"learned_minus_{control}_categorical_log_loss_bits"][
            "ci95_high"
        ]
        < 0.0
        for control in log_controls
    }
    mean_size = arms["learned"]["shortlist_size"]["mean"]
    mass = arms["learned"]["posterior_mass_captured"]["mean"]
    recall = arms["learned"]["posterior_argmax_recalled"]["mean"]
    mse = contrasts["learned_minus_basal_mse"]
    proper = contrasts["learned_minus_basal_proper_bits_per_coordinate"]
    components = {
        "categorical_log_loss_beats_all_controls": all(log_wins.values()),
        "mean_shortlist_size_at_most_locked_maximum": mean_size
        <= args.shortlist_maximum,
        "mean_posterior_mass_at_least_locked_minimum": mass
        >= args.minimum_posterior_mass,
        "posterior_argmax_recall_at_least_locked_minimum": recall
        >= args.minimum_argmax_recall,
        "shortlisted_prediction_beats_basal_on_mse_or_proper_score": mse["ci95_high"]
        < 0.0
        or proper["ci95_high"] < 0.0,
    }
    return {
        "scope": "long_interruption_crossover/evaluation_partition",
        "log_loss_control_wins": log_wins,
        "learned_mean_shortlist_size": mean_size,
        "learned_mean_posterior_mass_captured": mass,
        "learned_posterior_argmax_recall": recall,
        "learned_minus_basal_mse": mse,
        "learned_minus_basal_proper_bits_per_coordinate": proper,
        "components": components,
        "cheap_selector_advances_to_expensive_competition": all(components.values()),
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
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
    context_matrix = np.stack([item.context_state for item in crossfit.examples])
    projector = surprise_assay.ContextProjector.fit(
        context_matrix,
        dimensions=args.context_dimensions,
        scale_floor=args.context_scale_floor,
        clip=args.context_clip,
    )
    ordered, _unused_shuffled, overlay = surprise_assay.train_surprise_arms(
        crossfit.examples,
        projector,
        conversations=len(conversations),
        args=args,
    )
    learned, shuffled, frozen_random, global_counts, selector_training = (
        train_selectors(
            crossfit.examples,
            projector,
            ordered,
            conversations=len(conversations),
            args=args,
        )
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
    evaluation = evaluate_selectors(
        full_basal,
        codec,
        pairs,
        projector,
        ordered,
        learned,
        shuffled,
        frozen_random,
        global_counts,
        proper_variance=proper_variance,
        args=args,
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit exceeded")
    result = {
        "schema_version": 1,
        "experiment": "categorical_future_mode_selector_latent_development",
        "status": "complete_smoke" if args.smoke else "complete_development",
        "claim_scope": {
            "schema4_train_and_development_only": True,
            "native_alm_loaded": False,
            "free_byte_generation_run": False,
            "surprise_population_frozen_before_selector_training": True,
            "selector_target_arrives_after_causal_prediction": True,
            "autograd_framework_used": False,
            "backpropagation_used": False,
            "backpropagation_through_time_used": False,
            "single_layer_local_delta_rule": True,
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
            "raw_dimensions": args.raw_dimensions,
            "content_dimensions": args.content_dimensions,
            "active_content_features": args.active_content_features,
            "memory_decay": args.memory_decay,
            "transition_ridge": args.transition_ridge,
            "crossfit_folds": args.crossfit_folds,
            "crossfit_seed": args.crossfit_seed,
            "context_dimensions": args.context_dimensions,
            "selector_training_passes": args.selector_training_passes,
            "selector_random_features": args.selector_random_features,
            "selector_feature_scale": args.selector_feature_scale,
            "selector_learning_rate": args.selector_learning_rate,
            "selector_weight_decay": args.selector_weight_decay,
            "selector_initial_readout_scale": args.selector_initial_readout_scale,
            "selector_probability_floor": args.selector_probability_floor,
            "selector_seed": args.selector_seed,
            "selector_order_seed": args.selector_order_seed,
            "selector_shuffle_seed": args.selector_shuffle_seed,
            "random_control_seed": args.random_control_seed,
            "global_prior": args.global_prior,
            "shortlist_mass": args.shortlist_mass,
            "shortlist_maximum": args.shortlist_maximum,
            "minimum_posterior_mass": args.minimum_posterior_mass,
            "minimum_argmax_recall": args.minimum_argmax_recall,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "training_transitions": len(crossfit.examples),
            "completed_training_assistant_turns": len(assistant_turns),
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
        },
        "cross_fitted_basal": crossfit.audit,
        "context_projector": {
            "fingerprint": projector.fingerprint(),
            "development_rows_used": 0,
        },
        "surprise_population_training": overlay,
        "selector_training": selector_training,
        "evaluation": evaluation,
        "development_gate": _development_gate(evaluation, args),
        "wall_seconds": elapsed,
    }
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = surprise_assay._parser()
    parser.description = __doc__
    parser.add_argument("--selector-training-passes", type=int, default=1)
    parser.add_argument("--selector-random-features", type=int, default=96)
    parser.add_argument("--selector-feature-scale", type=float, default=0.75)
    parser.add_argument("--selector-learning-rate", type=float, default=0.08)
    parser.add_argument("--selector-weight-decay", type=float, default=1e-5)
    parser.add_argument("--selector-initial-readout-scale", type=float, default=0.01)
    parser.add_argument("--selector-probability-floor", type=float, default=1e-9)
    parser.add_argument("--selector-seed", type=int, default=261_101)
    parser.add_argument("--selector-order-seed", type=int, default=261_102)
    parser.add_argument("--selector-shuffle-seed", type=int, default=261_103)
    parser.add_argument("--random-control-seed", type=int, default=261_104)
    parser.add_argument("--global-prior", type=float, default=1.0)
    parser.add_argument("--shortlist-mass", type=float, default=0.80)
    parser.add_argument("--shortlist-maximum", type=int, default=8)
    parser.add_argument("--minimum-posterior-mass", type=float, default=0.75)
    parser.add_argument("--minimum-argmax-recall", type=float, default=0.75)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_ablation(args)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
