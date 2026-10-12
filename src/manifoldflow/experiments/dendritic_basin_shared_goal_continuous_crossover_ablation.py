"""Exact crossover assay for a signed continuous shared-goal controller.

The corpus adapter in ``dendritic_basin_shared_goal_crossover_ablation`` owns
the definition of a future target.  This experiment consumes those vectors
verbatim: it neither clusters nor normalizes them.  Three deterministic ridge
maps predict the target from full causal evidence, the pending event alone,
or full evidence paired with Sattolo-shuffled training targets.  Raw predictor
magnitude is retained and reported; ``MatchedSignedGoalByteHead`` applies its
own configured safety clamp at the byte-action boundary.

As in the discrete preregistration, every condition is a per-pair
difference-in-differences against goal-off.  Goal-off supplies a zero vector at
the selected nonzero beta, which must reproduce both q0 and the base trace
bit-exactly.  The frozen native cortex may itself retain earlier context; that
raw residual is reported rather than mistaken for a goal effect.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.goal_conditioned_byte_head import (
    MatchedSignedGoalByteHead,
    SignedEpisodeByteTrace,
    shuffled_signed_goal_control,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    CrossoverPair,
    TrainingExample,
    _load_native,
    _native_trace,
    _summary,
    load_crossover_pairs,
    load_training_examples,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    DEFAULT_SUCCESSOR_GAMMAS,
    RidgeMap,
    fit_ridge,
)


@dataclass(frozen=True)
class ContinuousMechanism:
    full_predictor: RidgeMap
    pending_predictor: RidgeMap
    shuffled_target_predictor: RidgeMap
    action_head: MatchedSignedGoalByteHead
    shuffled_action_head: MatchedSignedGoalByteHead
    prior_goal: np.ndarray
    target_dimensions: int
    predictive_target_permutation: np.ndarray
    action_target_permutation: np.ndarray


@dataclass(frozen=True)
class PairGoals:
    oracle_a: np.ndarray
    oracle_b: np.ndarray
    causal_full_a: np.ndarray
    causal_full_b: np.ndarray
    causal_pending_a: np.ndarray
    causal_pending_b: np.ndarray
    shuffled_predictor_a: np.ndarray
    shuffled_predictor_b: np.ndarray
    shuffled_episode_a: np.ndarray
    shuffled_episode_b: np.ndarray


@dataclass(frozen=True)
class PreparedPair:
    pair: CrossoverPair
    base_traces: dict[str, np.ndarray]


def _sattolo_indices(count: int, seed: int) -> np.ndarray:
    """Return a deterministic no-fixed-point permutation for ``count > 1``."""

    permutation = np.arange(count, dtype=np.int64)
    if count > 1:
        rng = np.random.default_rng(seed)
        for index in range(count - 1, 0, -1):
            other = int(rng.integers(0, index))
            permutation[index], permutation[other] = (
                permutation[other],
                permutation[index],
            )
    return permutation


def _training_arrays(
    examples: list[TrainingExample], feature_dimensions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full = np.asarray([row.evidence for row in examples], dtype=np.float64)
    targets = np.asarray([row.goal_target_vector for row in examples], dtype=np.float64)
    if full.ndim != 2 or targets.ndim != 2:
        raise ValueError("training evidence and goal targets must be rectangular")
    if full.shape[0] != targets.shape[0] or full.shape[0] < 2:
        raise ValueError("training evidence and targets must have matching rows")
    if feature_dimensions < 1 or full.shape[1] < feature_dimensions:
        raise ValueError("pending-event feature width exceeds causal evidence")
    if not np.all(np.isfinite(full)) or not np.all(np.isfinite(targets)):
        raise ValueError("training evidence and targets must be finite")
    return full, full[:, :feature_dimensions], targets


def fit_continuous_mechanism(
    examples: list[TrainingExample],
    *,
    feature_dimensions: int,
    ridge_alpha: float,
    context_order: int,
    context_buckets: int,
    association_buckets: int,
    local_prior_mass: float,
    association_prior_mass: float,
    goal_gain: float,
    max_goal_norm: float,
    goal_logit_clip: float,
    minimum_probability: float,
    control_seed: int,
) -> ContinuousMechanism:
    """Fit deterministic continuous predictors and real/shuffled action heads."""

    full, pending, targets = _training_arrays(examples, feature_dimensions)
    target_dimensions = int(targets.shape[1])
    predictive_permutation = _sattolo_indices(len(targets), control_seed + 101)
    full_predictor = fit_ridge(full, targets, ridge_alpha)
    pending_predictor = fit_ridge(pending, targets, ridge_alpha)
    shuffled_target_predictor = fit_ridge(
        full, targets[predictive_permutation], ridge_alpha
    )

    def new_head() -> MatchedSignedGoalByteHead:
        return MatchedSignedGoalByteHead(
            target_dimensions,
            context_order=context_order,
            context_bucket_count=context_buckets,
            association_bucket_count=association_buckets,
            local_prior_mass=local_prior_mass,
            association_prior_mass=association_prior_mass,
            goal_gain=goal_gain,
            max_goal_norm=max_goal_norm,
            logit_clip=goal_logit_clip,
            minimum_probability=minimum_probability,
        )

    action_head = new_head()
    episodes = [
        SignedEpisodeByteTrace(
            response=row.response,
            goal_vector=row.goal_target_vector,
            prefix=row.prefix,
        )
        for row in examples
    ]
    action_head.fit_episodes(episodes)
    shuffled_episodes, action_permutation = shuffled_signed_goal_control(
        episodes, seed=control_seed + 211
    )
    shuffled_action_head = new_head()
    shuffled_action_head.fit_episodes(shuffled_episodes)
    return ContinuousMechanism(
        full_predictor=full_predictor,
        pending_predictor=pending_predictor,
        shuffled_target_predictor=shuffled_target_predictor,
        action_head=action_head,
        shuffled_action_head=shuffled_action_head,
        prior_goal=np.mean(targets, axis=0),
        target_dimensions=target_dimensions,
        predictive_target_permutation=predictive_permutation,
        action_target_permutation=action_permutation,
    )


def _pair_target_matrix(pairs: list[CrossoverPair]) -> np.ndarray:
    targets: list[np.ndarray] = []
    for pair in pairs:
        if pair.goal_target_vector_a is None or pair.goal_target_vector_b is None:
            raise ValueError(
                "continuous assay requires the goal-target payload adapter output"
            )
        targets.extend((pair.goal_target_vector_a, pair.goal_target_vector_b))
    matrix = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("evaluation goal targets must be finite and rectangular")
    return matrix


def predict_pair_goals(
    pairs: list[CrossoverPair],
    mechanism: ContinuousMechanism,
    *,
    feature_dimensions: int,
    control_seed: int,
) -> tuple[list[PairGoals], dict[str, np.ndarray]]:
    """Predict raw continuous goals; no norm or direction is altered."""

    evidence = np.asarray(
        [vector for pair in pairs for vector in (pair.evidence_a, pair.evidence_b)],
        dtype=np.float64,
    )
    targets = _pair_target_matrix(pairs)
    if targets.shape[1] != mechanism.target_dimensions:
        raise ValueError("evaluation and training goal dimensions differ")
    if evidence.ndim != 2 or evidence.shape[1] < feature_dimensions:
        raise ValueError("evaluation evidence has an invalid feature width")
    full = mechanism.full_predictor.predict(evidence)
    pending = mechanism.pending_predictor.predict(evidence[:, :feature_dimensions])
    shuffled_predictor = mechanism.shuffled_target_predictor.predict(evidence)
    episode_permutation = _sattolo_indices(len(full), control_seed)
    shuffled_episode = full[episode_permutation]
    arrays = {
        "target": targets,
        "causal_full": full,
        "causal_pending_only": pending,
        "shuffled_predictive_target": shuffled_predictor,
        "shuffled_episode_goal": shuffled_episode,
    }
    rows = [
        PairGoals(
            oracle_a=targets[2 * index],
            oracle_b=targets[2 * index + 1],
            causal_full_a=full[2 * index],
            causal_full_b=full[2 * index + 1],
            causal_pending_a=pending[2 * index],
            causal_pending_b=pending[2 * index + 1],
            shuffled_predictor_a=shuffled_predictor[2 * index],
            shuffled_predictor_b=shuffled_predictor[2 * index + 1],
            shuffled_episode_a=shuffled_episode[2 * index],
            shuffled_episode_b=shuffled_episode[2 * index + 1],
        )
        for index in range(len(pairs))
    ]
    return rows, arrays


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0.0 else float((left @ right) / denominator)


def _representation_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    max_goal_norm: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if prediction.shape != target.shape or len(prediction) % 2:
        raise ValueError("representation rows must be aligned crossover pairs")
    correct_cosines = np.asarray(
        [_cosine(prediction[index], target[index]) for index in range(len(target))]
    )
    margins = []
    for index in range(0, len(target), 2):
        margins.append(
            _cosine(prediction[index], target[index])
            + _cosine(prediction[index + 1], target[index + 1])
            - _cosine(prediction[index], target[index + 1])
            - _cosine(prediction[index + 1], target[index])
        )
    margins = np.asarray(margins, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(
        margins[rng.integers(0, len(margins), size=(resamples, len(margins)))],
        axis=1,
    )
    prediction_norms = np.linalg.norm(prediction, axis=1)
    target_norms = np.linalg.norm(target, axis=1)
    return {
        "pairs": len(margins),
        "mean_correct_target_cosine": float(np.mean(correct_cosines)),
        "mean_paired_cosine_margin": float(np.mean(margins)),
        "median_paired_cosine_margin": float(np.median(margins)),
        "paired_cosine_win_fraction": float(np.mean(margins > 0.0)),
        "mean_paired_cosine_margin_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "mean_paired_cosine_margin_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "prediction_norm": {
            "mean": float(np.mean(prediction_norms)),
            "median": float(np.median(prediction_norms)),
            "minimum": float(np.min(prediction_norms)),
            "maximum": float(np.max(prediction_norms)),
            "fraction_above_head_clamp": float(
                np.mean(prediction_norms > max_goal_norm)
            ),
        },
        "target_norm": {
            "mean": float(np.mean(target_norms)),
            "median": float(np.median(target_norms)),
            "minimum": float(np.min(target_norms)),
            "maximum": float(np.max(target_norms)),
        },
    }


def _corrected_probability_trace(
    base: np.ndarray,
    local: np.ndarray,
    goal: np.ndarray,
    *,
    beta: float,
    log_ratio_clip: float,
) -> np.ndarray:
    if beta == 0.0 or np.array_equal(local, goal):
        return np.array(base, copy=True)
    ratio = np.clip(np.log(goal) - np.log(local), -log_ratio_clip, log_ratio_clip)
    logits = np.log(np.maximum(base, 1e-300)) + beta * ratio
    logits -= np.max(logits, axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def _trace_log_score(probability: np.ndarray, candidate: bytes) -> float:
    targets = np.frombuffer(candidate, dtype=np.uint8)
    return float(
        np.log2(np.maximum(probability[np.arange(len(targets)), targets], 1e-300)).sum()
    )


def _condition_margin(
    prepared: PreparedPair,
    head: MatchedSignedGoalByteHead,
    goal_a: np.ndarray,
    goal_b: np.ndarray,
    *,
    beta: float,
    log_ratio_clip: float,
) -> float:
    pair = prepared.pair
    scores: dict[str, float] = {}
    for context_name, context in (("a", pair.context_a), ("b", pair.context_b)):
        for response_name, response in (
            ("a", pair.response_a),
            ("b", pair.response_b),
        ):
            goal_vector = goal_a if context_name == "a" else goal_b
            local, conditioned = head.trace_probabilities(
                context, response, goal_vector
            )
            probability = _corrected_probability_trace(
                prepared.base_traces[context_name + response_name],
                local,
                conditioned,
                beta=beta,
                log_ratio_clip=log_ratio_clip,
            )
            scores[context_name + response_name] = _trace_log_score(
                probability, response
            )
    return (scores["aa"] - scores["ba"]) + (scores["bb"] - scores["ab"])


def _prepare_native_pairs(
    pairs: list[CrossoverPair], cortex: object
) -> list[PreparedPair]:
    prepared = []
    for pair in pairs:
        base = {
            context_name + response_name: _native_trace(cortex, context, response)
            for context_name, context in (("a", pair.context_a), ("b", pair.context_b))
            for response_name, response in (
                ("a", pair.response_a),
                ("b", pair.response_b),
            )
        }
        prepared.append(PreparedPair(pair, base))
    return prepared


def _base_pair_margin(prepared: PreparedPair) -> float:
    pair = prepared.pair
    scores = {
        "aa": _trace_log_score(prepared.base_traces["aa"], pair.response_a),
        "ab": _trace_log_score(prepared.base_traces["ab"], pair.response_b),
        "ba": _trace_log_score(prepared.base_traces["ba"], pair.response_a),
        "bb": _trace_log_score(prepared.base_traces["bb"], pair.response_b),
    }
    return (scores["aa"] - scores["ba"]) + (scores["bb"] - scores["ab"])


def _select_beta_on_dev_oracle(
    prepared_pairs: list[PreparedPair],
    pair_goals: list[PairGoals],
    head: MatchedSignedGoalByteHead,
    *,
    beta_values: tuple[float, ...],
    log_ratio_clip: float,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Select beta using oracle only, computing each qg trace once per pair."""

    if not beta_values:
        raise ValueError("at least one beta value is required")
    effects = {beta: [] for beta in beta_values}
    goal_off_residuals = []
    for prepared, goals in zip(prepared_pairs, pair_goals, strict=True):
        pair = prepared.pair
        traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for context_name, context, goal_vector in (
            ("a", pair.context_a, goals.oracle_a),
            ("b", pair.context_b, goals.oracle_b),
        ):
            for response_name, response in (
                ("a", pair.response_a),
                ("b", pair.response_b),
            ):
                traces[context_name + response_name] = head.trace_probabilities(
                    context, response, goal_vector
                )
        goal_off = _base_pair_margin(prepared)
        goal_off_residuals.append(goal_off)
        for beta in beta_values:
            scores = {}
            for key, response in (
                ("aa", pair.response_a),
                ("ab", pair.response_b),
                ("ba", pair.response_a),
                ("bb", pair.response_b),
            ):
                local, goal = traces[key]
                probability = _corrected_probability_trace(
                    prepared.base_traces[key],
                    local,
                    goal,
                    beta=beta,
                    log_ratio_clip=log_ratio_clip,
                )
                scores[key] = _trace_log_score(probability, response)
            raw_margin = (scores["aa"] - scores["ba"]) + (scores["bb"] - scores["ab"])
            effects[beta].append(raw_margin - goal_off)

    goal_off_mean = float(np.mean(goal_off_residuals))
    curve = {}
    selected: tuple[float, float] | None = None
    for beta in beta_values:
        values = np.asarray(effects[beta], dtype=np.float64)
        mean_effect = float(np.mean(values))
        curve[str(beta)] = {
            "oracle_adjusted_mean_effect_bits": mean_effect,
            "oracle_adjusted_win_fraction": float(np.mean(values > 0.0)),
            "goal_off_raw_mean_residual_bits": goal_off_mean,
        }
        key = (mean_effect, -beta)
        if selected is None or key > (selected[0], -selected[1]):
            selected = (mean_effect, beta)
    assert selected is not None
    return selected[1], curve


def _evaluate_conditions(
    prepared_pairs: list[PreparedPair],
    pair_goals: list[PairGoals],
    mechanism: ContinuousMechanism,
    *,
    beta: float,
    log_ratio_clip: float,
    require_exact_native: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    names = (
        "goal_off",
        "oracle_future",
        "causal_full",
        "causal_pending_only",
        "shuffled_predictive_target",
        "swapped_oracle",
        "swapped_causal",
        "prior_only",
        "shuffled_action_mapping",
        "shuffled_episode_goal",
    )
    margins: dict[str, list[float]] = {name: [] for name in names}
    zero = np.zeros(mechanism.target_dimensions, dtype=np.float64)
    max_local_difference = 0.0
    max_goal_off_base_difference = 0.0
    max_cross_context_base_difference = 0.0

    for prepared, goals in zip(prepared_pairs, pair_goals, strict=True):
        pair = prepared.pair
        for response_name, response in (("a", pair.response_a), ("b", pair.response_b)):
            local_a, off_a = mechanism.action_head.trace_probabilities(
                pair.context_a, response, zero
            )
            local_b, off_b = mechanism.action_head.trace_probabilities(
                pair.context_b, response, zero
            )
            max_local_difference = max(
                max_local_difference, float(np.max(np.abs(local_a - local_b)))
            )
            for context_name, local, off in (
                ("a", local_a, off_a),
                ("b", local_b, off_b),
            ):
                base = prepared.base_traces[context_name + response_name]
                corrected = _corrected_probability_trace(
                    base,
                    local,
                    off,
                    beta=beta,
                    log_ratio_clip=log_ratio_clip,
                )
                max_goal_off_base_difference = max(
                    max_goal_off_base_difference,
                    float(np.max(np.abs(corrected - base))),
                )
            max_cross_context_base_difference = max(
                max_cross_context_base_difference,
                float(
                    np.max(
                        np.abs(
                            prepared.base_traces["a" + response_name]
                            - prepared.base_traces["b" + response_name]
                        )
                    )
                ),
            )

        assignments = {
            "goal_off": (zero, zero, mechanism.action_head),
            "oracle_future": (
                goals.oracle_a,
                goals.oracle_b,
                mechanism.action_head,
            ),
            "causal_full": (
                goals.causal_full_a,
                goals.causal_full_b,
                mechanism.action_head,
            ),
            "causal_pending_only": (
                goals.causal_pending_a,
                goals.causal_pending_b,
                mechanism.action_head,
            ),
            "shuffled_predictive_target": (
                goals.shuffled_predictor_a,
                goals.shuffled_predictor_b,
                mechanism.action_head,
            ),
            "swapped_oracle": (
                goals.oracle_b,
                goals.oracle_a,
                mechanism.action_head,
            ),
            "swapped_causal": (
                goals.causal_full_b,
                goals.causal_full_a,
                mechanism.action_head,
            ),
            "prior_only": (
                mechanism.prior_goal,
                mechanism.prior_goal,
                mechanism.action_head,
            ),
            "shuffled_action_mapping": (
                goals.oracle_a,
                goals.oracle_b,
                mechanism.shuffled_action_head,
            ),
            "shuffled_episode_goal": (
                goals.shuffled_episode_a,
                goals.shuffled_episode_b,
                mechanism.action_head,
            ),
        }
        for name, (goal_a, goal_b, head) in assignments.items():
            margins[name].append(
                _condition_margin(
                    prepared,
                    head,
                    goal_a,
                    goal_b,
                    beta=beta,
                    log_ratio_clip=log_ratio_clip,
                )
            )

    raw = {
        name: np.asarray(values, dtype=np.float64) for name, values in margins.items()
    }
    adjusted = {name: values - raw["goal_off"] for name, values in raw.items()}
    invariants = {
        "max_abs_local_q0_cross_context_trace_difference": max_local_difference,
        "exact_local_q0_cross_context_trace_identity": max_local_difference == 0.0,
        "max_abs_goal_off_minus_base_trace_difference": (max_goal_off_base_difference),
        "exact_goal_off_base_trace_identity": max_goal_off_base_difference == 0.0,
        "max_abs_native_base_cross_context_trace_difference": (
            max_cross_context_base_difference
        ),
        "exact_native_base_cross_context_trace_identity": (
            max_cross_context_base_difference == 0.0
        ),
        "native_cross_context_identity_required": require_exact_native,
    }
    if max_local_difference != 0.0:
        raise AssertionError("continuous head local q0 traces were not identical")
    if max_goal_off_base_difference != 0.0:
        raise AssertionError("zero continuous goal did not reproduce the base trace")
    if require_exact_native and max_cross_context_base_difference != 0.0:
        raise AssertionError("native base retained context before the common suffix")
    return raw, adjusted, invariants


def _action_report(
    raw: dict[str, np.ndarray],
    adjusted: dict[str, np.ndarray],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    return {
        "effect_definition": (
            "per-pair condition margin minus that pair's zero-vector goal_off "
            "raw margin"
        ),
        "raw_goal_off_residual": _summary(
            raw["goal_off"], seed=seed + 90_001, resamples=resamples
        ),
        "conditions": {
            name: {
                "adjusted_effect_vs_goal_off": _summary(
                    adjusted[name], seed=seed + index * 101, resamples=resamples
                ),
                "raw_pairing_margin": _summary(
                    raw[name], seed=seed + index * 101 + 47, resamples=resamples
                ),
            }
            for index, name in enumerate(raw)
        },
    }


def _slice_pairs(pairs: list[CrossoverPair], maximum: int) -> list[CrossoverPair]:
    return pairs if maximum == 0 else pairs[:maximum]


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    decays = tuple(float(value) for value in args.state_decays)
    gammas = tuple(float(value) for value in args.successor_gammas)
    examples = load_training_examples(
        args.dataset_dir,
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=gammas,
    )
    dev_pairs = _slice_pairs(
        load_crossover_pairs(
            args.dataset_dir,
            "dev",
            dimensions=args.feature_dimensions,
            decays=decays,
            successor_gammas=gammas,
        ),
        args.max_dev_pairs,
    )
    test_pairs = _slice_pairs(
        load_crossover_pairs(
            args.dataset_dir,
            "test",
            dimensions=args.feature_dimensions,
            decays=decays,
            successor_gammas=gammas,
        ),
        args.max_test_pairs,
    )
    mechanism = fit_continuous_mechanism(
        examples,
        feature_dimensions=args.feature_dimensions,
        ridge_alpha=args.ridge_alpha,
        context_order=args.context_order,
        context_buckets=args.context_buckets,
        association_buckets=args.association_buckets,
        local_prior_mass=args.local_prior_mass,
        association_prior_mass=args.association_prior_mass,
        goal_gain=args.goal_gain,
        max_goal_norm=args.max_goal_norm,
        goal_logit_clip=args.goal_logit_clip,
        minimum_probability=args.minimum_probability,
        control_seed=args.control_seed,
    )
    dev_goals, dev_arrays = predict_pair_goals(
        dev_pairs,
        mechanism,
        feature_dimensions=args.feature_dimensions,
        control_seed=args.control_seed + 1_009,
    )
    test_goals, test_arrays = predict_pair_goals(
        test_pairs,
        mechanism,
        feature_dimensions=args.feature_dimensions,
        control_seed=args.control_seed + 2_009,
    )
    cortex = _load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("continuous crossover requires a frozen native checkpoint")
    prepared_dev = _prepare_native_pairs(dev_pairs, cortex)
    selected_beta, dev_curve = _select_beta_on_dev_oracle(
        prepared_dev,
        dev_goals,
        mechanism.action_head,
        beta_values=tuple(float(value) for value in args.beta_values),
        log_ratio_clip=args.log_ratio_clip,
    )
    dev_raw, dev_adjusted, dev_invariants = _evaluate_conditions(
        prepared_dev,
        dev_goals,
        mechanism,
        beta=selected_beta,
        log_ratio_clip=args.log_ratio_clip,
        require_exact_native=args.require_exact_native,
    )
    del prepared_dev
    prepared_test = _prepare_native_pairs(test_pairs, cortex)
    test_raw, test_adjusted, test_invariants = _evaluate_conditions(
        prepared_test,
        test_goals,
        mechanism,
        beta=selected_beta,
        log_ratio_clip=args.log_ratio_clip,
        require_exact_native=args.require_exact_native,
    )

    def representation(arrays: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
        return {
            name: _representation_summary(
                arrays[name],
                arrays["target"],
                max_goal_norm=args.max_goal_norm,
                seed=seed + index * 307,
                resamples=args.resamples,
            )
            for index, name in enumerate(
                (
                    "causal_full",
                    "causal_pending_only",
                    "shuffled_predictive_target",
                )
            )
        }

    validity = all(
        bool(invariants[check])
        for invariants in (dev_invariants, test_invariants)
        for check in (
            "exact_local_q0_cross_context_trace_identity",
            "exact_goal_off_base_trace_identity",
        )
    )
    dev_representation = representation(dev_arrays, args.control_seed + 10_003)
    test_representation = representation(test_arrays, args.control_seed + 20_003)
    return {
        "experiment": "shared_goal_signed_continuous_exact_crossover_ablation",
        "status": "complete" if validity else "invalid_invariant",
        "checkpoint": str(args.checkpoint),
        "claim_scope": (
            "continuous future-target compatibility on exact crossover pairs; "
            "autonomous goal pursuit requires a separate generation gate"
        ),
        "fit_determinism": {
            "feature_hash_is_fixed": True,
            "ridge_and_action_fit_are_deterministic": True,
            "independent_fit_seeds_used": False,
            "control_seed_affects_only_sattolo_controls_and_resampling": True,
        },
        "target_contract": {
            "kind": dev_pairs[0].goal_target_kind,
            "dimensions": mechanism.target_dimensions,
            "payload_adapter_output_used_verbatim": True,
            "predicted_vectors_normalized_before_action_head": False,
            "head_applies_configured_norm_clamp": True,
        },
        "config": {
            "dataset_dir": str(args.dataset_dir),
            "feature_dimensions": args.feature_dimensions,
            "state_decays": list(decays),
            "successor_gammas": list(gammas),
            "ridge_alpha": args.ridge_alpha,
            "context_order": args.context_order,
            "context_buckets": args.context_buckets,
            "association_buckets": args.association_buckets,
            "local_prior_mass": args.local_prior_mass,
            "association_prior_mass": args.association_prior_mass,
            "goal_gain": args.goal_gain,
            "max_goal_norm": args.max_goal_norm,
            "goal_logit_clip": args.goal_logit_clip,
            "minimum_probability": args.minimum_probability,
            "beta_values": list(args.beta_values),
            "selected_beta": selected_beta,
            "log_ratio_clip": args.log_ratio_clip,
            "control_seed": args.control_seed,
            "resamples": args.resamples,
            "max_dev_pairs": args.max_dev_pairs,
            "max_test_pairs": args.max_test_pairs,
            "require_exact_native": args.require_exact_native,
        },
        "counts": {
            "train_episodes": len(examples),
            "dev_pairs": len(dev_pairs),
            "test_pairs": len(test_pairs),
            "action_head_persistent_bytes_each": (
                mechanism.action_head.allocated_bytes
            ),
            "real_and_shuffled_action_heads_persistent_bytes": (
                mechanism.action_head.allocated_bytes
                + mechanism.shuffled_action_head.allocated_bytes
            ),
            "real_action_head_transient_cache_bytes_at_report": (
                mechanism.action_head.transient_cache_bytes
            ),
            "shuffled_action_head_transient_cache_bytes_at_report": (
                mechanism.shuffled_action_head.transient_cache_bytes
            ),
        },
        "control_permutations": {
            "predictive_training_target_fixed_points": int(
                np.sum(
                    mechanism.predictive_target_permutation == np.arange(len(examples))
                )
            ),
            "action_training_target_fixed_points": int(
                np.sum(mechanism.action_target_permutation == np.arange(len(examples)))
            ),
        },
        "dev_beta_curve_oracle_selection_only": dev_curve,
        "representation": {
            "dev": dev_representation,
            "test": test_representation,
        },
        "invariants": {"dev": dev_invariants, "test": test_invariants},
        "action": {
            "dev_at_selected_beta": _action_report(
                dev_raw,
                dev_adjusted,
                seed=args.control_seed + 30_007,
                resamples=args.resamples,
            ),
            "test_at_frozen_beta": _action_report(
                test_raw,
                test_adjusted,
                seed=args.control_seed + 40_009,
                resamples=args.resamples,
            ),
        },
        "gate": {
            "validity_invariants_passed": validity,
            "test_oracle_adjusted_positive": bool(
                np.mean(test_adjusted["oracle_future"]) > 0.0
            ),
            "test_swapped_oracle_adjusted_reverses": bool(
                np.mean(test_adjusted["swapped_oracle"]) < 0.0
            ),
            "test_causal_full_adjusted_positive": bool(
                np.mean(test_adjusted["causal_full"]) > 0.0
            ),
            "test_full_representation_beats_pending_margin": bool(
                test_representation["causal_full"]["mean_paired_cosine_margin"]
                > test_representation["causal_pending_only"][
                    "mean_paired_cosine_margin"
                ]
            ),
        },
        "wall_seconds": time.perf_counter() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--feature-dimensions", type=int, default=96)
    parser.add_argument(
        "--state-decays", type=float, nargs="+", default=(0.5, 0.85, 0.97)
    )
    parser.add_argument(
        "--successor-gammas",
        type=float,
        nargs="+",
        default=DEFAULT_SUCCESSOR_GAMMAS,
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--context-order", type=int, default=4)
    parser.add_argument("--context-buckets", type=int, default=2048)
    parser.add_argument("--association-buckets", type=int, default=64)
    parser.add_argument("--local-prior-mass", type=float, default=1.0)
    parser.add_argument("--association-prior-mass", type=float, default=4.0)
    parser.add_argument("--goal-gain", type=float, default=1.0)
    parser.add_argument("--max-goal-norm", type=float, default=8.0)
    parser.add_argument("--goal-logit-clip", type=float, default=4.0)
    parser.add_argument("--minimum-probability", type=float, default=1e-12)
    parser.add_argument(
        "--beta-values", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0)
    )
    parser.add_argument("--log-ratio-clip", type=float, default=2.0)
    parser.add_argument("--control-seed", type=int, default=3)
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument(
        "--max-dev-pairs",
        type=int,
        default=0,
        help="zero uses every dev pair",
    )
    parser.add_argument(
        "--max-test-pairs",
        type=int,
        default=0,
        help="zero uses every test pair",
    )
    parser.add_argument(
        "--require-exact-native",
        action="store_true",
        help="also require native base traces to forget pre-suffix context",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.feature_dimensions < 8:
        raise ValueError("feature-dimensions must be at least eight")
    if any(not 0.0 < value < 1.0 for value in args.state_decays):
        raise ValueError("state decays must lie in (0, 1)")
    if any(not 0.0 < value < 1.0 for value in args.successor_gammas):
        raise ValueError("successor gammas must lie in (0, 1)")
    if any(value < 0.0 for value in args.beta_values):
        raise ValueError("beta values must be non-negative")
    if args.resamples < 100:
        raise ValueError("resamples must be at least 100")
    if args.max_dev_pairs < 0 or args.max_test_pairs < 0:
        raise ValueError("maximum pair counts must be non-negative")
    report = run_ablation(args)
    payload = json.dumps(report, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
