from __future__ import annotations

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_shared_goal_continuous_crossover_ablation as continuous,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    CrossoverPair,
    TrainingExample,
)


def _training_examples() -> list[TrainingExample]:
    examples = []
    for index in range(16):
        sign = 1.0 if index % 2 == 0 else -1.0
        target = np.asarray([2.5 * sign, 0.25 * index])
        evidence = np.asarray([sign, index / 16, sign, 0.5])
        examples.append(
            TrainingExample(
                episode_id=str(index),
                response=b"A" if sign > 0 else b"B",
                prefix=b"tail",
                evidence=evidence,
                response_vector=np.asarray([sign, 0.0]),
                goal_target_vector=target,
            )
        )
    return examples


def _mechanism():
    return continuous.fit_continuous_mechanism(
        _training_examples(),
        feature_dimensions=2,
        ridge_alpha=1e-6,
        context_order=4,
        context_buckets=127,
        association_buckets=17,
        local_prior_mass=1.0,
        association_prior_mass=1.0,
        goal_gain=2.0,
        max_goal_norm=8.0,
        goal_logit_clip=4.0,
        minimum_probability=1e-12,
        control_seed=7,
    )


def test_fit_keeps_raw_continuous_magnitude_and_strict_training_shuffles() -> None:
    mechanism = _mechanism()
    evidence = np.asarray([row.evidence for row in _training_examples()])

    prediction = mechanism.full_predictor.predict(evidence)

    assert np.max(np.linalg.norm(prediction, axis=1)) > 2.0
    assert np.all(
        mechanism.predictive_target_permutation
        != np.arange(len(mechanism.predictive_target_permutation))
    )
    assert np.all(
        mechanism.action_target_permutation
        != np.arange(len(mechanism.action_target_permutation))
    )


def test_representation_report_uses_pairing_cosine_but_preserves_raw_norms() -> None:
    target = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    prediction = 3.0 * target

    report = continuous._representation_summary(
        prediction, target, max_goal_norm=2.0, seed=3, resamples=200
    )

    assert report["mean_paired_cosine_margin"] == 2.0
    assert report["paired_cosine_win_fraction"] == 1.0
    assert report["prediction_norm"]["mean"] == 3.0
    assert report["prediction_norm"]["fraction_above_head_clamp"] == 1.0


def _prepared_pair(mechanism) -> continuous.PreparedPair:
    pair = CrossoverPair(
        pair_id="p",
        context_a=b"first context-tail",
        context_b=b"other context-tail",
        response_a=b"A",
        response_b=b"B",
        evidence_a=np.asarray([1.0, 0.0, 1.0, 0.5]),
        evidence_b=np.asarray([-1.0, 0.0, -1.0, 0.5]),
        source_label="toy",
        goal_target_vector_a=np.asarray([2.5, 0.0]),
        goal_target_vector_b=np.asarray([-2.5, 0.0]),
    )
    base = {}
    zero = np.zeros(2)
    for context_name, context in (("a", pair.context_a), ("b", pair.context_b)):
        for response_name, response in (("a", b"A"), ("b", b"B")):
            local, _ = mechanism.action_head.trace_probabilities(
                context, response, zero
            )
            base[context_name + response_name] = local
    return continuous.PreparedPair(pair, base)


def test_zero_goal_at_selected_beta_is_exact_and_signed_oracle_reverses() -> None:
    mechanism = _mechanism()
    prepared = _prepared_pair(mechanism)
    base = prepared.base_traces["aa"]
    local, goal_off = mechanism.action_head.trace_probabilities(
        prepared.pair.context_a, b"A", np.zeros(2)
    )

    corrected = continuous._corrected_probability_trace(
        base, local, goal_off, beta=2.0, log_ratio_clip=2.0
    )
    true_margin = continuous._condition_margin(
        prepared,
        mechanism.action_head,
        np.asarray([2.5, 0.0]),
        np.asarray([-2.5, 0.0]),
        beta=2.0,
        log_ratio_clip=2.0,
    )
    swapped_margin = continuous._condition_margin(
        prepared,
        mechanism.action_head,
        np.asarray([-2.5, 0.0]),
        np.asarray([2.5, 0.0]),
        beta=2.0,
        log_ratio_clip=2.0,
    )

    assert np.array_equal(corrected, base)
    assert true_margin > 0.0
    assert swapped_margin < 0.0


def test_condition_matrix_uses_exact_goal_off_and_includes_all_controls() -> None:
    mechanism = _mechanism()
    prepared = _prepared_pair(mechanism)
    goals = continuous.PairGoals(
        oracle_a=np.asarray([2.5, 0.0]),
        oracle_b=np.asarray([-2.5, 0.0]),
        causal_full_a=np.asarray([2.0, 0.1]),
        causal_full_b=np.asarray([-2.0, 0.1]),
        causal_pending_a=np.asarray([1.5, 0.2]),
        causal_pending_b=np.asarray([-1.5, 0.2]),
        shuffled_predictor_a=np.asarray([-0.1, 0.1]),
        shuffled_predictor_b=np.asarray([0.1, 0.1]),
        shuffled_episode_a=np.asarray([-2.0, 0.1]),
        shuffled_episode_b=np.asarray([2.0, 0.1]),
    )

    raw, adjusted, invariants = continuous._evaluate_conditions(
        [prepared],
        [goals],
        mechanism,
        beta=2.0,
        log_ratio_clip=2.0,
        require_exact_native=False,
    )

    assert set(raw) == {
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
    }
    assert np.array_equal(adjusted["goal_off"], np.zeros(1))
    assert invariants["exact_local_q0_cross_context_trace_identity"] is True
    assert invariants["exact_goal_off_base_trace_identity"] is True


def test_sattolo_control_is_deterministic_and_has_no_fixed_points() -> None:
    first = continuous._sattolo_indices(32, seed=19)
    second = continuous._sattolo_indices(32, seed=19)

    assert np.array_equal(first, second)
    assert np.all(first != np.arange(32))
