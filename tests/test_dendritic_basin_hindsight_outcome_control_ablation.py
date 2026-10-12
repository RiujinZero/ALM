from __future__ import annotations

import json

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_hindsight_outcome_control_ablation as assay,
)


def _training(
    *, seed: int = 7, count: int = 512, horizon: int = 24
) -> tuple[
    assay.VectorCode,
    list[assay.ObservedTrajectory],
    list[assay.TrainingAudit],
]:
    code = assay.VectorCode()
    trajectories, audit = assay.generate_observed_trajectories(
        count,
        horizon=horizon,
        code=code,
        rng=np.random.default_rng(seed),
        key_prefix=f"train-{seed}",
    )
    return code, trajectories, audit


def test_logs_withhold_scalar_labels_but_cue_exposes_intention_identity() -> None:
    code, trajectories, audit = _training(seed=8)
    probes = assay.generate_crossover_probes(
        8,
        horizon=24,
        code=code,
        rng=np.random.default_rng(9),
        key_prefix="eval-9",
    )

    assert not hasattr(trajectories[0], "goal")
    assert not hasattr(trajectories[0], "reward")
    assert not hasattr(trajectories[0], "success")
    assert all(
        code.decode_cue(trajectory.observations[0]) == row.intended_outcome
        for trajectory, row in zip(trajectories, audit, strict=True)
    )
    assert {row.key for row in trajectories}.isdisjoint(probe.key for probe in probes)
    for group in range(8):
        rows = [probe for probe in probes if probe.group == group]
        assert len(rows) == code.goals
        assert {row.desired_outcome for row in rows} == set(range(code.goals))
        assert all(
            np.array_equal(rows[0].prefix[1:], row.prefix[1:]) for row in rows[1:]
        )
        assert len({row.prefix[0].tobytes() for row in rows}) == code.goals

    coverage = assay._training_coverage(audit, goals=code.goals)
    assert coverage["action_zero_is_unique_marginal_majority"]
    assert coverage["action_zero_is_unique_majority_in_every_context"]
    assert coverage["action_fractions"][0] > 0.50
    assert coverage["minimum_context_action_coverage"] >= 8
    assert 0.74 < coverage["cue_follow_fraction"] < 0.86


def test_far_lag_goal_is_self_supervised_and_requires_register_state() -> None:
    code, trajectories, _ = _training(seed=10)
    probes = assay.generate_crossover_probes(
        8,
        horizon=24,
        code=code,
        rng=np.random.default_rng(11),
    )
    causal = assay.LaggedHindsightGoal(code.dimension, lag=24)
    causal.fit(trajectories)
    shuffled = assay.LaggedHindsightGoal(code.dimension, lag=24)
    shuffled.fit(
        assay.shuffled_terminal_control(
            trajectories,
            code=code,
            horizon=24,
            rng=np.random.default_rng(12),
        )
    )

    causal_predictions = [causal.forecast(probe.prefix) for probe in probes]
    reset_predictions = [
        causal.forecast(probe.prefix, reset_index=12) for probe in probes
    ]
    shuffled_predictions = [shuffled.forecast(probe.prefix) for probe in probes]

    assert all(
        code.decode_outcome(prediction) == probe.desired_outcome
        for prediction, probe in zip(causal_predictions, probes, strict=True)
    )
    assert all(
        np.array_equal(prediction, np.zeros(code.dimension))
        for prediction in reset_predictions
    )
    shuffled_accuracy = np.mean(
        [
            code.decode_outcome(prediction) == probe.desired_outcome
            for prediction, probe in zip(shuffled_predictions, probes, strict=True)
        ]
    )
    assert shuffled_accuracy == 0.25

    jittered = {
        offset: [
            causal.forecast(assay.jittered_decision_prefix(probe.prefix, offset))
            for probe in probes
        ]
        for offset in (-1, 0, 1)
    }
    jitter_accuracy = {
        offset: np.mean(
            [
                code.decode_outcome(prediction) == probe.desired_outcome
                for prediction, probe in zip(values, probes, strict=True)
            ]
        )
        for offset, values in jittered.items()
    }
    assert jitter_accuracy == {-1: 0.25, 0: 1.0, 1: 0.25}
    for probe in probes:
        for offset in (-1, 0, 1):
            changed = assay.jittered_decision_prefix(probe.prefix, offset)
            assert np.array_equal(changed[0], probe.prefix[0])
            assert np.array_equal(changed[-4:], probe.prefix[-4:])


def test_allowed_input_cue_context_behavior_clone_is_a_strong_control() -> None:
    code, trajectories, _ = _training(seed=16, count=2048)
    probes = assay.generate_crossover_probes(
        16,
        horizon=24,
        code=code,
        rng=np.random.default_rng(17),
    )
    clone = assay.ObservedCueContextBehaviorClone(
        observation_dimension=code.dimension,
        actions=code.goals,
    )
    clone.fit(trajectories, horizon=24)
    before = clone.fingerprint()
    metrics, selected = assay._score_behavior_clone(probes, clone)

    assert metrics["action_accuracy"] == 1.0
    assert metrics["less_locally_likely_action_accuracy"] == 1.0
    assert metrics["complete_crossover_group_rate"] == 1.0
    assert len(selected) == len(probes)
    assert clone.fingerprint() == before


def test_consequence_interaction_beats_allocation_matched_additive_control() -> None:
    code, trajectories, _ = _training(seed=13)
    probes = assay.generate_crossover_probes(
        12,
        horizon=24,
        code=code,
        rng=np.random.default_rng(14),
    )
    additive_wide = assay.ActionConditionedOutcomeModel(
        "additive_wide", observation_dimension=code.dimension, actions=code.goals
    )
    coupled = assay.ActionConditionedOutcomeModel(
        "coupled", observation_dimension=code.dimension, actions=code.goals
    )
    additive_wide.fit(trajectories, horizon=24)
    coupled.fit(trajectories, horizon=24)
    exemplar = probes[0].prefix[-1]

    assert additive_wide.parameter_count == coupled.parameter_count
    assert additive_wide.active_feature_count(exemplar, 0) == (
        coupled.active_feature_count(exemplar, 0)
    )
    additive_correct = 0
    coupled_correct = 0
    for probe in probes:
        goal = code.terminal(probe.desired_outcome)
        additive_correct += int(
            additive_wide.choose(probe.prefix[-1], goal) == probe.required_action
        )
        coupled_correct += int(
            coupled.choose(probe.prefix[-1], goal) == probe.required_action
        )
    assert coupled_correct == len(probes)
    assert additive_correct / len(probes) < 0.50


def test_context_coupled_writer_persists_selected_action_through_generation() -> None:
    _, trajectories, _ = _training(seed=15)
    writers = assay.fit_writers(trajectories, table_bits=10)
    histograms = {
        tuple(np.bincount(assay.action_command(action), minlength=9))
        for action in range(4)
    }

    assert len(histograms) == 1
    assert writers["additive_wide"].parameter_count == (
        writers["coupled"].parameter_count
    )
    assert writers["additive_wide"].raw_active_feature_count(1) == (
        writers["coupled"].raw_active_feature_count(1)
    )
    for action in range(4):
        decoded, generated = assay.generate_action(writers["coupled"], action)
        assert decoded == action
        assert generated == assay.action_command(action)
    assert {
        assay.generate_action(writers["local"], action)[0] for action in range(4)
    } == {0}
    for kind in ("local", "additive", "additive_wide"):
        assert (
            sum(
                assay.generate_action(writers[kind], action)[0] == action
                for action in range(4)
            )
            < 4
        )
    assert {
        assay.generate_action(writers["coupled"], action, disconnected=True)[0]
        for action in range(4)
    } == {0}


@pytest.fixture(scope="module")
def small_report() -> dict[str, object]:
    return assay.run_ablation(
        train_trajectories=512,
        crossover_groups=8,
        horizon=24,
        table_bits=10,
        seed=19,
    )


def test_full_report_separates_inference_selection_and_realization(
    small_report: dict[str, object],
) -> None:
    report = small_report
    assert report["schema_version"] == 2
    assert report["dataset_audit"]["train_evaluation_key_overlap"] == 0
    assert report["dataset_audit"]["matched_post_cue_prefixes_identical"]
    assert (
        report["dataset_audit"]["balanced_evaluation_action_zero_wrong_fraction"]
        == 0.75
    )
    assert report["goal_forecast"]["causal"]["accuracy"] == 1.0
    assert report["goal_forecast"]["swapped"]["accuracy"] == 0.0
    assert report["selector"]["observable_action_prior"]["action_accuracy"] == 0.25
    assert (
        report["selector"]["cue_context_behavior_clone_oracle_retained_cue"][
            "action_accuracy"
        ]
        >= 0.90
    )
    assert report["selector"]["coupled__causal"]["action_accuracy"] == 1.0
    assert (
        report["selector"]["coupled__causal"]["less_locally_likely_action_accuracy"]
        == 1.0
    )
    assert (
        report["writer_interface_oracle_clamped_action"]["coupled"]["exact_action_rate"]
        == 1.0
    )
    assert (
        report["end_to_end"]["coupled__coupled_selector__causal"][
            "desired_outcome_proxy_consistency"
        ]
        == 1.0
    )
    assert (
        report["end_to_end"]["coupled_disconnected__coupled_selector__causal"][
            "desired_outcome_proxy_consistency"
        ]
        == 0.25
    )
    assert (
        report["end_to_end"][
            "coupled_writer__cue_context_behavior_clone_oracle_retained_cue"
        ]["desired_outcome_proxy_consistency"]
        >= 0.90
    )
    assert report["headline_findings"]["decision_time_jitter_goal_accuracy"] == {
        "one_step_early": 0.25,
        "exact_lag": 1.0,
        "one_step_late": 0.25,
    }
    assert any(
        "injective observed one-hot cue" in value
        for value in report["scientific_contract"]["learner_inputs"]
    )
    assert report["provenance"]["bit_generator"] == "PCG64"
    assert report["provenance"]["experiment_source_sha256"]
    assert report["provenance"]["focused_test_sha256"]
    assert all(report["invariants"].values())
    json.loads(json.dumps(report))
