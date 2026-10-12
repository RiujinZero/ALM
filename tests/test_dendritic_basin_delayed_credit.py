from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.delayed_credit import (
    DendriticDelayedCreditLearner,
    TeachingMode,
)
from manifoldflow.dendritic_basin.tasks.delayed_credit import DelayedCausalTask
from manifoldflow.experiments.dendritic_basin_long_range_credit_probe import (
    run_probe,
)


def test_delayed_causal_task_has_balanced_hidden_opposing_roles() -> None:
    task = DelayedCausalTask(16, delay_steps=7, seed=3)
    batch = task.sample(100)
    selected = batch.basal_states[
        np.arange(len(batch.outcomes)), batch.apical_receivers
    ]

    np.testing.assert_array_equal(
        batch.outcomes,
        selected ^ task.hidden_polarities[batch.apical_receivers],
    )
    assert np.sum(task.hidden_polarities == 0) == 8
    assert np.sum(task.hidden_polarities == 1) == 8
    assert batch.delay_steps == 7


def test_apical_instruction_changes_only_the_routed_local_table() -> None:
    learner = DendriticDelayedCreditLearner(
        8, teaching_mode=TeachingMode.APICAL, eligibility_decay=0.9
    )
    before = learner.outcome_counts.copy()
    step = learner.run_trial(
        np.asarray((0, 1, 1, 0, 1, 0, 0, 1), dtype=np.uint8),
        apical_receiver=3,
        delay_steps=5,
        outcome=1,
    )
    changed = np.argwhere(learner.outcome_counts != before)

    np.testing.assert_array_equal(changed, np.asarray(((3, 0, 1),)))
    assert step.update_count == 1
    assert step.off_route_updates == 0
    assert learner.outcome_counts[3, 0, 1] == 0.9**5


def test_broadcast_instruction_contaminates_every_eligible_receiver() -> None:
    learner = DendriticDelayedCreditLearner(
        8, teaching_mode=TeachingMode.BROADCAST
    )
    step = learner.run_trial(
        np.asarray((0, 1, 1, 0, 1, 0, 0, 1), dtype=np.uint8),
        apical_receiver=3,
        delay_steps=5,
        outcome=1,
    )

    assert step.update_count == 8
    assert step.off_route_updates == 7
    assert np.count_nonzero(learner.outcome_counts) == 8


def test_receiver_specific_credit_solves_hidden_roles_sample_efficiently() -> None:
    report = run_probe(
        receiver_count=16,
        train_trials=4_000,
        eval_trials=4_000,
        delay_steps=8,
        eligibility_decay=0.97,
        seed=5,
    )
    conditions = report["conditions"]
    absent = conditions["none"]
    broadcast = conditions["broadcast"]
    apical = conditions["apical"]

    assert apical["eval_bits_per_outcome"] < 0.02
    assert apical["eval_accuracy"] == 1.0
    assert apical["mapping_accuracy"] == 1.0
    assert apical["eval_bits_per_outcome"] < broadcast["eval_bits_per_outcome"]
    assert broadcast["eval_bits_per_outcome"] < absent["eval_bits_per_outcome"]
    assert apical["off_route_updates"] == 0
    assert broadcast["off_route_update_fraction"] == 15 / 16
