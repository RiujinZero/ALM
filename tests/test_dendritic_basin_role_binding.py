from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.role_binding import BindingMode, RoleBindingLearner
from manifoldflow.dendritic_basin.tasks.role_binding import CompositionalRoleTask
from manifoldflow.experiments.dendritic_basin_binding_separation_probe import (
    run_probe,
)


def test_unseen_pair_split_holds_out_both_role_directions() -> None:
    task = CompositionalRoleTask(seed=4)
    train = {tuple(pair) for pair in task.train_pairs}
    for actor, patient in task.eval_pairs:
        assert (int(actor), int(patient)) not in train
        assert (int(patient), int(actor)) not in train


def test_role_gate_updates_actor_assembly_only() -> None:
    task = CompositionalRoleTask(seed=2)
    learner = RoleBindingLearner(binding_mode=BindingMode.ROLE_GATED)
    actor = task.codebook[1]
    patient = task.codebook[3]
    step = learner.observe(actor, patient, outcome=1)

    assert step.update_count == 1
    assert step.off_role_updates == 0
    assert len(learner.outcome_counts) == 1
    assert learner.probability(actor, patient) > 0.5
    assert learner.probability(patient, actor) == 0.5


def test_bag_coactivity_creates_an_off_role_association() -> None:
    task = CompositionalRoleTask(seed=2)
    learner = RoleBindingLearner(binding_mode=BindingMode.BAG)
    actor = task.codebook[1]
    patient = task.codebook[3]
    step = learner.observe(actor, patient, outcome=1)

    assert step.update_count == 2
    assert step.off_role_updates == 1
    assert learner.probability(patient, actor) > 0.5


def test_role_binding_recombines_unseen_pairs_without_false_binding() -> None:
    report = run_probe(
        entity_count=16,
        train_trials=4_000,
        eval_trials=4_000,
        seed=6,
    )
    conditions = report["conditions"]
    bag = conditions["bag"]
    conjunctive = conditions["conjunctive"]
    role = conditions["role_gated"]

    assert role["unseen_pair_bits_per_outcome"] < 0.01
    assert role["unseen_pair_accuracy"] == 1.0
    assert role["opposed_role_accuracy"] == 1.0
    assert role["unseen_pair_bits_per_outcome"] < bag["unseen_pair_bits_per_outcome"]
    assert role["unseen_pair_bits_per_outcome"] < conjunctive[
        "unseen_pair_bits_per_outcome"
    ]
    assert role["off_role_updates"] == 0
    assert bag["off_role_update_fraction"] == 0.5


def test_entity_codebook_is_distributed_and_unique() -> None:
    task = CompositionalRoleTask(seed=7)
    np.testing.assert_array_equal(task.codebook.sum(axis=1), np.full(16, 8))
    assert len({row.tobytes() for row in task.codebook}) == 16
