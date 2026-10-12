"""Oracle tests for optional compiled dendritic arithmetic."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.compiled_kernels import (
    access_branch_survival_compiled,
    access_branch_survival_python,
    interaction_objective_compiled,
    interaction_objective_python,
    select_column_winners_compiled,
    update_interaction_stats_compiled,
)
from manifoldflow.dendritic_basin.interaction_dendrite import RunningJointStats


def test_compiled_access_survival_matches_ordered_python_oracle() -> None:
    rng = np.random.default_rng(113)
    traces = rng.normal(0.2, 0.5, size=(7, 24)).astype(np.float32)
    saliences = np.asarray([0.1, 0.8, 0.2, 1.1, 0.4, 0.9, 0.3])
    indices = np.asarray(
        [rng.choice(24, size=9, replace=False) for _ in range(11)],
        dtype=np.int32,
    )
    weights = rng.normal(0.15, 0.08, size=indices.shape)
    synapse_counts = np.full(11, 9, dtype=np.int32)
    group_targets = rng.uniform(0.15, 0.8, size=(11, 4))
    group_counts = np.full(11, 4, dtype=np.int32)
    group_counts[0] = 0
    strengths = rng.uniform(0.2, 1.0, size=11)
    last_used = np.arange(11, dtype=np.int64)
    capture_saliences = rng.uniform(0.0, 1.0, size=11)
    arguments = (
        traces,
        saliences,
        indices,
        weights,
        synapse_counts,
        group_targets,
        group_counts,
        strengths,
        last_used,
        capture_saliences,
        12,
        2,
        True,
        4,
        0.35,
        0.25,
        0.15,
    )

    expected = access_branch_survival_python(*arguments)
    actual = access_branch_survival_compiled(*arguments)

    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])


def test_compiled_interaction_objective_matches_python_oracle() -> None:
    rng = np.random.default_rng(17)
    samples = rng.random((64, 2))
    parameters = (0.72, -0.31, 0.44, 0.19)

    expected = interaction_objective_python(samples, *parameters)
    actual = interaction_objective_compiled(samples, *parameters)

    assert actual == expected


def test_compiled_column_tournament_is_local_stable_and_sparse() -> None:
    weights = np.asarray([0.7, 0.7, 0.2, 0.9, 0.8, 0.4])
    slot_ids = np.asarray([0, 1, 2, 4, 5, 8], dtype=np.int64)
    candidate_positions = np.arange(6, dtype=np.int64)

    winners, count = select_column_winners_compiled(
        weights,
        slot_ids,
        candidate_positions,
        column_size=4,
        column_count=3,
        max_per_column=2,
        max_columns=2,
        max_active=4,
    )

    # Columns 1 and 0 win. Tied branches 0/1 retain anatomical order.
    assert winners[:count].tolist() == [3, 4, 0, 1]


def test_compiled_interaction_stats_match_branch_local_welford_oracle() -> None:
    rng = np.random.default_rng(31)
    branch_count = 7
    arrays = [np.zeros(branch_count) for _ in range(8)]
    stats_n = np.zeros(branch_count, dtype=np.int64)
    oracles = [RunningJointStats() for _ in range(branch_count)]

    for _ in range(40):
        slot_ids = np.asarray([0, 2, 3, 6], dtype=np.int64)
        a_values = rng.random(branch_count)
        b_values = rng.random(branch_count)
        responses = rng.random(branch_count)
        update_interaction_stats_compiled(
            slot_ids,
            a_values,
            b_values,
            responses,
            stats_n,
            *arrays,
        )
        for slot_id in slot_ids:
            oracles[int(slot_id)].update(
                a_values[slot_id], b_values[slot_id], responses[slot_id]
            )

    fields = (
        "mean_a",
        "mean_b",
        "mean_d",
        "m2_a",
        "m2_b",
        "m2_d",
        "cov_da_acc",
        "cov_db_acc",
    )
    for slot_id, oracle in enumerate(oracles):
        assert stats_n[slot_id] == oracle.n
        for field, values in zip(fields, arrays, strict=True):
            assert values[slot_id] == getattr(oracle, field)
