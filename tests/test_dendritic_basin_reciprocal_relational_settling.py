from __future__ import annotations

import numpy as np

from manifoldflow.experiments.dendritic_basin_reciprocal_relational_settling_smoke import (  # noqa: E501
    build_world,
    run_seed,
)


def test_relational_world_balances_filler_frequency() -> None:
    world = build_world(3)

    counts = np.bincount(world.targets)

    np.testing.assert_array_equal(counts, np.full_like(counts, counts[0]))


def test_reciprocal_selective_settling_resolves_complementary_cues() -> None:
    row = run_seed(7)

    assert row["local_cortex_only_accuracy"] == 0.5
    assert row["one_way"]["accuracy"] == 0.25
    assert row["reciprocal_selective"]["accuracy"] == 1.0
    assert row["reciprocal_selective"]["correct_memory_fraction"] == 1.0


def test_competition_relationship_and_development_are_causal() -> None:
    row = run_seed(11)
    settled = row["reciprocal_selective"]["accuracy"]

    assert row["reciprocal_without_competition"]["accuracy"] < settled
    assert row["relation_scrambled"]["accuracy"] < settled
    assert row["staged_misaligned_access"]["accuracy"] < 0.4
    assert row["joint_aligned_access"]["accuracy"] > 0.9
