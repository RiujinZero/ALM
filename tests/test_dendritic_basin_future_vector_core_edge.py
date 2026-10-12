"""Tests for the mass-weighted shared-core/edge split."""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.future_vector_core_edge import (
    CoreEdgeSplit,
    mass_weighted_core_edge_split,
    reconstruction_error,
)


def test_own_core_and_edge_reconstruct_the_candidate_exactly() -> None:
    means = tuple(
        np.asarray(item, dtype=np.float64)
        for item in ((0.0, 0.0), (2.0, 0.0), (0.0, 4.0))
    )
    probabilities = np.asarray([0.5, 0.3, 0.2])
    split = mass_weighted_core_edge_split(means, probabilities, chosen=(0, 1, 2))
    for position, index in enumerate(split.chosen):
        error = reconstruction_error(means[index], split.core, split.edges[position])
        assert error == pytest.approx(0.0, abs=1e-12)


def test_core_is_the_mass_weighted_mean_of_chosen_candidates_only() -> None:
    means = tuple(
        np.asarray(item, dtype=np.float64) for item in ((0.0,), (10.0,), (1_000.0,))
    )
    probabilities = np.asarray([0.6, 0.4, 100.0])  # index 2 excluded from `chosen`
    split = mass_weighted_core_edge_split(means, probabilities, chosen=(0, 1))
    expected_core = (0.6 * 0.0 + 0.4 * 10.0) / (0.6 + 0.4)
    assert split.core == pytest.approx(np.asarray([expected_core]))
    assert split.core_probability_mass == pytest.approx(1.0)


def test_edges_are_immutable_and_shape_matched_to_core() -> None:
    means = tuple(np.asarray(item, dtype=np.float64) for item in ((1.0, 2.0), (3.0, 4.0)))
    split = mass_weighted_core_edge_split(means, np.asarray([0.5, 0.5]), chosen=(0, 1))
    with pytest.raises(ValueError):
        split.core[0] = 99.0
    with pytest.raises(ValueError):
        split.edges[0][0] = 99.0


def test_mismatched_edge_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        CoreEdgeSplit(
            core=np.zeros(2),
            edges=(np.zeros(3),),
            chosen=(0,),
            core_probability_mass=1.0,
        )


def test_foreign_core_can_be_scored_against_a_different_episodes_edge() -> None:
    # reconstruction_error takes core/edge/target independently, so a
    # "foreign" core from a different split can be substituted directly --
    # this is the causal test the ablation script performs across pairs.
    means_a = tuple(np.asarray(item, dtype=np.float64) for item in ((0.0,), (2.0,)))
    means_b = tuple(np.asarray(item, dtype=np.float64) for item in ((50.0,), (52.0,)))
    split_a = mass_weighted_core_edge_split(means_a, np.asarray([0.5, 0.5]), chosen=(0, 1))
    split_b = mass_weighted_core_edge_split(means_b, np.asarray([0.5, 0.5]), chosen=(0, 1))
    own = reconstruction_error(means_a[0], split_a.core, split_a.edges[0])
    foreign = reconstruction_error(means_a[0], split_b.core, split_a.edges[0])
    assert own == pytest.approx(0.0, abs=1e-12)
    assert foreign > own


def test_empty_chosen_is_rejected() -> None:
    means = (np.zeros(2),)
    with pytest.raises(ValueError):
        mass_weighted_core_edge_split(means, np.asarray([1.0]), chosen=())


def test_zero_mass_chosen_candidates_are_rejected() -> None:
    means = tuple(np.zeros(2) for _ in range(2))
    with pytest.raises(ValueError):
        mass_weighted_core_edge_split(means, np.asarray([0.0, 0.0]), chosen=(0, 1))
