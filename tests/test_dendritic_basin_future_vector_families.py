"""Tests for shared-core future-vector families."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.future_vector_families import (
    agglomerative_families,
    random_size_matched_families,
    shortlist_families,
)


def test_agglomeration_recovers_two_shared_cores() -> None:
    means = tuple(
        np.asarray(item, dtype=np.float64)
        for item in ((0.0, 0.0), (0.1, 0.0), (4.0, 4.0), (4.1, 4.0))
    )
    families = agglomerative_families(means, np.ones(2), maximum_families=2)
    assert families == ((0, 1), (2, 3))


def test_family_first_selection_keeps_edges_inside_selected_core() -> None:
    means = tuple(
        np.asarray(item, dtype=np.float64)
        for item in ((0.0, 0.0), (0.1, 0.0), (4.0, 4.0), (4.1, 4.0))
    )
    result = shortlist_families(
        means,
        np.asarray([0.35, 0.30, 0.20, 0.15]),
        np.ones(2),
        ((0, 1), (2, 3)),
        family_mass=0.6,
        maximum_selected_families=1,
        candidate_mass=0.8,
        maximum_candidates=3,
    )
    assert result.selected_families == (0,)
    assert result.eligible_candidates == (0, 1)
    assert result.selected_candidates == (0, 1)
    assert result.shared_core_fraction > 0.9


def test_random_control_preserves_sizes_and_partition() -> None:
    families = ((0, 1, 2), (3,), (4, 5))
    shuffled = random_size_matched_families(families, seed=7)
    assert tuple(len(item) for item in shuffled) == (3, 1, 2)
    assert sorted(item for family in shuffled for item in family) == list(range(6))
