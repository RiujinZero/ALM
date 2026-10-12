"""Tests for the local next-token prediction readout (design doc section
27 of `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`)."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.prediction_table import (
    PredictionTable,
    bits_per_token,
    mix_distributions,
)


def test_distribution_sums_to_one() -> None:
    table = PredictionTable(vocab_size=5)
    table.update(1.0, 2)
    table.update(1.0, 2)
    dist = table.distribution()
    np.testing.assert_allclose(dist.sum(), 1.0)


def test_distribution_favors_repeatedly_observed_token() -> None:
    table = PredictionTable(vocab_size=4)
    for _ in range(20):
        table.update(1.0, 1)
    dist = table.distribution()
    assert dist[1] > dist[0] and dist[1] > dist[2] and dist[1] > dist[3]


def test_zero_activation_does_not_move_the_table() -> None:
    table = PredictionTable(vocab_size=3)
    before = table.counts.copy()
    table.update(0.0, 1)
    np.testing.assert_allclose(table.counts, before * table.decay)


def test_decay_shrinks_old_evidence_over_time() -> None:
    table = PredictionTable(vocab_size=3, decay=0.9)
    table.update(1.0, 0)
    first = table.counts[0]
    for _ in range(50):
        table.update(0.0, 0)
    assert table.counts[0] < first


def test_mix_distributions_with_no_predictors_returns_base() -> None:
    base = np.array([0.5, 0.5])
    mixed = mix_distributions(base, [])
    np.testing.assert_allclose(mixed, base)


def test_mix_distributions_weights_toward_a_confident_predictor() -> None:
    base = np.array([0.5, 0.5])
    confident = np.array([0.99, 0.01])
    mixed = mix_distributions(base, [(5.0, confident)])
    assert mixed[0] > base[0]
    np.testing.assert_allclose(mixed.sum(), 1.0)


def test_mix_distributions_ignores_non_positive_weights() -> None:
    base = np.array([0.5, 0.5])
    mixed = mix_distributions(base, [(0.0, np.array([1.0, 0.0]))])
    np.testing.assert_allclose(mixed, base)


def test_bits_per_token_is_zero_for_perfect_confidence() -> None:
    assert bits_per_token([1.0, 1.0, 1.0]) == 0.0


def test_bits_per_token_matches_log2_for_uniform_probability() -> None:
    import math

    result = bits_per_token([0.25, 0.25])
    assert math.isclose(result, 2.0, rel_tol=1e-9)
