from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.contextual_competition import (
    competitive_readout,
    normalized_competition,
)


def test_kappa_zero_recovers_exact_softmax() -> None:
    logits = np.array([2.0, 0.5, -1.0, 3.0])
    drive = np.exp(logits)
    gain = np.ones_like(drive)

    influence = normalized_competition(drive, gain, kappa=0.0)

    expected = np.exp(logits) / np.sum(np.exp(logits))
    np.testing.assert_allclose(influence, expected, rtol=1e-12)
    assert np.isclose(np.sum(influence), 1.0)


def test_weak_total_drive_yields_weak_total_influence() -> None:
    strong = normalized_competition(
        np.array([10.0, 10.0]), np.array([1.0, 1.0]), kappa=1.0
    )
    weak = normalized_competition(
        np.array([0.01, 0.01]), np.array([1.0, 1.0]), kappa=1.0
    )

    assert np.sum(weak) < np.sum(strong)
    assert np.sum(weak) < 0.05


def test_context_gain_can_favor_a_weakly_driven_competitor() -> None:
    drive = np.array([1.0, 1.0])
    gain = np.array([0.1, 5.0])

    influence = normalized_competition(drive, gain, kappa=0.0)

    assert influence[1] > influence[0]


def test_exponent_sharpens_the_winner() -> None:
    drive = np.array([2.0, 1.0])
    gain = np.array([1.0, 1.0])

    soft = normalized_competition(drive, gain, exponent=1.0, kappa=0.0)
    sharp = normalized_competition(drive, gain, exponent=4.0, kappa=0.0)

    assert sharp[0] > soft[0]


def test_competitive_readout_matches_manual_weighted_sum() -> None:
    drive = np.array([3.0, 1.0])
    gain = np.array([1.0, 1.0])
    values = np.array([[1.0, 0.0], [0.0, 1.0]])

    influence, y = competitive_readout(drive, gain, values, kappa=0.0)

    np.testing.assert_allclose(y, influence[0] * values[0] + influence[1] * values[1])


def test_rejects_negative_drive_or_mismatched_shapes() -> None:
    try:
        normalized_competition(np.array([-1.0]), np.array([1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative drive")

    try:
        normalized_competition(np.array([1.0, 2.0]), np.array([1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for shape mismatch")
