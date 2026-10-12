"""Design doc sections 10-12: competitive recurrent basin settling and its
local Hebbian plasticity."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.basin import BasinConfig, RecurrentBasin


def test_settle_with_zero_gains_returns_clipped_drive() -> None:
    """With recurrent_gain=inhibition_gain=0, h should just track the
    (non-negative-clipped) drive at every step -- the degenerate case
    that isolates settling's iteration mechanics from its dynamics.
    """
    config = BasinConfig(
        n_units=4, settle_steps=3, recurrent_gain=0.0, inhibition_gain=0.0
    )
    basin = RecurrentBasin(config)
    drive = np.array([0.5, -0.2, 0.9, 0.0])

    h = basin.settle(drive)

    np.testing.assert_allclose(h, np.clip(drive, 0.0, None))


def test_settle_output_is_non_negative() -> None:
    config = BasinConfig(n_units=5, settle_steps=4, seed=1)
    basin = RecurrentBasin(config)
    rng = np.random.default_rng(0)
    drive = rng.uniform(0, 1, size=5)

    h = basin.settle(drive)

    assert np.all(h >= 0)


def test_settle_is_deterministic_given_the_same_state() -> None:
    config = BasinConfig(n_units=4, seed=2)
    basin = RecurrentBasin(config)
    drive = np.array([0.3, 0.7, 0.1, 0.4])

    h_a = basin.settle(drive)
    h_b = basin.settle(drive)

    np.testing.assert_allclose(h_a, h_b)


def test_learn_updates_w_rec_toward_the_settled_state_correlation() -> None:
    config = BasinConfig(
        n_units=3, hebbian_lr=0.1, hebbian_decay=0.0, normalize_rows=False, seed=3
    )
    basin = RecurrentBasin(config)
    w_before = basin.w_rec.copy()
    h = np.array([1.0, 1.0, 0.0])

    basin.learn(h)

    expected = w_before + 0.1 * np.outer(h, h)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(basin.w_rec, expected)


def test_learn_decays_existing_weights() -> None:
    config = BasinConfig(
        n_units=3, hebbian_lr=0.0, hebbian_decay=0.5, normalize_rows=False, seed=4
    )
    basin = RecurrentBasin(config)
    basin.w_rec = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

    basin.learn(np.zeros(3))

    expected = 0.5 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    np.testing.assert_allclose(basin.w_rec, expected)


def test_learn_row_normalizes_by_default_and_stays_bounded() -> None:
    """Regression check for the numerical blow-up this default fixes:
    many repeated Hebbian updates on a strongly-correlated state must
    stay finite and bounded, not explode (see the implementation journal
    -- this is exactly what happened before `normalize_rows` existed).
    """
    config = BasinConfig(n_units=4, hebbian_lr=0.2, hebbian_decay=0.01, seed=5)
    basin = RecurrentBasin(config)
    h = np.array([1.0, 1.0, 0.5, 0.0])

    for _ in range(2000):
        basin.learn(h)

    assert np.all(np.isfinite(basin.w_rec))
    assert np.abs(basin.w_rec).max() <= 1.0 + 1e-6
