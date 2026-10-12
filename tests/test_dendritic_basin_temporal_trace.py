"""Tests for multi-timescale temporal traces (design doc section 14 of
`Manifold_Dendritic_Stream_Learner_Design_v0.2.md`)."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.temporal_trace import (
    MultiTimescaleTraces,
    lambda_from_halflife,
)


def test_lambda_from_halflife_decays_to_half_after_halflife_steps() -> None:
    lam = lambda_from_halflife(4.0)
    assert 0.0 < lam < 1.0
    assert abs(lam**4 - 0.5) < 1e-9


def test_update_matches_the_decay_recurrence_directly() -> None:
    traces = MultiTimescaleTraces(n_channels=2, timescales=(1, 4))
    x1 = np.array([1.0, 0.0])
    x2 = np.array([0.0, 1.0])

    traces.update(x1)
    traces.update(x2)

    expected = np.zeros((2, 2))
    for ch in range(2):
        for ti, lam in enumerate(traces.lambdas):
            q = 0.0
            q = lam * q + x1[ch]
            q = lam * q + x2[ch]
            expected[ti, ch] = q
    np.testing.assert_allclose(traces.q, expected)


def test_longer_timescale_retains_more_of_an_old_impulse() -> None:
    traces = MultiTimescaleTraces(n_channels=1, timescales=(1, 32))
    traces.update(np.array([1.0]))
    for _ in range(5):
        traces.update(np.array([0.0]))

    short_ts_value = traces.value(0, 0)
    long_ts_value = traces.value(1, 0)
    assert long_ts_value > short_ts_value


def test_reset_zeroes_all_traces() -> None:
    traces = MultiTimescaleTraces(n_channels=3)
    traces.update(np.array([1.0, 1.0, 1.0]))
    traces.reset()
    assert np.all(traces.q == 0.0)


def test_channel_traces_returns_one_column() -> None:
    traces = MultiTimescaleTraces(n_channels=3, timescales=(1, 2, 4))
    traces.update(np.array([1.0, 0.0, 0.0]))
    column = traces.channel_traces(0)
    assert column.shape == (3,)
    np.testing.assert_allclose(column, traces.q[:, 0])
