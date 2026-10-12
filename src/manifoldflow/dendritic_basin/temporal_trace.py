"""Multi-timescale temporal traces (design doc section 14 of
`Manifold_Dendritic_Stream_Learner_Design_v0.2.md`): cheap temporal
context without a fixed context window.

  q_i^(tau)(t+1) = lambda_tau * q_i^(tau)(t) + x_i(t)

Each channel keeps several traces at different decay rates, giving a
representation along the lines of "active now" / "active very recently"
/ "active recently" / "active moderately long ago" / "active much longer
ago" -- all from one scalar update rule per timescale, no stored window
of past inputs.
"""

from __future__ import annotations

import numpy as np


def lambda_from_halflife(halflife_steps: float) -> float:
    """decay rate lambda such that a unit impulse decays to half its
    value after `halflife_steps` steps: lambda = 2^(-1/halflife_steps).
    """
    return float(2.0 ** (-1.0 / halflife_steps))


DEFAULT_TIMESCALES = (1, 2, 4, 8, 16, 32, 64, 128)


class MultiTimescaleTraces:
    """Decaying traces for `n_channels` scalar event channels, at each of
    `timescales` (halflife in steps). State shape: (len(timescales),
    n_channels).
    """

    def __init__(
        self, n_channels: int, timescales: tuple[int, ...] = DEFAULT_TIMESCALES
    ):
        self.n_channels = n_channels
        self.timescales = timescales
        self.lambdas = np.array([lambda_from_halflife(tau) for tau in timescales])
        self.q = np.zeros((len(timescales), n_channels))

    def update(self, x: np.ndarray) -> None:
        """q(t+1) = lambda*q(t) + x(t), applied per timescale row."""
        self.q = self.lambdas[:, None] * self.q + x[None, :]

    def reset(self) -> None:
        self.q[:] = 0.0

    def value(self, timescale_index: int, channel: int) -> float:
        return float(self.q[timescale_index, channel])

    def channel_traces(self, channel: int) -> np.ndarray:
        """All timescales' current trace value for one channel."""
        return self.q[:, channel]
