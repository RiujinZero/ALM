"""Phase 9 (design doc section 54: "Visual world -- ... persistence,
occlusion, identity."): the continuous-vector analogue of
`persistent_assembly.py`'s sticky, non-decaying state. Where
`MultiTimescaleTraces` always exponentially decays toward zero
(`q(t+1) = lambda*q(t) + x(t)`), this register instead *holds* the last
observed value of a channel exactly, for as long as observation says
"nothing new to see" -- the continuous counterpart of "did token j
immediately follow token i, since last reset" from Phase 3, generalized
to a value rather than a binary flag.
"""

from __future__ import annotations

import numpy as np


class PersistentContinuousRegister:
    """Holds the last-observed value of each of `n_channels` real-valued
    channels. `update` only overwrites a channel's stored value while
    `visible` is true for that call; while `visible` is false, the
    register keeps whatever it last held -- unlike a decaying trace, it
    never drifts toward zero on its own.
    """

    def __init__(self, n_channels: int):
        self.values = np.zeros(n_channels)

    def update(self, x: np.ndarray, visible: bool) -> None:
        if visible:
            self.values = np.asarray(x, dtype=np.float64).copy()

    def snapshot(self) -> np.ndarray:
        return self.values.copy()

    def reset(self) -> None:
        self.values[:] = 0.0
