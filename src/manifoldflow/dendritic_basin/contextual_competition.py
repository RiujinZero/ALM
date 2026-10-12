"""Shared-pool normalized competition for context-dependent influence.

Every earlier competitive mechanism in this package picks winners its own
way: kWTA thresholds, top-K depression, independent per-assembly clipping.
None of them share one pool-normalized denominator, so "how much influence
does this activity have relative to everything else competing for the same
downstream effect" has never been a single, reusable local quantity.

This module gives that quantity a name:

    d_i(t)  = (A_i(x_t) * E_i(t)) ** n
    r_i(t)  = d_i(t) / (kappa + sum_j d_j(t))

``E_i`` is incoming drive, ``A_i`` is contextual gain supplied by other
active content, ``n`` sets the competitive sharpness, and ``kappa`` is a
semisaturation baseline that lets weak total drive produce weak total
influence (``sum_i r_i < 1`` whenever ``sum_i d_i`` is small relative to
``kappa``) without special-casing "nothing is happening." Setting
``kappa = 0`` and ``d_i = exp(logit_i)`` recovers ordinary softmax exactly;
this is verified as a regression identity in the test suite, not just
asserted here.
"""

from __future__ import annotations

import numpy as np


def normalized_competition(
    drive: np.ndarray,
    gain: np.ndarray,
    *,
    exponent: float = 1.0,
    kappa: float = 1e-6,
) -> np.ndarray:
    """Return pool-normalized influence ``r`` for one competing population.

    ``drive`` and ``gain`` must be non-negative and broadcastable to the
    same shape. ``kappa`` must be positive so a population with zero total
    drive returns all-zero influence rather than dividing by zero.
    """
    drive = np.asarray(drive, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    if drive.shape != gain.shape:
        raise ValueError("drive and gain must share a shape")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    if np.any(drive < 0.0) or np.any(gain < 0.0):
        raise ValueError("drive and gain must be non-negative")
    excitation = drive * gain
    d = np.power(excitation, exponent)
    denominator = kappa + float(np.sum(d))
    if denominator <= 0.0:
        return np.zeros_like(d)
    return d / denominator


def competitive_readout(
    drive: np.ndarray,
    gain: np.ndarray,
    values: np.ndarray,
    *,
    exponent: float = 1.0,
    kappa: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(influence, y)`` where ``y = sum_i r_i * values_i``."""
    influence = normalized_competition(drive, gain, exponent=exponent, kappa=kappa)
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != influence.shape[0]:
        raise ValueError("values must have one row per competitor")
    y = np.tensordot(influence, values, axes=(0, 0))
    return influence, y
