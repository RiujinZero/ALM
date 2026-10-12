"""Mutual constraint settling between learned access and cortical returns.

The episodic and cortical organs remain independently useful.  This field only
lets their *already learned* relation-bound activities constrain one another
before one return is expressed.  It owns no semantic labels, output head, or
autonomous teaching signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReciprocalSettlingResult:
    activity: np.ndarray
    cortical_distribution: np.ndarray
    iterations: int
    active_traces: int
    entropy: float


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values, initial=0.0))
    weights = np.exp(np.clip(shifted, -80.0, 0.0))
    return weights / max(float(np.sum(weights)), 1e-300)


def _row_normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def settle_relational_returns(
    access_drive: np.ndarray,
    return_distributions: np.ndarray,
    cortical_distribution: np.ndarray,
    *,
    access_signatures: np.ndarray | None = None,
    steps: int = 6,
    access_gain: float = 12.0,
    cortical_gain: float = 4.0,
    recurrent_gain: float = 1.5,
    conflict_gain: float = 2.0,
    return_gain: float = 1.0,
    selective_competition: bool = True,
) -> ReciprocalSettlingResult:
    """Settle compatible access/return pairs without selecting a keyed item.

    All candidate traces receive the query in parallel.  Partial returned
    activity changes cortex; cortex feeds back to those same native return
    pathways; compatible traces cooperate and incompatible traces inhibit.
    The operation is inference-only.  Learning remains tied to later external
    observations in the owning access and expression organs.
    """
    drive = np.asarray(access_drive, dtype=np.float64)
    returns = np.asarray(return_distributions, dtype=np.float64)
    cortex = np.asarray(cortical_distribution, dtype=np.float64)
    if drive.ndim != 1:
        raise ValueError("access drive must be one-dimensional")
    if returns.shape != (len(drive), len(cortex)):
        raise ValueError("one return distribution is required per access trace")
    if steps < 1:
        raise ValueError("settling steps must be positive")
    if not len(drive):
        return ReciprocalSettlingResult(
            drive.copy(), cortex.copy(), 0, 0, 0.0
        )

    drive_scale = max(float(np.max(np.abs(drive))), 1e-12)
    normalized_drive = drive / drive_scale
    return_geometry = _row_normalize(returns)
    if access_signatures is None:
        access_geometry = np.eye(len(drive), dtype=np.float64)
    else:
        signatures = np.asarray(access_signatures, dtype=np.float64)
        if signatures.shape[0] != len(drive):
            raise ValueError("one access signature is required per trace")
        access_geometry = _row_normalize(signatures) @ _row_normalize(
            signatures
        ).T
        access_geometry = np.clip(access_geometry, 0.0, 1.0)
    return_similarity = np.clip(
        return_geometry @ return_geometry.T, 0.0, 1.0
    )
    cooperation = access_geometry * return_similarity
    conflict = access_geometry * (1.0 - return_similarity)
    np.fill_diagonal(cooperation, 0.0)
    np.fill_diagonal(conflict, 0.0)

    activity = _softmax(access_gain * normalized_drive)
    baseline = cortex / max(float(np.sum(cortex)), 1e-300)
    for _ in range(steps):
        recalled = activity @ returns
        cortex = baseline + return_gain * recalled
        cortex /= max(float(np.sum(cortex)), 1e-300)
        # Square-root overlap is sensitive to low-mass identity distinctions
        # without treating the cortex's current argmax as ground truth.
        feedback = np.sum(
            np.sqrt(np.maximum(returns, 0.0) * cortex[None, :]), axis=1
        )
        recurrent = recurrent_gain * (cooperation @ activity)
        if selective_competition:
            recurrent -= conflict_gain * (conflict @ activity)
            activity = _softmax(
                access_gain * normalized_drive
                + cortical_gain * feedback
                + recurrent
            )
        else:
            logits = (
                access_gain * normalized_drive
                + cortical_gain * feedback
                + recurrent
            )
            activity = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
            activity /= max(float(np.sum(activity)), 1e-300)
    entropy = -float(
        np.sum(activity * np.log2(np.maximum(activity, 1e-300)))
    )
    return ReciprocalSettlingResult(
        activity,
        cortex,
        steps,
        int(np.count_nonzero(activity > 1e-3)),
        entropy,
    )
