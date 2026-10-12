from __future__ import annotations

import numpy as np
import torch

from manifoldflow.experiments.transformer_moving_point_size_control import (
    ALM_GROWN_BODY_BYTES,
    RingCausalTransformer,
    _loss,
    episode_arrays,
    parameter_bytes,
)


def test_transformer_body_matches_grown_alm_within_one_percent() -> None:
    model = RingCausalTransformer()
    assert parameter_bytes(model) == 99_312
    mismatch = abs(parameter_bytes(model) - ALM_GROWN_BODY_BYTES) / ALM_GROWN_BODY_BYTES
    assert mismatch < 0.01


def test_episode_arrays_preserve_reset_and_ring_motion() -> None:
    observations, positions = episode_arrays(
        12, ring_size=8, steps_per_episode=6, seed=7
    )
    assert observations.shape == (12, 7, 9)
    assert positions.shape == (12, 6)
    assert np.all(observations[:, 0, 0] == 1.0)
    differences = (positions[:, 1:] - positions[:, :-1]) % 8
    assert np.all((differences == 1) | (differences == 7))
    assert np.all(differences == differences[:, :1])


def test_training_loss_omits_unpredictable_reset_to_start_transition() -> None:
    model = RingCausalTransformer()
    observations, positions = episode_arrays(
        3, ring_size=8, steps_per_episode=6, seed=11
    )
    loss = _loss(model, torch.from_numpy(observations), torch.from_numpy(positions))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
