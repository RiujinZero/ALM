from __future__ import annotations

import numpy as np
import torch

from manifoldflow.experiments.transformer_echolocation_size_control import (
    ALM_GROWN_BODY_BYTES,
    CausalSensoryTransformer,
    parameter_bytes,
)


def test_transformer_resident_parameters_match_grown_alm_body() -> None:
    model = CausalSensoryTransformer()
    assert sum(parameter.numel() for parameter in model.parameters()) == 24_802
    assert parameter_bytes(model) == 99_208
    mismatch = abs(parameter_bytes(model) - ALM_GROWN_BODY_BYTES)
    assert mismatch / ALM_GROWN_BODY_BYTES < 0.01


def test_transformer_emits_one_causal_position_per_sensory_step() -> None:
    torch.manual_seed(3)
    model = CausalSensoryTransformer()
    vectors = torch.from_numpy(np.zeros((2, 40, 5), dtype=np.float32))
    assert model(vectors).shape == (2, 40)


def test_causal_mask_prevents_future_sensory_changes_from_altering_past() -> None:
    torch.manual_seed(5)
    model = CausalSensoryTransformer()
    first = torch.zeros((1, 40, 5))
    second = first.clone()
    second[:, 20:] = 1.0
    with torch.no_grad():
        first_output = model(first)
        second_output = model(second)
    torch.testing.assert_close(first_output[:, :20], second_output[:, :20])
