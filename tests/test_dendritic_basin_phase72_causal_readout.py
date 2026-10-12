from __future__ import annotations

import numpy as np

from manifoldflow.experiments.dendritic_basin_phase72_causal_readout import (
    sampling_distribution,
)


def test_sampling_distribution_applies_top_k_and_nucleus_before_tv() -> None:
    distribution = np.arange(1, 257, dtype=np.float64)
    distribution /= distribution.sum()

    effective = sampling_distribution(
        distribution, temperature=0.8, top_k=16, top_p=0.9
    )

    assert np.isclose(effective.sum(), 1.0)
    assert np.count_nonzero(effective) <= 16


def test_greedy_sampling_distribution_is_one_hot() -> None:
    distribution = np.full(256, 1.0 / 256)
    distribution[17] += 0.1

    effective = sampling_distribution(
        distribution, temperature=0.0, top_k=32, top_p=0.9
    )

    assert effective[17] == 1.0
    assert np.count_nonzero(effective) == 1
