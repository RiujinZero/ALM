from __future__ import annotations

import numpy as np

from manifoldflow.experiments.dendritic_basin_phase80_expression_ceiling import (
    _common_prefix,
    _first_discriminating_position,
    _first_word,
    _margin,
)


def test_controlled_identity_helpers_locate_the_first_real_choice() -> None:
    expected = b" red ball"
    contrast = b" blue kite"

    shared = _common_prefix(expected, contrast)

    assert shared == b" "
    assert _first_word(expected[len(shared) :]) == b"red"
    assert _first_discriminating_position(
        expected[len(shared) :], contrast[len(shared) :]
    ) == 0


def test_greedy_margin_requires_the_target_to_win() -> None:
    distribution = np.full(256, 1e-6, dtype=np.float64)
    distribution[ord("r")] = 0.3
    distribution[ord("b")] = 0.6
    distribution /= distribution.sum()

    margin, rank, winner_count = _margin(distribution, ord("r"))

    assert margin < 0.0
    assert rank == 2
    assert winner_count == 1
