from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.reciprocal_relational_coupling import (
    settle_relational_returns,
)


def test_complementary_organs_resolve_ambiguous_relation() -> None:
    access = np.asarray([1.0, 1.0, 0.25, 0.25])
    returns = np.asarray(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ]
    )
    cortex = np.asarray([0.65, 0.35, 0.0, 0.0])
    signatures = np.asarray(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )

    result = settle_relational_returns(
        access, returns, cortex, access_signatures=signatures
    )

    assert int(np.argmax(result.activity)) == 0
    assert result.activity[0] > result.activity[1]
    assert result.active_traces < len(access)


def test_competition_prevents_uniform_pooling() -> None:
    access = np.ones(8)
    returns = np.eye(8)
    cortex = np.asarray([0.7, *([0.3 / 7] * 7)])
    signatures = np.ones((8, 4))

    selective = settle_relational_returns(
        access, returns, cortex, access_signatures=signatures
    )
    pooled = settle_relational_returns(
        access,
        returns,
        cortex,
        access_signatures=signatures,
        selective_competition=False,
    )

    assert selective.activity[0] > pooled.activity[0]
    assert selective.entropy < pooled.entropy


def test_input_arrays_are_not_mutated() -> None:
    access = np.asarray([0.8, 0.4])
    returns = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    cortex = np.asarray([0.6, 0.4])
    originals = tuple(value.copy() for value in (access, returns, cortex))

    settle_relational_returns(access, returns, cortex)

    for value, original in zip((access, returns, cortex), originals, strict=True):
        np.testing.assert_array_equal(value, original)
