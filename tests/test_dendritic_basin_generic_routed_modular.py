from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.modular_arithmetic import build_modular_ladder
from manifoldflow.experiments.dendritic_basin_generic_routed_modular_smoke import (
    GEOMETRIES,
    _compact_prefixes,
    development_examples,
    generic_byte_evidence,
)


def test_generic_organs_receive_same_unparsed_byte_stream() -> None:
    ladder = build_modular_ladder(
        operand_limit=8, seed=7, partition_by_operand_pair=True
    )
    expressions = _compact_prefixes(ladder.train)
    assert expressions
    assert all(expression.endswith(b"= ") for expression in expressions)
    widths = {
        geometry: generic_byte_evidence(expressions[:4], geometry, np).shape
        for geometry in GEOMETRIES
    }
    assert widths == {
        "persistent": (4, 256),
        "coincident": (4, 256),
        "ordered": (4, 256),
        "slow": (4, 512),
    }


def test_mixed_curriculum_adds_raw_lessons_without_changing_targets() -> None:
    ladder = build_modular_ladder(
        operand_limit=8, seed=11, partition_by_operand_pair=True
    )
    endpoint_x, endpoint_y = development_examples(ladder.train, "endpoint")
    mixed_x, mixed_y = development_examples(ladder.train, "mixed")
    assert mixed_x[: len(endpoint_x)] == endpoint_x
    assert len(mixed_x) == 2 * len(endpoint_x)
    np.testing.assert_array_equal(mixed_y[: len(endpoint_y)], endpoint_y)
    np.testing.assert_array_equal(mixed_y[len(endpoint_y) :], endpoint_y)
    assert all(b"ring " in expression for expression in mixed_x[len(endpoint_x) :])


def test_operand_pair_partition_is_disjoint_across_every_modulus() -> None:
    ladder = build_modular_ladder(
        operand_limit=12, seed=13, partition_by_operand_pair=True
    )
    train_pairs = {tuple(row) for row in ladder.train.operands.tolist()}
    candidate_pairs = {tuple(row) for row in ladder.candidate.operands.tolist()}
    held_out_pairs = {tuple(row) for row in ladder.held_out_pairs.operands.tolist()}
    assert train_pairs.isdisjoint(candidate_pairs)
    assert train_pairs.isdisjoint(held_out_pairs)
    assert candidate_pairs.isdisjoint(held_out_pairs)
