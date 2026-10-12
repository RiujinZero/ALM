"""Tests for recursive relation-tree composition on parity (design doc
sections 4-5, 10 of `Dendritic_Basin_Learning_Design_v0.2_Interaction_Dendrites.md`).
Kept to small bit-widths so CI stays fast; the larger-scale validation
(128-1024 bits, matching the design's own reported scale) is run via
`experiments/dendritic_basin_parity_ladder.py` and recorded in the
implementation journal, not re-run here on every test invocation.
"""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.relation_tree import (
    ComposedNode,
    LeafNode,
    build_relation_tree,
    evaluate_node,
    evaluate_tree_accuracy,
    train_relation_tree,
)


def test_build_relation_tree_rejects_non_power_of_two() -> None:
    with pytest.raises(ValueError):
        build_relation_tree(leaf_count=5, seed=0)


def test_build_relation_tree_has_n_minus_one_nodes() -> None:
    _root, levels = build_relation_tree(leaf_count=8, seed=0)
    total_nodes = sum(len(level) for level in levels)
    assert total_nodes == 7  # N-1 relation units for N=8 leaves


def test_build_relation_tree_levels_are_bottom_up() -> None:
    _root, levels = build_relation_tree(leaf_count=8, seed=0)
    assert len(levels) == 3  # log2(8)
    assert len(levels[0]) == 4  # directly above the leaves
    assert len(levels[-1]) == 1  # the root


def test_evaluate_node_on_a_leaf_reads_the_bit_directly() -> None:
    leaf = LeafNode(index=2)
    bits = np.array([0.0, 1.0, 1.0, 0.0])
    assert evaluate_node(leaf, bits) == 1.0


def test_4bit_parity_tree_trains_to_high_accuracy() -> None:
    rng = np.random.default_rng(0)
    root, levels = build_relation_tree(leaf_count=4, seed=1)

    unresolved = train_relation_tree(
        levels, n_bits=4, rounds=60, perturbations_per_round=12, step_size=0.3, rng=rng
    )

    assert unresolved == 0
    accuracy = evaluate_tree_accuracy(root, n_bits=4, n_eval=300, rng=rng)
    assert accuracy >= 0.95


def test_8bit_parity_tree_trains_to_high_accuracy() -> None:
    rng = np.random.default_rng(2)
    root, levels = build_relation_tree(leaf_count=8, seed=3)

    unresolved = train_relation_tree(
        levels, n_bits=8, rounds=60, perturbations_per_round=12, step_size=0.3, rng=rng
    )

    assert unresolved == 0
    accuracy = evaluate_tree_accuracy(root, n_bits=8, n_eval=300, rng=rng)
    assert accuracy >= 0.90


def test_evaluate_tree_accuracy_is_complement_invariant() -> None:
    """A "relation unit" that exactly XNORs its two leaf inputs (i.e. the
    precise complement of 2-bit parity, XOR) must still score perfectly,
    since either preserves the information up to a global complement
    (section 6/10).
    """

    class _ExactXnorUnit:
        def respond(self, a: float, b: float) -> float:
            return 1.0 if a == b else 0.0  # A XNOR B: the complement of A XOR B

    root = ComposedNode(unit=_ExactXnorUnit(), left=LeafNode(0), right=LeafNode(1))
    rng = np.random.default_rng(4)
    accuracy = evaluate_tree_accuracy(root, n_bits=2, n_eval=200, rng=rng)
    assert accuracy == 1.0
