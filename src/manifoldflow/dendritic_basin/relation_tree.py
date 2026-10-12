"""Recursive relation-on-relation composition (design doc section 5 of
`Dendritic_Basin_Learning_Design_v0.2_Interaction_Dendrites.md`; sections
5, 10, 17 of `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`):
`r_1 = f(A, B)`, then `r_2 = f(r_1, C)`, turning parity from an
exponential (`2^N`) pattern-memorization problem into `N-1` independently
locally-learned relation units over a binary tree.

Training is bottom-up: a level's relation units only see valid (a, b)
samples once every unit below them has already been trained, since a
unit's inputs are its children's *outputs* on random full inputs, not raw
leaves once above the first level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldflow.dendritic_basin.relation_unit import RelationUnit
from manifoldflow.dendritic_basin.tasks import parity as parity_task


@dataclass
class LeafNode:
    index: int


@dataclass
class ComposedNode:
    unit: RelationUnit
    left: LeafNode | ComposedNode
    right: LeafNode | ComposedNode


def evaluate_node(node: LeafNode | ComposedNode, bits: np.ndarray) -> float:
    if isinstance(node, LeafNode):
        return float(bits[node.index])
    a = evaluate_node(node.left, bits)
    b = evaluate_node(node.right, bits)
    return node.unit.respond(a, b)


def build_relation_tree(
    leaf_count: int,
    seed: int,
    n_candidates: int = 2,
    acceptance_threshold: float = 0.20,
) -> tuple[ComposedNode, list[list[ComposedNode]]]:
    """A binary tree of `leaf_count - 1` RelationUnits over `leaf_count`
    leaves (`leaf_count` must be a power of two). Returns (root, levels),
    with `levels[0]` the level directly above the leaves and `levels[-1]`
    the single root-containing level -- i.e. bottom-up training order.
    """
    if leaf_count < 2 or (leaf_count & (leaf_count - 1)) != 0:
        raise ValueError(f"leaf_count must be a power of two >= 2, got {leaf_count}")

    nodes: list[LeafNode | ComposedNode] = [LeafNode(i) for i in range(leaf_count)]
    levels: list[list[ComposedNode]] = []
    unit_id = 0
    while len(nodes) > 1:
        next_level: list[ComposedNode] = []
        for i in range(0, len(nodes), 2):
            unit = RelationUnit(
                n_candidates=n_candidates,
                seed=seed + unit_id,
                acceptance_threshold=acceptance_threshold,
            )
            next_level.append(ComposedNode(unit, nodes[i], nodes[i + 1]))
            unit_id += 1
        levels.append(next_level)
        nodes = list(next_level)
    return nodes[0], levels


def _make_draw_batch(node: ComposedNode, n_bits: int):
    def draw_batch(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
        samples = []
        for _ in range(n):
            bits, _target = parity_task.sample_bits(n_bits, rng)
            a = evaluate_node(node.left, bits)
            b = evaluate_node(node.right, bits)
            samples.append((a, b))
        return samples

    return draw_batch


def train_relation_tree(
    levels: list[list[ComposedNode]],
    n_bits: int,
    rounds: int,
    perturbations_per_round: int,
    step_size: float,
    rng: np.random.Generator,
    eval_batch_size: int = 200,
    max_episodes_per_node: int = 5,
) -> int:
    """Train every node bottom-up. A node whose unit doesn't clear the
    quality gate on the first episode is retried (fresh candidate
    dendrites, same position in the tree) up to `max_episodes_per_node`
    times -- matching the design's own reported protocol that a single
    episode is not always enough, but a retry reliably is (see
    docs/ManifoldFlow_ALM_Journal.md's XOR-reliability findings). Returns
    the number of nodes that never cleared the gate within the retry
    budget (0 in the fully successful case).
    """
    unresolved = 0
    for level in levels:
        for node in level:
            draw_batch = _make_draw_batch(node, n_bits)
            n_candidates = len(node.unit.candidates)
            threshold = node.unit.acceptance_threshold

            accepted = False
            for episode in range(max_episodes_per_node):
                if episode > 0:
                    fresh_seed = int(rng.integers(0, 2**31))
                    node.unit = RelationUnit(
                        n_candidates=n_candidates,
                        seed=fresh_seed,
                        acceptance_threshold=threshold,
                    )
                node.unit.adapt(
                    draw_batch,
                    rounds,
                    perturbations_per_round,
                    step_size,
                    rng,
                    eval_batch_size,
                )
                if node.unit.is_accepted():
                    accepted = True
                    break
            if not accepted:
                unresolved += 1
    return unresolved


def evaluate_tree_accuracy(
    root: ComposedNode, n_bits: int, n_eval: int, rng: np.random.Generator
) -> float:
    """Accuracy up to a global complement (section 6/10: whether a node
    lands on XOR or XNOR doesn't matter for overall parity, since either
    preserves the information up to inversion -- so predicting the
    complement everywhere is exactly as good as predicting directly).
    """
    correct = 0
    for _ in range(n_eval):
        bits, target = parity_task.sample_bits(n_bits, rng)
        output = evaluate_node(root, bits)
        predicted = int(output > 0.5)
        correct += int(predicted == target)
    direct_accuracy = correct / n_eval
    return max(direct_accuracy, 1.0 - direct_accuracy)
