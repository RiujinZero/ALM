"""Causal smoke for relation-specific reciprocal settling across two organs.

Each query is deliberately underdetermined in either organ alone.  A structural
cue narrows memory to four episodes, while a cortical filler cue narrows output
to two identities.  Their intersection identifies one learned binding.  No
episode, entity, relation, or filler label is supplied to the settling rule;
those names exist only in the generator and evaluator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RelationalWorld:
    access: np.ndarray
    returns: np.ndarray
    queries: np.ndarray
    identity_queries: np.ndarray
    local_fillers: np.ndarray
    neutral_fillers: np.ndarray
    targets: np.ndarray
    access_similarity: np.ndarray
    return_similarity: np.ndarray
    unique_slice: slice


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / np.sum(result, axis=1, keepdims=True)


def build_world(seed: int) -> RelationalWorld:
    """Create balanced swapped assignments with complementary partial cues."""
    rng = np.random.default_rng(seed)
    groups = 8
    contexts_per_group = 4
    entities = 4
    relations = 2
    object_categories = contexts_per_group
    objects_per_category = 2
    objects = object_categories * objects_per_category
    contexts = groups * contexts_per_group

    # Geometry is fixed, but meanings and assignments are independently
    # permuted each seed. Scaling represents receptor multiplicity, not labels.
    group_code = 2.0 * np.eye(groups, dtype=np.float64)
    unique_code = np.eye(contexts, dtype=np.float64)
    entity_code = 4.0 * np.eye(entities, dtype=np.float64)
    relation_code = 4.0 * np.eye(relations, dtype=np.float64)
    object_order = rng.permutation(objects)

    access: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    identity_queries: list[np.ndarray] = []
    local_fillers: list[np.ndarray] = []
    targets: list[int] = []
    category_permutations = np.asarray(
        [rng.permutation(object_categories) for _ in range(groups)]
    )
    member_permutations = np.asarray(
        [rng.permutation(objects_per_category) for _ in range(contexts)]
    )

    unique_start = groups
    unique_stop = unique_start + contexts
    for context in range(contexts):
        group = context // contexts_per_group
        within_group = context % contexts_per_group
        for relation in range(relations):
            for entity in range(entities):
                category = int(category_permutations[group, within_group])
                member = int(
                    member_permutations[context, (entity + relation) % 2]
                )
                object_id = int(object_order[category * objects_per_category + member])
                structural = np.concatenate(
                    (
                        group_code[group],
                        unique_code[context],
                        entity_code[entity],
                        relation_code[relation],
                    )
                )
                partial = structural.copy()
                partial[unique_start:unique_stop] = 0.0
                filler = np.zeros(objects, dtype=np.float64)
                members = object_order[
                    category * objects_per_category :
                    (category + 1) * objects_per_category
                ]
                filler[members] = 0.5
                returned = np.zeros(objects, dtype=np.float64)
                returned[object_id] = 1.0
                access.append(structural)
                queries.append(partial)
                identity_queries.append(structural.copy())
                local_fillers.append(filler)
                returns.append(returned)
                targets.append(object_id)

    access_array = np.asarray(access)
    return_array = np.asarray(returns)
    access_norm = access_array / np.linalg.norm(access_array, axis=1, keepdims=True)
    return_norm = return_array / np.linalg.norm(return_array, axis=1, keepdims=True)
    return RelationalWorld(
        access=access_array,
        returns=return_array,
        queries=np.asarray(queries),
        identity_queries=np.asarray(identity_queries),
        local_fillers=np.asarray(local_fillers),
        neutral_fillers=np.full((len(targets), objects), 1.0 / objects),
        targets=np.asarray(targets, dtype=np.int64),
        access_similarity=access_norm @ access_norm.T,
        return_similarity=return_norm @ return_norm.T,
        unique_slice=slice(unique_start, unique_stop),
    )


def _access_drive(
    world: RelationalWorld,
    access: np.ndarray | None = None,
    queries: np.ndarray | None = None,
) -> np.ndarray:
    memory = world.access if access is None else access
    memory = memory / np.linalg.norm(memory, axis=1, keepdims=True)
    query = world.queries if queries is None else queries
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    return query @ memory.T


def _decode(cortex: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.argmax(cortex, axis=1) == targets))


def evaluate(
    world: RelationalWorld,
    *,
    reciprocal: bool,
    selective_competition: bool,
    scrambled_return_seed: int | None = None,
    staged_access_seed: int | None = None,
    developmental_probe: bool = False,
    steps: int = 8,
) -> dict[str, float]:
    rng = np.random.default_rng(scrambled_return_seed)
    returns = world.returns.copy()
    if scrambled_return_seed is not None:
        returns = returns[rng.permutation(len(returns))]

    access = world.access.copy()
    if staged_access_seed is not None:
        # Matched-current developmental-loss control. The same number and
        # magnitude of unique receptors survive, but their event alignment was
        # unavailable when the cross-organ relationship formed.
        staged_rng = np.random.default_rng(staged_access_seed)
        access[:, world.unique_slice] = access[
            staged_rng.permutation(len(access)), world.unique_slice
        ]

    queries = world.identity_queries if developmental_probe else world.queries
    local = (
        world.neutral_fillers.copy()
        if developmental_probe
        else world.local_fillers.copy()
    )
    drive = _access_drive(world, access, queries)
    h = _softmax(18.0 * drive)
    # One-way recall must commit using memory evidence before cortex sees the
    # return. Tiny target-independent jitter makes exact structural ties fair.
    rows = np.arange(len(drive), dtype=np.uint64)[:, None]
    columns = np.arange(len(access), dtype=np.uint64)[None, :]
    jitter = ((rows * 1103515245 + columns * 12345) % 104729) * 1e-12
    one_way_winner = np.argmax(drive + jitter, axis=1)
    one_way_h = np.zeros_like(h)
    one_way_h[np.arange(len(h)), one_way_winner] = 1.0
    one_way_cortex = local + 1.5 * (one_way_h @ returns)
    one_way_cortex /= np.sum(one_way_cortex, axis=1, keepdims=True)
    one_way_accuracy = _decode(one_way_cortex, world.targets)
    if not reciprocal:
        return {
            "accuracy": one_way_accuracy,
            "one_way_accuracy": one_way_accuracy,
            "mean_active_memories": float(np.mean(np.sum(h > 1e-3, axis=1))),
        }

    # Reciprocal dynamics begins before a memory has committed. Partial
    # returns may bias cortex, which may in turn resolve the memory population.
    cortex = local + 1.5 * (h @ returns)
    cortex /= np.sum(cortex, axis=1, keepdims=True)

    access_norm = access / np.linalg.norm(access, axis=1, keepdims=True)
    access_similarity = access_norm @ access_norm.T
    return_similarity = returns @ returns.T
    cooperation = access_similarity * return_similarity
    conflict = access_similarity * (1.0 - return_similarity)
    for _ in range(steps):
        feedback = cortex @ returns.T
        recurrent = h @ cooperation.T
        if selective_competition:
            recurrent -= 1.25 * (h @ conflict.T)
            logits = 18.0 * drive + 8.0 * feedback + 2.0 * recurrent
            h = _softmax(logits)
        else:
            # Removing divisive competition permits every independently
            # plausible explanation to remain active.
            logits = 5.0 * drive + 2.0 * feedback + recurrent
            h = 1.0 / (1.0 + np.exp(-logits))
            h /= max(1.0, float(np.mean(np.sum(h, axis=1))))
        cortex = local + 1.5 * (h @ returns)
        cortex /= np.sum(cortex, axis=1, keepdims=True)
    return {
        "accuracy": _decode(cortex, world.targets),
        "one_way_accuracy": one_way_accuracy,
        "correct_memory_fraction": float(
            np.mean(np.argmax(h, axis=1) == np.arange(len(h)))
        ),
        "mean_active_memories": float(np.mean(np.sum(h > 1e-3, axis=1))),
    }


def run_seed(seed: int) -> dict[str, Any]:
    world = build_world(seed)
    return {
        "seed": seed,
        "local_cortex_only_accuracy": _decode(
            world.local_fillers, world.targets
        ),
        "one_way": evaluate(
            world, reciprocal=False, selective_competition=True
        ),
        "reciprocal_selective": evaluate(
            world, reciprocal=True, selective_competition=True
        ),
        "reciprocal_without_competition": evaluate(
            world, reciprocal=True, selective_competition=False
        ),
        "relation_scrambled": evaluate(
            world,
            reciprocal=True,
            selective_competition=True,
            scrambled_return_seed=seed * 101 + 17,
        ),
        "staged_misaligned_access": evaluate(
            world,
            reciprocal=True,
            selective_competition=True,
            staged_access_seed=seed * 101 + 29,
            developmental_probe=True,
        ),
        "joint_aligned_access": evaluate(
            world,
            reciprocal=True,
            selective_competition=True,
            developmental_probe=True,
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    paths = {
        "local_cortex_only": ("local_cortex_only_accuracy",),
        "one_way": ("one_way", "accuracy"),
        "reciprocal_selective": ("reciprocal_selective", "accuracy"),
        "reciprocal_without_competition": (
            "reciprocal_without_competition",
            "accuracy",
        ),
        "relation_scrambled": ("relation_scrambled", "accuracy"),
        "staged_misaligned_access": ("staged_misaligned_access", "accuracy"),
        "joint_aligned_access": ("joint_aligned_access", "accuracy"),
    }
    result = {}
    for name, path in paths.items():
        values = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
        result[name] = float(np.mean(values))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [run_seed(args.seed_offset + offset) for offset in range(args.seeds)]
    report = {
        "experiment": "reciprocal_relational_constraint_settling",
        "causal_contract": (
            "partial structural and filler cues settle before the external target "
            "is scored; generated/recalled activity never updates coupling"
        ),
        "interpretive_limit": (
            "the staged control is a matched-current developmental-loss stress, "
            "not yet an online demonstration that ALM learns co-development"
        ),
        "aggregate": aggregate(rows),
        "seeds": rows,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
