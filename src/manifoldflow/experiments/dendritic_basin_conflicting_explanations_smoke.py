"""Test learned organ routing when several organs offer plausible contradictions.

This is the adversarial sibling of the clean competitive-resonance smoke.  Each
episode contains four simultaneously decodable processes.  One process is the
externally confirmed continuation while the other three agree on a different,
plausible continuation.  No process identity is supplied to the organs or the
route learner; the only reliability cue is the integrity of the process in the
shared cortical episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.experiments.dendritic_basin_competitive_resonance_routing_smoke import (  # noqa: E501
    FAMILIES,
    Organ,
    _fit_organ,
    _fit_ridge,
    _organ_outputs,
    _relation_bytes,
    _state,
    generate_process_batch,
    organ_evidence,
)
from manifoldflow.experiments.dendritic_basin_echolocation_smoke import _cupy
from manifoldflow.experiments.dendritic_basin_three_organ_echolocation_smoke import (
    DEFAULT_BODY_BYTES,
)


@dataclass(frozen=True)
class ConflictBatch:
    sequences: np.ndarray
    targets: np.ndarray
    reliable_organs: np.ndarray
    organ_targets: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--train-per-family", type=int, default=1200)
    parser.add_argument("--candidate-per-family", type=int, default=300)
    parser.add_argument("--route-train", type=int, default=6000)
    parser.add_argument("--test-examples", type=int, default=4000)
    parser.add_argument("--decoy-scale", type=float, default=0.7)
    parser.add_argument("--relation-capacity", type=int, default=512)
    parser.add_argument("--growth-rounds", type=int, default=8)
    parser.add_argument("--nursery-size", type=int, default=256)
    parser.add_argument("--hill-steps", type=int, default=64)
    parser.add_argument("--fit-samples", type=int, default=6000)
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_BODY_BYTES)
    parser.add_argument("--columns", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.decoy_scale < 1.0:
        parser.error("--decoy-scale must be between zero and one")
    if args.columns != 1 or args.max_body_bytes != DEFAULT_BODY_BYTES:
        parser.error("this smoke is fixed to the ALM-64MB-1C body class")
    return args


def _family_target_pools(count: int, seed: int) -> list[list[np.ndarray]]:
    """Create raw episode pools indexed only by evaluator family and target."""
    # Oversampling makes every family/target bucket comfortably large.
    raw = generate_process_batch(max(count * 12, 512), seed)
    pools: list[list[list[np.ndarray]]] = [[[] for _ in range(8)] for _ in FAMILIES]
    for sequence, target, family in zip(
        raw.sequences, raw.targets, raw.families, strict=True
    ):
        pools[int(family)][int(target)].append(sequence)
    return [
        [np.asarray(bucket, dtype=np.float32) for bucket in family] for family in pools
    ]


def generate_conflict_batch(
    examples: int, seed: int, *, decoy_scale: float
) -> ConflictBatch:
    """Combine four valid processes so three organs support the same decoy.

    The reliable process has full amplitude.  The three decoys remain valid
    process instances but arrive with reduced integrity.  Which geometry is
    reliable is balanced and random; it is never appended to the input.
    """
    rng = np.random.default_rng(seed)
    pools = _family_target_pools(max(32, examples // 16 + 1), seed + 101)
    sequences = []
    targets = []
    reliable_organs = []
    organ_targets = []
    for _ in range(examples):
        reliable = int(rng.integers(len(FAMILIES)))
        target = int(rng.integers(8))
        decoy = (target + int(rng.integers(1, 8))) % 8
        components = []
        per_organ = np.full(len(FAMILIES), decoy, dtype=np.int64)
        per_organ[reliable] = target
        for family in range(len(FAMILIES)):
            bucket = pools[family][int(per_organ[family])]
            components.append(bucket[int(rng.integers(len(bucket)))])
        scales = np.full(len(FAMILIES), decoy_scale, dtype=np.float32)
        scales[reliable] = 1.0
        # Dendritic current is bounded per receptor.  A max-union preserves the
        # integrity/amplitude of every coincident explanation; summing followed
        # by clipping would erase that information wherever components overlap.
        mixed = np.maximum.reduce(
            [
                scale * component
                for scale, component in zip(scales, components, strict=True)
            ]
        )
        sequences.append(mixed)
        targets.append(target)
        reliable_organs.append(reliable)
        organ_targets.append(per_organ)
    return ConflictBatch(
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(reliable_organs, dtype=np.int64),
        np.asarray(organ_targets, dtype=np.int64),
    )


def calibrate_resonance(organ: Organ, batch: ConflictBatch, cp: Any) -> Organ:
    """Learn future disagreement from the organ's current local state."""
    state = _state(organ_evidence(batch.sequences, organ.name, cp), organ.relations, cp)
    prediction = state @ organ.continuation_beta[:-1] + organ.continuation_beta[-1]
    external = cp.eye(8, dtype=cp.float32)[cp.asarray(batch.targets)]
    disagreement = cp.mean((prediction - external) ** 2, axis=1, keepdims=True)
    resonance_beta = _fit_ridge(state, disagreement, 10.0, cp)
    return replace(organ, resonance_beta=resonance_beta)


def evaluate_conflict(
    organs: list[Organ],
    batch: ConflictBatch,
    cp: Any,
    *,
    shuffle_seed: int | None = None,
) -> dict[str, Any]:
    predictions = []
    disagreements = []
    for organ in organs:
        prediction, disagreement = _organ_outputs(organ, batch.sequences, cp)
        predictions.append(prediction)
        disagreements.append(disagreement)
    prediction_stack = cp.stack(predictions, axis=1)
    disagreement_stack = cp.stack(disagreements, axis=1)
    if shuffle_seed is not None:
        rng = cp.random.RandomState(shuffle_seed)
        for column in range(len(organs)):
            disagreement_stack[:, column] = disagreement_stack[
                rng.permutation(len(batch.targets)), column
            ]
    winner = cp.argmin(disagreement_stack, axis=1)
    selected = prediction_stack[cp.arange(len(batch.targets)), winner]
    selected_class = cp.argmax(selected, axis=1)
    uniform_class = cp.argmax(cp.mean(prediction_stack, axis=1), axis=1)
    # A deliberately competition-free path sums all confident explanations.
    unnormalized_class = cp.argmax(cp.sum(prediction_stack, axis=1), axis=1)
    target = cp.asarray(batch.targets)
    reliable = cp.asarray(batch.reliable_organs)
    organ_target = cp.asarray(batch.organ_targets)
    organ_classes = cp.argmax(prediction_stack, axis=2)
    own_proposal_accuracy = cp.mean(organ_classes == organ_target)
    row = cp.arange(len(batch.targets))
    reliable_proposal_accuracy = cp.mean(organ_classes[row, reliable] == target)
    decoy_mask = cp.arange(len(FAMILIES))[None, :] != reliable[:, None]
    decoy_proposal_accuracy = cp.mean((organ_classes == organ_target)[decoy_mask])
    route_matrix = np.zeros((len(FAMILIES), len(FAMILIES)), dtype=np.int64)
    winner_cpu = cp.asnumpy(winner)
    for expected in range(len(FAMILIES)):
        mask = batch.reliable_organs == expected
        route_matrix[expected] = np.bincount(winner_cpu[mask], minlength=len(FAMILIES))
    return {
        "examples": len(batch.targets),
        "competitive_continuation_accuracy": float(
            cp.mean(selected_class == target).get()
        ),
        "reliable_organ_route_accuracy": float(cp.mean(winner == reliable).get()),
        "uniform_pool_accuracy": float(cp.mean(uniform_class == target).get()),
        "competition_removed_accuracy": float(
            cp.mean(unnormalized_class == target).get()
        ),
        "organ_own_proposal_accuracy": float(own_proposal_accuracy.get()),
        "reliable_organ_proposal_accuracy": float(reliable_proposal_accuracy.get()),
        "decoy_organ_proposal_accuracy": float(decoy_proposal_accuracy.get()),
        "route_matrix": route_matrix.tolist(),
    }


def run_seed(args: argparse.Namespace, seed: int, cp: Any) -> dict[str, Any]:
    clean_train = generate_process_batch(args.train_per_family, seed * 1009 + 11)
    clean_candidate = generate_process_batch(
        args.candidate_per_family, seed * 1009 + 23
    )
    organs = [
        _fit_organ(name, clean_train, clean_candidate, args, seed * 101 + i * 17, cp)
        for i, name in enumerate(FAMILIES)
    ]
    route_train = generate_conflict_batch(
        args.route_train, seed * 1009 + 41, decoy_scale=args.decoy_scale
    )
    organs = [calibrate_resonance(organ, route_train, cp) for organ in organs]
    test = generate_conflict_batch(
        args.test_examples, seed * 1009 + 59, decoy_scale=args.decoy_scale
    )
    body_bytes = int(
        sum(
            _relation_bytes(organ.relations)
            + organ.continuation_beta.nbytes
            + organ.resonance_beta.nbytes
            for organ in organs
        )
    )
    if body_bytes > args.max_body_bytes:
        raise MemoryError("conflicting-explanations organism exceeds ALM-64MB ceiling")
    return {
        "seed": seed,
        "held_out": evaluate_conflict(organs, test, cp),
        "shuffled_resonance": evaluate_conflict(
            organs, test, cp, shuffle_seed=seed + 701
        ),
        "grown_body_bytes": body_bytes,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "competitive_continuation_accuracy",
        "reliable_organ_route_accuracy",
        "uniform_pool_accuracy",
        "competition_removed_accuracy",
        "organ_own_proposal_accuracy",
        "reliable_organ_proposal_accuracy",
        "decoy_organ_proposal_accuracy",
    )
    result: dict[str, Any] = {}
    for metric in metrics:
        values = np.asarray([row["held_out"][metric] for row in rows])
        result[metric] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
        }
    shuffled = np.asarray(
        [row["shuffled_resonance"]["competitive_continuation_accuracy"] for row in rows]
    )
    result["shuffled_resonance_accuracy"] = {
        "mean": float(shuffled.mean()),
        "min": float(shuffled.min()),
        "max": float(shuffled.max()),
        "std": float(shuffled.std()),
    }
    return result


def main() -> None:
    args = parse_args()
    cp = _cupy()
    rows = []
    for offset in range(args.seeds):
        seed = args.seed_offset + offset
        print(f"conflicting explanations seed {seed}", file=sys.stderr, flush=True)
        rows.append(run_seed(args, seed, cp))
    report = {
        "experiment": "alm_64mb_1c_conflicting_explanations_routing",
        "alm_name": "ALM-64MB-1C",
        "organs": list(FAMILIES),
        "causal_order_contract": (
            "past/current local state selects the route; only later external "
            "observations train resonance and no family identity enters development "
            "or inference"
        ),
        "conflict_contract": (
            "all organs receive valid process evidence; the three unreliable organs "
            "agree on one plausible decoy while the reliable organ predicts the "
            "external future"
        ),
        "args": vars(args) | {"report": str(args.report) if args.report else None},
        "aggregate": aggregate(rows),
        "seeds": rows,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
