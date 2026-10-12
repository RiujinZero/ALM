"""Grow one raw-vector modular ALM until its durable body reaches 64 MB."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.tasks.modular_arithmetic import (
    ModularSplit,
    build_modular_ladder,
)
from manifoldflow.experiments.dendritic_basin_echolocation_smoke import (
    GrownRelations,
    _cupy,
    grow_relations,
    relation_activations,
)
from manifoldflow.experiments.dendritic_basin_three_organ_echolocation_smoke import (
    DEFAULT_BODY_BYTES,
    _relation_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operand-limit", type=int, default=32)
    parser.add_argument("--seen-moduli", type=int, nargs="+", default=[2, 3, 5, 7])
    parser.add_argument("--unseen-modulus", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first-capacity", type=int, default=4096)
    parser.add_argument("--first-nursery-size", type=int, default=1024)
    parser.add_argument("--first-growth-rounds", type=int, default=8)
    parser.add_argument("--first-hill-steps", type=int, default=64)
    parser.add_argument("--second-nursery-size", type=int, default=8192)
    parser.add_argument("--competition-rounds", type=int, default=8)
    parser.add_argument("--second-hill-steps", type=int, default=16)
    parser.add_argument("--feature-block", type=int, default=8192)
    parser.add_argument("--target-body-bytes", type=int, default=DEFAULT_BODY_BYTES)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.target_body_bytes != DEFAULT_BODY_BYTES:
        parser.error("this developmental run is fixed to ALM-64MB-1C")
    return args


def _growth_args(
    *, capacity: int, rounds: int, nursery: int, hill_steps: int, fit_samples: int
) -> argparse.Namespace:
    return argparse.Namespace(
        relation_capacity=capacity,
        growth_rounds=rounds,
        nursery_size=nursery,
        hill_steps=hill_steps,
        fit_samples=fit_samples,
    )


def _activation_block(
    features: Any, relations: GrownRelations, start: int, stop: int, cp: Any
) -> Any:
    a = features[:, relations.parent_a[start:stop]]
    b = features[:, relations.parent_b[start:stop]]
    drive = (
        relations.w_a[start:stop] * a
        + relations.w_b[start:stop] * b
        - relations.mu[start:stop]
    )
    return cp.exp(
        -0.5
        * (drive / cp.exp(relations.log_sigma[start:stop])[None, :]) ** 2
    )


def _feature_blocks(
    raw: Any,
    first_state: Any,
    second_input: Any,
    second: GrownRelations,
    block_size: int,
    cp: Any,
) -> Iterator[tuple[int, Any]]:
    yield 0, raw
    yield raw.shape[1], first_state
    offset = raw.shape[1] + first_state.shape[1]
    for start in range(0, len(second.score), block_size):
        stop = min(start + block_size, len(second.score))
        yield offset + start, _activation_block(
            second_input, second, start, stop, cp
        )


def _kernel_pair(
    train_raw: Any,
    candidate_raw: Any,
    first: GrownRelations,
    second: GrownRelations,
    block_size: int,
    cp: Any,
) -> tuple[Any, Any, list[Any], Any, Any, Any, Any]:
    train_first = relation_activations(train_raw, first, cp)
    candidate_first = relation_activations(candidate_raw, first, cp)
    train_second_input = cp.concatenate((train_raw, train_first), axis=1)
    candidate_second_input = cp.concatenate((candidate_raw, candidate_first), axis=1)
    kernel = cp.zeros((len(train_raw), len(train_raw)), dtype=cp.float32)
    cross = cp.zeros((len(candidate_raw), len(train_raw)), dtype=cp.float32)
    means: list[Any] = []
    train_blocks = _feature_blocks(
        train_raw, train_first, train_second_input, second, block_size, cp
    )
    candidate_blocks = _feature_blocks(
        candidate_raw,
        candidate_first,
        candidate_second_input,
        second,
        block_size,
        cp,
    )
    for (train_offset, train_block), (candidate_offset, candidate_block) in zip(
        train_blocks, candidate_blocks, strict=True
    ):
        if train_offset != candidate_offset:
            raise AssertionError("feature block alignment changed")
        mean = train_block.mean(axis=0)
        centered = train_block - mean
        kernel += centered @ centered.T
        cross += (candidate_block - mean) @ centered.T
        means.append(mean)
    return (
        kernel,
        cross,
        means,
        train_first,
        candidate_first,
        train_second_input,
        candidate_second_input,
    )


def _decode(logits: Any, moduli: np.ndarray, classes: int, cp: Any) -> Any:
    valid = cp.arange(classes)[None, :] < cp.asarray(moduli)[:, None]
    return cp.argmax(cp.where(valid, logits, -cp.inf), axis=1)


def _metrics(
    split: ModularSplit,
    raw: Any,
    first: GrownRelations,
    second: GrownRelations,
    weights: Any,
    bias: Any,
    block_size: int,
    classes: int,
    cp: Any,
) -> dict[str, Any]:
    first_state = relation_activations(raw, first, cp)
    second_input = cp.concatenate((raw, first_state), axis=1)
    logits = cp.broadcast_to(bias, (len(raw), classes)).copy()
    for offset, block in _feature_blocks(
        raw, first_state, second_input, second, block_size, cp
    ):
        logits += block @ weights[offset : offset + block.shape[1]]
    decoded = _decode(logits, split.moduli, classes, cp)
    truth = cp.asarray(split.residues)
    error = decoded - truth
    return {
        "examples": len(split.residues),
        "exact_accuracy": float(cp.mean(decoded == truth).get()),
        "integer_mae": float(cp.mean(cp.abs(error)).get()),
    }


def _numpy_relations(relations: GrownRelations) -> dict[str, np.ndarray]:
    return {
        name: getattr(relations, name).get()
        for name in (
            "parent_a",
            "parent_b",
            "w_a",
            "w_b",
            "mu",
            "log_sigma",
            "score",
        )
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cp = _cupy()
    ladder = build_modular_ladder(
        seen_moduli=tuple(args.seen_moduli),
        unseen_modulus=args.unseen_modulus,
        operand_limit=args.operand_limit,
        seed=1_000_003 * (args.seed + 1),
    )
    classes = max(*ladder.seen_moduli, ladder.unseen_modulus_value)
    relation_bytes = 2 * np.dtype(np.int64).itemsize + 5 * np.dtype(np.float32).itemsize
    readout_bytes = classes * np.dtype(np.float32).itemsize
    fixed_readout_bytes = 4 * readout_bytes
    relation_total = (
        args.target_body_bytes - fixed_readout_bytes
    ) // (relation_bytes + readout_bytes)
    second_capacity = relation_total - args.first_capacity
    if second_capacity < 1:
        raise ValueError("first-depth anatomy leaves no second-depth tissue")
    second_rounds = (
        math.ceil(second_capacity / args.second_nursery_size)
        + args.competition_rounds
    )
    train_raw = cp.asarray(ladder.train.inputs, dtype=cp.float32)
    candidate_raw = cp.asarray(ladder.candidate.inputs, dtype=cp.float32)
    started = time.perf_counter()
    print(
        f"growing first depth to {args.first_capacity:,} relations",
        file=sys.stderr,
        flush=True,
    )
    first = grow_relations(
        train_raw,
        _growth_args(
            capacity=args.first_capacity,
            rounds=args.first_growth_rounds,
            nursery=args.first_nursery_size,
            hill_steps=args.first_hill_steps,
            fit_samples=len(train_raw),
        ),
        args.seed + 101,
        cp,
    )
    train_first = relation_activations(train_raw, first, cp)
    second_input = cp.concatenate((train_raw, train_first), axis=1)
    print(
        f"growing second depth to {second_capacity:,} relations "
        f"across {second_rounds} rounds",
        file=sys.stderr,
        flush=True,
    )
    second = grow_relations(
        second_input,
        _growth_args(
            capacity=second_capacity,
            rounds=second_rounds,
            nursery=args.second_nursery_size,
            hill_steps=args.second_hill_steps,
            fit_samples=len(train_raw),
        ),
        args.seed + 211,
        cp,
    )
    del train_first, second_input
    cp.get_default_memory_pool().free_all_blocks()

    (
        kernel,
        candidate_cross,
        means,
        train_first,
        _candidate_first,
        train_second_input,
        _candidate_second_input,
    ) = _kernel_pair(
        train_raw,
        candidate_raw,
        first,
        second,
        args.feature_block,
        cp,
    )
    target = cp.eye(classes, dtype=cp.float32)[cp.asarray(ladder.train.residues)]
    centered_target = target - target.mean(axis=0)
    identity = cp.eye(len(train_raw), dtype=cp.float32)
    best: tuple[float, float, Any, float] | None = None
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        dual = cp.linalg.solve(kernel + alpha * identity, centered_target)
        logits = candidate_cross @ dual + target.mean(axis=0)
        decoded = _decode(logits, ladder.candidate.moduli, classes, cp)
        accuracy = float(
            cp.mean(decoded == cp.asarray(ladder.candidate.residues)).get()
        )
        mse = float(
            cp.mean(
                (logits - cp.eye(classes)[cp.asarray(ladder.candidate.residues)])
                ** 2
            ).get()
        )
        if best is None or (-accuracy, mse) < (-best[0], best[1]):
            best = (accuracy, mse, dual, alpha)
    assert best is not None
    dual = best[2]

    feature_count = 3 + len(first.score) + len(second.score)
    weights = cp.empty((feature_count, classes), dtype=cp.float32)
    bias = target.mean(axis=0).copy()
    mean_index = 0
    for offset, block in _feature_blocks(
        train_raw,
        train_first,
        train_second_input,
        second,
        args.feature_block,
        cp,
    ):
        centered = block - means[mean_index]
        block_weights = centered.T @ dual
        weights[offset : offset + block.shape[1]] = block_weights
        bias -= means[mean_index] @ block_weights
        mean_index += 1

    grown_body_bytes = (
        _relation_bytes(first)
        + _relation_bytes(second)
        + int(weights.nbytes + bias.nbytes)
    )
    if grown_body_bytes > args.target_body_bytes:
        raise MemoryError("grown body crossed the ALM-64MB ceiling")

    train_metrics = _metrics(
        ladder.train,
        train_raw,
        first,
        second,
        weights,
        bias,
        args.feature_block,
        classes,
        cp,
    )
    held_metrics = _metrics(
        ladder.held_out_pairs,
        cp.asarray(ladder.held_out_pairs.inputs),
        first,
        second,
        weights,
        bias,
        args.feature_block,
        classes,
        cp,
    )
    unseen_metrics = _metrics(
        ladder.unseen_modulus,
        cp.asarray(ladder.unseen_modulus.inputs),
        first,
        second,
        weights,
        bias,
        args.feature_block,
        classes,
        cp,
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    first_numpy = _numpy_relations(first)
    second_numpy = _numpy_relations(second)
    np.savez(
        args.checkpoint,
        **{f"first_{key}": value for key, value in first_numpy.items()},
        **{f"second_{key}": value for key, value in second_numpy.items()},
        readout_weights=weights.get(),
        readout_bias=bias.get(),
    )
    report = {
        "experiment": "alm_64mb_1c_modular_arithmetic_filled_body",
        "alm_name": "ALM-64MB-1C",
        "target_body_bytes": args.target_body_bytes,
        "grown_body_bytes": grown_body_bytes,
        "unused_body_bytes": args.target_body_bytes - grown_body_bytes,
        "body_fill_fraction": grown_body_bytes / args.target_body_bytes,
        "first_relations": len(first.score),
        "second_relations": len(second.score),
        "total_relations": len(first.score) + len(second.score),
        "candidate_selected_ridge": best[3],
        "candidate_exact_accuracy": best[0],
        "training_fit": train_metrics,
        "held_out_operand_combinations": held_metrics,
        "unseen_modulus": unseen_metrics,
        "unseen_modulus_chance": 1.0 / ladder.unseen_modulus_value,
        "development_seconds": elapsed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_file_bytes": args.checkpoint.stat().st_size,
        "args": vars(args)
        | {
            "report": str(args.report),
            "checkpoint": str(args.checkpoint),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
