"""Conventional expression-structured modular arithmetic on ALM-64MB-1C."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)
from manifoldflow.dendritic_basin.tasks.modular_arithmetic import (
    ModularSplit,
    build_modular_ladder,
    modular_expression,
    modular_trajectory_prefix,
    unfolded_modular_lesson,
)
from manifoldflow.experiments.dendritic_basin_three_organ_echolocation_smoke import (
    DEFAULT_BODY_BYTES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operand-limit", type=int, default=32)
    parser.add_argument("--seen-moduli", type=int, nargs="+", default=[2, 3, 5, 7])
    parser.add_argument("--unseen-modulus", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--stable-epochs", type=int, default=3)
    parser.add_argument("--mask-probability", type=float, default=0.0)
    parser.add_argument(
        "--curriculum", choices=("endpoint", "unfolded"), default="endpoint"
    )
    parser.add_argument("--max-context-bytes", type=int, default=20)
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_BODY_BYTES)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.mask_probability <= 1.0:
        parser.error("mask-probability must be in [0, 1]")
    return args


def _develop_epoch(
    learner: SequenceCortexOpponentLearner,
    split: ModularSplit,
    order: np.ndarray,
    rng: np.random.Generator,
    mask_probability: float,
    curriculum: str,
) -> tuple[int, int, dict[str, int]]:
    observed = 0
    masked = 0
    masked_fields = {"a": 0, "b": 0, "n": 0}
    for raw_index in order:
        index = int(raw_index)
        a, b = split.operands[index]
        n = int(split.moduli[index])
        c = int(split.residues[index])
        mask_fields: frozenset[str] = frozenset()
        if rng.random() < mask_probability:
            field = str(rng.choice(("a", "b", "n")))
            mask_fields = frozenset((field,))
            masked += 1
            masked_fields[field] += 1
        if curriculum == "unfolded":
            if mask_fields:
                raise ValueError("field masking and unfolded lessons are separate arms")
            expression = unfolded_modular_lesson(int(a), int(b), n, c)
        else:
            expression = modular_expression(
                int(a), int(b), n, c, mask_fields=mask_fields
            )
        learner.reset_episode()
        learner.observe_chunk_probabilities(
            np.frombuffer(expression, dtype=np.uint8), learn=True
        )
        observed += len(expression)
    learner.reset_episode()
    return observed, masked, masked_fields


def _generate_answer(
    learner: SequenceCortexOpponentLearner,
    a: int,
    b: int,
    n: int,
    *,
    mask_fields: frozenset[str] = frozenset(),
) -> str:
    learner.reset_episode()
    prefix = modular_expression(a, b, n, mask_fields=mask_fields)
    learner.observe_chunk_probabilities(
        np.frombuffer(prefix, dtype=np.uint8), learn=False
    )
    generated = bytearray()
    for _ in range(3):
        byte_id = int(np.argmax(learner.predict_next_distribution()))
        learner.advance_observation(byte_id, learn=False)
        if byte_id == ord("\n"):
            break
        generated.append(byte_id)
    learner.reset_episode()
    return generated.decode("ascii", errors="replace")


def _evaluate(
    learner: SequenceCortexOpponentLearner,
    split: ModularSplit,
    masks: list[frozenset[str]] | None = None,
) -> dict[str, Any]:
    if masks is None:
        masks = [frozenset()] * len(split.residues)
    predictions = [
        _generate_answer(
            learner, int(a), int(b), int(n), mask_fields=mask_fields
        )
        for (a, b), n, mask_fields in zip(
            split.operands, split.moduli, masks, strict=True
        )
    ]
    expected = [str(int(value)) for value in split.residues]
    exact = np.asarray(
        [
            prediction == target
            for prediction, target in zip(predictions, expected, strict=True)
        ]
    )
    numeric = np.asarray(
        [int(value) if value.isdigit() else -1 for value in predictions]
    )
    valid = numeric >= 0
    return {
        "examples": len(expected),
        "exact_accuracy": float(exact.mean()),
        "numeric_output_fraction": float(valid.mean()),
        "sample_predictions": [
            {
                "expression": modular_expression(
                    int(split.operands[i, 0]),
                    int(split.operands[i, 1]),
                    int(split.moduli[i]),
                    mask_fields=masks[i],
                ).decode("ascii"),
                "predicted": predictions[i],
                "expected": expected[i],
            }
            for i in range(min(12, len(expected)))
        ],
    }


def _evaluate_by_modulus(
    learner: SequenceCortexOpponentLearner, split: ModularSplit
) -> dict[str, dict[str, Any]]:
    rows = {}
    for modulus in np.unique(split.moduli):
        mask = split.moduli == modulus
        subset = ModularSplit(
            split.inputs[mask],
            split.residues[mask],
            split.moduli[mask],
            split.operands[mask],
        )
        result = _evaluate(learner, subset)
        result["chance_accuracy"] = 1.0 / int(modulus)
        rows[str(int(modulus))] = result
    return rows


def _mask_ablation_panel(
    learner: SequenceCortexOpponentLearner,
    split: ModularSplit,
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    fields = ("a", "b", "n")
    random_masks = [
        frozenset((str(rng.choice(fields)),)) for _ in range(len(split.residues))
    ]
    return {
        f"mask_{field}": _evaluate(
            learner, split, [frozenset((field,))] * len(split.residues)
        )
        for field in fields
    } | {"mask_one_random_field": _evaluate(learner, split, random_masks)}


def _evaluate_process_transitions(
    learner: SequenceCortexOpponentLearner, split: ModularSplit
) -> dict[str, Any]:
    correct = 0
    total = 0
    by_modulus: dict[str, list[int]] = {}
    for (a_raw, b_raw), n_raw in zip(
        split.operands, split.moduli, strict=True
    ):
        a, b, n = int(a_raw), int(b_raw), int(n_raw)
        counts = by_modulus.setdefault(str(n), [0, 0])
        for completed in range(b):
            prefix = modular_trajectory_prefix(a, b, n, completed)
            learner.reset_episode()
            learner.observe_chunk_probabilities(
                np.frombuffer(prefix, dtype=np.uint8), learn=False
            )
            predicted = int(np.argmax(learner.predict_next_distribution()))
            expected = ord(str((a % n + completed + 1) % n))
            hit = int(predicted == expected)
            correct += hit
            total += 1
            counts[0] += hit
            counts[1] += 1
    learner.reset_episode()
    return {
        "transitions": total,
        "next_position_accuracy": correct / max(total, 1),
        "by_modulus": {
            modulus: {
                "transitions": counts[1],
                "next_position_accuracy": counts[0] / max(counts[1], 1),
            }
            for modulus, counts in by_modulus.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    ladder = build_modular_ladder(
        seen_moduli=tuple(args.seen_moduli),
        unseen_modulus=args.unseen_modulus,
        operand_limit=args.operand_limit,
        seed=1_000_003 * (args.seed + 1),
    )
    learner = SequenceCortexOpponentLearner(
        max_context_bytes=args.max_context_bytes,
        max_sequence_nursery_paths=500_000,
        packed_paths=True,
    )
    rng = np.random.default_rng(args.seed + 701)
    epochs = []
    stable = 0
    previous_paths = -1
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(ladder.train.residues))
        observed, masked, masked_fields = _develop_epoch(
            learner,
            ladder.train,
            order,
            rng,
            args.mask_probability,
            args.curriculum,
        )
        if learner.embodied_bytes > args.max_body_bytes:
            sleep = learner.consolidate_to_byte_budget(args.max_body_bytes)
        else:
            sleep = None
        current_paths = learner.live_sequence_count
        stable = stable + 1 if current_paths == previous_paths else 0
        previous_paths = current_paths
        epochs.append(
            {
                "epoch": epoch,
                "observed_bytes": observed,
                "masked_expressions": masked,
                "masked_fields": masked_fields,
                "body_bytes": learner.embodied_bytes,
                "mature_sequence_paths": current_paths,
                "nursery_sequence_paths": learner.sequence_nursery_count,
                "sleep": sleep,
            }
        )
        if stable >= args.stable_epochs:
            break
    elapsed = time.perf_counter() - started
    evaluations = {
        "training": _evaluate(learner, ladder.train),
        "training_by_modulus": _evaluate_by_modulus(learner, ladder.train),
        "candidate": _evaluate(learner, ladder.candidate),
        "candidate_by_modulus": _evaluate_by_modulus(
            learner, ladder.candidate
        ),
        "held_out_operand_combinations": _evaluate(
            learner, ladder.held_out_pairs
        ),
        "held_out_by_modulus": _evaluate_by_modulus(
            learner, ladder.held_out_pairs
        ),
        "held_out_mask_ablations": _mask_ablation_panel(
            learner, ladder.held_out_pairs, seed=args.seed + 1901
        ),
        "unseen_modulus": _evaluate(learner, ladder.unseen_modulus),
    }
    if args.curriculum == "unfolded":
        evaluations["held_out_process_transitions"] = (
            _evaluate_process_transitions(learner, ladder.held_out_pairs)
        )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("wb") as handle:
        pickle.dump({"learner": learner}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "experiment": "alm_64mb_1c_conventional_modular_expression",
        "alm_name": "ALM-64MB-1C",
        "input_contract": "raw ASCII bytes formatted as 'a + b mod n = c\\n'",
        "prediction_boundary": "immediately after the equals sign and following space",
        "development_masking": (
            "each selected presentation masks exactly one uniformly sampled field; "
            "evaluation is always clean"
        ),
        "curriculum": args.curriculum,
        "curriculum_contract": (
            "unfolded lessons expose ring position at every unit addition, then "
            "present the compressed equation; evaluation presents only the equation"
        ),
        "epochs_completed": len(epochs),
        "stopped_for_anatomical_stability": len(epochs) < args.epochs,
        "body_bytes": learner.embodied_bytes,
        "body_ceiling_bytes": args.max_body_bytes,
        "development_seconds": elapsed,
        "epochs": epochs,
        "evaluations": evaluations,
        "unseen_modulus_chance": 1.0 / ladder.unseen_modulus_value,
        "checkpoint": str(args.checkpoint),
        "checkpoint_file_bytes": args.checkpoint.stat().st_size,
        "args": vars(args)
        | {"report": str(args.report), "checkpoint": str(args.checkpoint)},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
