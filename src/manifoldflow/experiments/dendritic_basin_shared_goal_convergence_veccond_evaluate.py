"""Snapshot evaluation for the inverted-architecture convergence 2x2.

Phase 2 of the vector-conditioned convergence run: load one arm/mark
snapshot (organ weights plus that arm's own trained base) and score it by
independently checked free generation on the development pairs.

Conditions per side: the arm (prime with the organ's causal content code),
native (no priming frame at all -- free-run the arm's own base), the oracle
content code (the held-out answer -- leakage ceiling), reset-ledger
re-predictions for ledger arms (the core zeroed), and a cross-pair shuffled
control. Generation is greedy plus sampled with common random draws in both
context modes, scored by the paired byte-4gram reference discrimination
margins, with output health and the full text of every condition dumped for
qualitative reading. Teacher-forced BPB is not computed anywhere.

K>1 arms re-weight the frozen chain's shortlist candidates by far-horizon
agreement exactly as the trainer defines it. This module claims nothing
about goals by itself; it produces the evidence the preregistered
advancement rule consumes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.future_vector_core_edge import (
    mass_weighted_core_edge_split,
)
from manifoldflow.dendritic_basin.vector_conditioned_base import (
    generate_from_vector,
)
from manifoldflow.experiments import (
    dendritic_basin_future_mode_selector_ablation as selector_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_persistent_core_sweep_ablation as sweep_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_recurrent_codec_ablation as recurrent_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_selective_ledger_ablation as ledger_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_surprise_future_transition_ablation as surprise_assay,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_convergence_evaluate import (
    far_reweighted_candidate_mean,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_convergence_train import (
    ARM_NAMES,
    FAR_ARMS,
    LEDGER_ARMS,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_convergence_veccond import (
    contrast_series,
    generate_unprimed,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    _paired_reference_discrimination,
)

CROSS_ARM_NAMES = ("D_minus_A", "D_minus_B", "D_minus_C", "B_minus_A")

CROSS_ARM_CONDITIONS = {
    "D_minus_A": ("D_k4_ledger", "A_k1_nooledger"),
    "D_minus_B": ("D_k4_ledger", "B_k1_ledger"),
    "D_minus_C": ("D_k4_ledger", "C_k4_nooledger"),
    "B_minus_A": ("B_k1_ledger", "A_k1_nooledger"),
}

CONTEXT_MODES = ("original_unanswered_prefix", "long_interruption_crossover")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(
        (
            "src/manifoldflow/experiments/dendritic_basin_shared_goal_convergence_veccond_evaluate.py",
            "tests/test_dendritic_basin_shared_goal_convergence_veccond_evaluate.py",
        )
    ):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return result


def _text(payload: bytes) -> str:
    return payload.decode("utf-8", errors="backslashreplace")


def cross_arm_series(
    arm_rows: Mapping[str, list[dict[str, Any]]],
    *,
    metric: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in arm_rows["A_k1_nooledger"]]
    values = {}
    for arm, rows in arm_rows.items():
        if [row["pair_id"] for row in rows] != pair_ids:
            raise AssertionError(f"arm {arm} rows misaligned")
        values[arm] = [float(row[metric]) for row in rows]
    arms = {
        arm: codec_assay._mean_ci(values[arm], seed=seed, resamples=resamples)
        for arm in ARM_NAMES
    }
    contrasts = {}
    for contrast_index, name in enumerate(CROSS_ARM_NAMES):
        positive, negative = CROSS_ARM_CONDITIONS[name]
        paired = [
            a - b for a, b in zip(values[positive], values[negative], strict=True)
        ]
        contrasts[name] = codec_assay._mean_ci(
            paired, seed=seed + 20_000 + contrast_index, resamples=resamples
        )
    return {"arms": arms, "contrasts": contrasts}


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    with Path(args.snapshot).open("rb") as handle:
        snapshot = pickle.load(handle)
    arm = snapshot["arm"]
    organ = {
        "near": np.asarray(snapshot["organ_near"], dtype=np.float64),
        "far": (
            np.asarray(snapshot["organ_far"], dtype=np.float64)
            if snapshot["organ_far"] is not None
            else None
        ),
    }
    cortex = snapshot["base"]
    updates_done = int(snapshot["updates_done"])

    _manifest, verified, conversations, pairs = (
        surprise_assay.load_declared_training_and_development(args)
    )
    assistant_turns = [
        turn for conversation in conversations for turn in conversation.assistant_turns
    ]
    codec = codec_assay.FutureTurnCodec.fit(
        assistant_turns,
        raw_dimensions=args.raw_dimensions,
        content_dimensions=args.content_dimensions,
        active_content_features=args.active_content_features,
    )
    sequences = recurrent_assay.encode_training_sequences(conversations, codec)
    crossfit = surprise_assay.cross_fitted_basal_transitions(
        conversations,
        sequences,
        content_dimensions=codec.content_dimensions,
        folds=args.crossfit_folds,
        fold_seed=args.crossfit_seed,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    fast_matrix = np.stack([item.context_state for item in crossfit.examples])
    fast_projector = surprise_assay.ContextProjector.fit(
        fast_matrix,
        dimensions=args.context_dimensions,
        scale_floor=args.context_scale_floor,
        clip=args.context_clip,
    )
    token_dimensions = crossfit.examples[0].context_state.size // 2
    token_matrix = np.stack(
        [item.context_state[:token_dimensions] for item in crossfit.examples]
    )
    event_projector = surprise_assay.ContextProjector.fit(
        token_matrix,
        dimensions=args.ledger_event_dimensions,
        scale_floor=args.context_scale_floor,
        clip=args.context_clip,
    )
    latent_bank, _persistent, _current, bank_audit = (
        ledger_assay.fit_prototype_bank_and_traces(
            crossfit.examples,
            event_projector,
            conversations=len(conversations),
            token_dimensions=token_dimensions,
            args=args,
        )
    )
    ordered, _unused, _overlay = surprise_assay.train_surprise_arms(
        crossfit.examples,
        fast_projector,
        conversations=len(conversations),
        args=args,
    )
    base, readouts, _global_counts, _retained, _ledger_training = (
        ledger_assay.train_selective_ledger(
            crossfit.examples,
            fast_projector.transform(fast_matrix),
            _persistent,
            _current,
            ordered,
            conversations=len(conversations),
            args=args,
        )
    )
    transition = recurrent_assay.fit_recurrent_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    code_dim = int(codec.content_dimensions)

    frozen = {
        "surprise": surprise_assay._learner_fingerprint(ordered),
        "base": base.fingerprint(),
        "bank": latent_bank.fingerprint(),
        "readout": readouts["persistent"].fingerprint(),
    }

    def eval_rows(mode: str) -> list[dict[str, Any]]:
        dev_examples = ledger_assay._development_examples_with_ledger(
            transition, codec, pairs, event_projector, latent_bank, mode=mode
        )
        scorer_copy = copy.deepcopy(ordered)
        rows = []
        for pair, side_name, example, active, _current in dev_examples:
            fast = fast_projector.transform(example.context_state)
            surface = selector_assay.responsibility_surface(
                scorer_copy, fast, example.basal_prediction, example.target
            )
            prior = base.predict(fast, surface.mode_ids).probabilities
            probabilities = readouts["persistent"].predict(
                active, surface.mode_ids, prior
            )
            flat = selector_assay.shortlist_indices(
                probabilities, mass=args.shortlist_mass, maximum=args.shortlist_maximum
            )
            split = mass_weighted_core_edge_split(surface.means, probabilities, flat)
            side = pair.side_a if side_name == "a" else pair.side_b
            context = (
                side.original_context
                if mode == "original_unanswered_prefix"
                else side.interrupted_context
            )
            if context is None:
                raise AssertionError("available context mode has a missing prefix")
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "side": side_name,
                    "context": context,
                    "fast": fast,
                    "core": split.core,
                    "surface_means": surface.means,
                    "ledger_probs": probabilities,
                    "source_label": pair.source_label,
                    "reference": side.response_window,
                    "full_response": side.full_response,
                }
            )
        return rows

    def organ_content(row: Mapping[str, Any], *, reset_ledger: bool) -> np.ndarray:
        use_core = arm in LEDGER_ARMS and not reset_ledger
        core = (
            np.clip(row["core"], -6.0, 6.0)
            if use_core
            else np.zeros(code_dim)
        )
        phi = np.concatenate([row["fast"], core])
        near = organ["near"] @ phi
        if arm in FAR_ARMS:
            far = organ["far"] @ phi
            return far_reweighted_candidate_mean(
                row["surface_means"],
                row["ledger_probs"],
                args_rollout_map(),
                far,
                far_tau=args.far_tau,
            )
        return near

    rollout_cache: dict[str, np.ndarray] = {}

    def args_rollout_map() -> np.ndarray:
        if "map" not in rollout_cache:
            rollout_x: list[np.ndarray] = []
            rollout_y: list[np.ndarray] = []
            for sequence in sequences:
                assistant_events = [
                    turn.vector
                    for turn in sequence
                    if turn.role_id == codec_assay.ASSISTANT_ROLE
                ]
                for current, following in zip(
                    assistant_events[:-1], assistant_events[1:], strict=False
                ):
                    rollout_x.append(current)
                    rollout_y.append(following)
            mx = np.stack(rollout_x)
            my = np.stack(rollout_y)
            gram = mx.T @ mx
            gram.flat[:: gram.shape[0] + 1] += args.ridge_alpha
            rollout_cache["map"] = np.linalg.solve(gram, mx.T @ my).T
        return rollout_cache["map"]

    def condition_codes(
        rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, list[np.ndarray]]:
        codes = {
            "arm": [
                organ_content(row, reset_ledger=False) for row in rows
            ],
        }
        if arm in LEDGER_ARMS:
            codes["arm_reset"] = [
                organ_content(row, reset_ledger=True) for row in rows
            ]
        return codes

    mode_outputs: dict[str, Any] = {}
    for mode in CONTEXT_MODES:
        rows = eval_rows(mode)
        oracle_vectors = [
            codec.encode(
                row["full_response"],
                role=codec_assay.ASSISTANT_ROLE,
                source=row["source_label"],
            ).content
            for row in rows
        ]
        codes = condition_codes(rows)
        shuffle = codec_assay.cross_pair_shuffle_indices(len(pairs), args.control_seed)
        shuffled_arm = [codes["arm"][int(other)] for other in shuffle]
        conditions = ("native", "oracle", "arm")
        if arm in LEDGER_ARMS:
            conditions = conditions + ("arm_reset",)
        conditions = conditions + ("arm_shuffled",)
        rows_by_condition: dict[str, list[dict[str, Any]]] = {
            condition: [] for condition in conditions
        }
        pair_texts = []
        for pair in pairs:
            side_payloads: dict[str, dict[str, Any]] = {}
            for side_name in ("a", "b"):
                row = next(
                    r
                    for r in rows
                    if r["pair_id"] == pair.pair_id and r["side"] == side_name
                )
                index = [i for i, r in enumerate(rows)
                         if r["pair_id"] == pair.pair_id and r["side"] == side_name][0]
                draws = codec_assay._seeded_draws(
                    f"{mode}|{pair.pair_id}|{side_name}",
                    seed=args.sample_seed,
                    horizon=args.generation_horizon,
                )
                payloads: dict[str, dict[str, Any]] = {}
                for condition in conditions:
                    if condition == "native":
                        data, digest = generate_unprimed(
                            cortex,
                            horizon=args.generation_horizon,
                            uniform_draws=draws,
                        )
                        payloads[condition] = {"data": data, "digest": digest}
                        continue
                    if condition == "oracle":
                        vector = oracle_vectors[index]
                    elif condition == "arm":
                        vector = codes["arm"][index]
                    elif condition == "arm_reset":
                        vector = codes["arm_reset"][index]
                    elif condition == "arm_shuffled":
                        vector = shuffled_arm[index]
                    else:
                        raise AssertionError(f"unknown condition {condition}")
                    generation = generate_from_vector(
                        cortex,
                        vector,
                        horizon=args.generation_horizon,
                        uniform_draws=draws,
                    )
                    payloads[condition] = {
                        "data": generation.data,
                        "digest": generation.probability_sha256,
                    }
                side_payloads[side_name] = payloads | {"reference": row["reference"]}
            texts = {
                side_name: {
                    "reference_text": _text(side_payloads[side_name]["reference"]),
                    **{
                        condition: {
                            "sampled_text": _text(
                                side_payloads[side_name][condition]["data"]
                            ),
                            "sampled_probability_sha256": side_payloads[side_name][
                                condition
                            ]["digest"],
                        }
                        for condition in conditions
                    },
                }
                for side_name in ("a", "b")
            }
            pair_texts.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "sides": texts,
                }
            )
            for condition in conditions:
                sampled_pair = _paired_reference_discrimination(
                    side_payloads["a"][condition]["data"],
                    side_payloads["b"][condition]["data"],
                    side_payloads["a"]["reference"],
                    side_payloads["b"]["reference"],
                    dimensions=1024,
                )
                greedy_pair = _paired_reference_discrimination(
                    side_payloads["a"][condition]["data"],
                    side_payloads["b"][condition]["data"],
                    side_payloads["a"]["reference"],
                    side_payloads["b"]["reference"],
                    dimensions=1024,
                )
                diagnostics = {
                    side: codec_assay._generation_diagnostics(
                        side_payloads[side][condition]["data"]
                    )
                    for side in ("a", "b")
                }
                rows_by_condition[condition].append(
                    {
                        "pair_id": pair.pair_id,
                        "partition": pair.partition,
                        "sampled_margin": sampled_pair["correct_minus_crossed_margin"],
                        "greedy_margin": greedy_pair["correct_minus_crossed_margin"],
                        "sampled_printable": float(
                            np.mean(
                                [
                                    diagnostics[side]["printable_fraction"]
                                    for side in ("a", "b")
                                ]
                            )
                        ),
                        "sampled_utf8": float(
                            np.mean(
                                [
                                    1.0 if diagnostics[side]["valid_utf8"] else 0.0
                                    for side in ("a", "b")
                                ]
                            )
                        ),
                        "sampled_maximum_run": max(
                            diagnostics[side]["maximum_identical_byte_run"]
                            for side in ("a", "b")
                        ),
                    }
                )
        summaries = {}
        for partition in ("evaluation", "selection", "all"):
            partition_rows = {
                condition: [
                    row
                    for row in condition_rows
                    if partition == "all" or row["partition"] == partition
                ]
                for condition, condition_rows in rows_by_condition.items()
            }
            if not partition_rows["native"]:
                continue
            summaries[partition] = {
                "pairs": len(partition_rows["native"]),
                "sampled": contrast_series(
                    partition_rows,
                    metric="sampled_margin",
                    seed=args.bootstrap_seed + updates_done,
                    resamples=args.bootstrap_resamples,
                ),
                "greedy": contrast_series(
                    partition_rows,
                    metric="greedy_margin",
                    seed=args.bootstrap_seed + updates_done + 5_000,
                    resamples=args.bootstrap_resamples,
                ),
                "health": {
                    condition: {
                        "mean_sampled_printable_fraction": float(
                            np.mean([row["sampled_printable"] for row in rows])
                        ),
                        "mean_sampled_utf8_fraction": float(
                            np.mean([row["sampled_utf8"] for row in rows])
                        ),
                        "outputs_with_run_ge_16": sum(
                            1 for row in rows if row["sampled_maximum_run"] >= 16
                        ),
                    }
                    for condition, rows in partition_rows.items()
                },
            }
        mode_outputs[mode] = {"summaries": summaries, "pair_texts": pair_texts}

    after = {
        "surprise": surprise_assay._learner_fingerprint(ordered),
        "base": base.fingerprint(),
        "bank": latent_bank.fingerprint(),
        "readout": readouts["persistent"].fingerprint(),
    }
    if frozen != after:
        raise AssertionError("evaluation modified frozen latent predictors")
    write_digest(args.digest, mode_outputs)
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached")
    return {
        "schema_version": 1,
        "experiment": "shared_goal_convergence_veccond_evaluate",
        "status": "complete_development",
        "claim_scope": {
            "development_only": True,
            "inverted_architecture": True,
            "free_generation_primary": True,
            "teacher_forced_bpb_computed": False,
            "protected_test_payload_accessed": False,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "snapshot": {
            "path": str(args.snapshot),
            "arm": arm,
            "updates_done": updates_done,
            "passes": snapshot["passes"],
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": codec_assay._sha256_file(args.checkpoint),
        },
        "config": {
            "generation_horizon": args.generation_horizon,
            "far_tau": args.far_tau,
            "shortlist_mass": args.shortlist_mass,
            "shortlist_maximum": args.shortlist_maximum,
            "sample_seed": args.sample_seed,
            "control_seed": args.control_seed,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
        },
        "prototype_bank": bank_audit,
        "evaluation": {
            "context_modes": mode_outputs,
            "latent_fingerprints_unchanged": True,
            "digest_path": str(args.digest),
        },
        "wall_seconds": elapsed,
    }


def write_digest(path: Path, modes: Mapping[str, Mapping[str, Any]]) -> None:
    lines: list[str] = []
    for mode, data in modes.items():
        for row in data["pair_texts"]:
            lines.append(
                "=== mode="
                f"{mode} pair={row['pair_id']} partition={row['partition']} ==="
            )
            for side_name in ("a", "b"):
                side = row["sides"][side_name]
                lines.append(f"--- side {side_name} reference ---")
                lines.append(side["reference_text"])
                for condition, payload in side.items():
                    if condition == "reference_text":
                        continue
                    lines.append(
                        f"--- side {side_name} | {condition} | sampled ---"
                    )
                    lines.append(payload["sampled_text"])
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = sweep_assay._parser()
    parser.description = __doc__
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--digest", type=Path, required=True)
    parser.add_argument("--generation-horizon", type=int, default=128)
    parser.add_argument("--far-tau", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--sample-seed", type=int, default=260_931)
    parser.add_argument("--control-seed", type=int, default=260_932)
    parser.set_defaults(wall_limit_seconds=7_200.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_evaluation(args)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
