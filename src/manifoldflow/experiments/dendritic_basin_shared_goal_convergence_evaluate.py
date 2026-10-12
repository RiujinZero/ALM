"""Geometric-checkpoint evaluation for the convergence-arm snapshots.

Phase 2 of the convergence experiment: load one trainer snapshot (organ
weights and receivers for the four 2x2 arms at a recorded update mark) and
score it by independently checked free generation on the development pairs.

Conditions per side: the four arms (A K1 no-ledger, B K1 ledger, C K4
no-ledger, D K4 ledger) with each arm's live receiver and its organ's
causal near-turn code; reset-ledger re-predictions for the ledger arms
(the core zeroed); a shuffled-code control for the full arm (cross-pair
Sattolo); native (no receiver); and the oracle code (the held-out answer
itself -- a leakage ceiling). Free generation is primary: sampled and
greedy branches in both context modes, scored by the paired byte-4gram
reference discrimination margins, with output health and the full text of
every condition dumped for qualitative reading. Teacher-forced BPB is not
computed anywhere.

K>1 arms re-weight the frozen chain's shortlist candidates by far-horizon
agreement exactly as the trainer defines it: a_i proportional to
resp_i * exp(-||R(c_i) - y_far||^2 / (2 tau^2)) with the frozen recurrent
transition R and the organ's far prediction. This module claims nothing
about goals or task completion by itself; it produces the evidence the
preregistered advancement rule consumes.
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
from manifoldflow.experiments.dendritic_basin_shared_goal_convergence_train import (
    ARM_NAMES,
    FAR_ARMS,
    LEDGER_ARMS,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_convergence_train import (
    implementation_hashes as trainer_hashes,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    _load_native,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    _paired_reference_discrimination,
)

EVAL_CONDITIONS = (
    "native",
    "oracle_code_coupled",
    "A_k1_nooledger",
    "B_k1_ledger",
    "C_k4_nooledger",
    "D_k4_ledger",
    "B_k1_ledger_reset",
    "D_k4_ledger_reset",
    "D_k4_ledger_shuffled",
)

CONTRAST_NAMES = (
    "D_minus_A",
    "D_minus_B",
    "D_minus_C",
    "B_minus_A",
    "D_reset_minus_D",
    "D_shuffled_minus_D",
    "oracle_minus_native",
)

CONTRAST_CONDITIONS = {
    "D_minus_A": ("D_k4_ledger", "A_k1_nooledger"),
    "D_minus_B": ("D_k4_ledger", "B_k1_ledger"),
    "D_minus_C": ("D_k4_ledger", "C_k4_nooledger"),
    "B_minus_A": ("B_k1_ledger", "A_k1_nooledger"),
    "D_reset_minus_D": ("D_k4_ledger_reset", "D_k4_ledger"),
    "D_shuffled_minus_D": ("D_k4_ledger_shuffled", "D_k4_ledger"),
    "oracle_minus_native": ("oracle_code_coupled", "native"),
}

CONTEXT_MODES = ("original_unanswered_prefix", "long_interruption_crossover")

# Locked evaluation bootstrap seed. The latent chain keeps the inherited
# surprise-chain bootstrap_seed; only the margin summaries use this one.
EVALUATION_BOOTSTRAP_SEED = 260_933


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(
        (
            "src/manifoldflow/experiments/dendritic_basin_shared_goal_convergence_evaluate.py",
            "tests/test_dendritic_basin_shared_goal_convergence_evaluate.py",
        )
    ):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    for relative, record in trainer_hashes().items():
        result.setdefault(relative, record)
    return result


def _text(payload: bytes) -> str:
    return payload.decode("utf-8", errors="backslashreplace")


def contrast_series(
    rows_by_condition: Mapping[str, list[dict[str, Any]]],
    *,
    metric: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in rows_by_condition["native"]]
    values = {}
    for condition, rows in rows_by_condition.items():
        if [row["pair_id"] for row in rows] != pair_ids:
            raise AssertionError(f"condition {condition} rows misaligned")
        values[condition] = [float(row[metric]) for row in rows]
    arms = {
        condition: codec_assay._mean_ci(
            values[condition], seed=seed, resamples=resamples
        )
        for condition in rows_by_condition
    }
    contrasts = {}
    for contrast_index, name in enumerate(CONTRAST_NAMES):
        positive, negative = CONTRAST_CONDITIONS[name]
        paired = [
            a - b
            for a, b in zip(values[positive], values[negative], strict=True)
        ]
        contrasts[name] = codec_assay._mean_ci(
            paired, seed=seed + 10_000 + contrast_index, resamples=resamples
        )
    return {"conditions": arms, "contrasts": contrasts}


def far_reweighted_candidate_mean(
    means: Sequence[np.ndarray],
    probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    far: np.ndarray,
    *,
    far_tau: float,
) -> np.ndarray:
    """Reweight non-basal futures, falling back to basal when none exist."""

    rows = tuple(np.asarray(mean, dtype=np.float64) for mean in means)
    responsibilities = np.asarray(probabilities, dtype=np.float64)
    transition = np.asarray(transition_matrix, dtype=np.float64)
    far_target = np.asarray(far, dtype=np.float64)
    if not rows:
        raise ValueError("candidate means must include the basal prediction")
    if responsibilities.shape != (len(rows),):
        raise ValueError("candidate means and responsibilities are misaligned")
    basal = rows[0]
    if basal.ndim != 1:
        raise ValueError("basal prediction must be a vector")
    if len(rows) == 1:
        return basal.copy()
    candidates = np.stack(rows[1:])
    candidate_responsibilities = responsibilities[1:]
    if transition.shape != (basal.size, basal.size):
        raise ValueError("transition matrix and candidate dimensions are misaligned")
    if far_target.shape != basal.shape or candidates.shape[1:] != basal.shape:
        raise ValueError("far target and candidate dimensions are misaligned")
    responsibility_mass = float(candidate_responsibilities.sum())
    if responsibility_mass <= 0.0:
        return basal.copy()
    weights = candidate_responsibilities * np.exp(
        -np.square(
            np.sqrt(
                np.mean(
                    np.square(candidates @ transition.T - far_target[None, :]),
                    axis=1,
                )
            )
        )
        / (2.0 * far_tau * far_tau)
    )
    total = float(weights.sum())
    if total <= 0.0 or not np.all(np.isfinite(weights)):
        return candidate_responsibilities / responsibility_mass @ candidates
    return (weights / total) @ candidates


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    with Path(args.snapshot).open("rb") as handle:
        snapshot = pickle.load(handle)
    organs = {
        arm: {
            "near": np.asarray(data["near"], dtype=np.float64),
            "far": (
                np.asarray(data["far"], dtype=np.float64)
                if data["far"] is not None
                else None
            ),
        }
        for arm, data in snapshot["organs"].items()
    }
    receivers = snapshot["receivers"]
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

    # Content-space rollout map for the K>1 far-agreement reweighting: an
    # assistant code mapped to the next assistant code of the same
    # conversation (closed-form ridge over the training sequences). This is
    # the same target definition the organ's far head trains on, so the two
    # routes to the far state are comparable.
    rollout_features: list[np.ndarray] = []
    rollout_targets: list[np.ndarray] = []
    for sequence in sequences:
        assistant_events = [
            turn.vector
            for turn in sequence
            if turn.role_id == codec_assay.ASSISTANT_ROLE
        ]
        for current, following in zip(
            assistant_events[:-1], assistant_events[1:], strict=False
        ):
            rollout_features.append(current)
            rollout_targets.append(following)
    rollout_x = np.stack(rollout_features)
    rollout_y = np.stack(rollout_targets)
    rollout_gram = rollout_x.T @ rollout_x
    rollout_gram.flat[:: rollout_gram.shape[0] + 1] += args.ridge_alpha
    rollout_map = np.linalg.solve(rollout_gram, rollout_x.T @ rollout_y).T
    transition_matrix = rollout_map
    code_dim = int(codec.content_dimensions)

    cortex = _load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("a frozen native checkpoint is required")

    frozen = {
        "basal": transition.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(ordered),
        "base": base.fingerprint(),
        "bank": latent_bank.fingerprint(),
        "readout": readouts["persistent"].fingerprint(),
    }

    def eval_rows(mode: str) -> list[dict[str, Any]]:
        examples = ledger_assay._development_examples_with_ledger(
            transition, codec, pairs, event_projector, latent_bank, mode=mode
        )
        scorer_copy = copy.deepcopy(ordered)
        rows = []
        for pair, side_name, example, active, _current in examples:
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
            # predict() and surface.means both carry a basal entry first:
            # responsibilities align to the means rows as-is, basal included.
            responsibilities = np.asarray(probabilities, dtype=np.float64)
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "side": side_name,
                    "context": context,
                    "fast": fast,
                    "core": split.core,
                    "candidates": surface.means,
                    "responsibilities": responsibilities,
                    "source_label": pair.source_label,
                    "reference": side.response_window,
                    "full_response": side.full_response,
                }
            )
        return rows

    def organ_code(
        arm: str, row: Mapping[str, Any], *, reset_ledger: bool
    ) -> np.ndarray:
        use_core = arm in LEDGER_ARMS and not reset_ledger
        core = row["core"] if use_core else np.zeros(code_dim)
        phi = np.concatenate([row["fast"], core])
        organ = organs[arm]
        near = organ["near"] @ phi
        if arm in FAR_ARMS:
            far = organ["far"] @ phi
            return reweighted(row, far)
        return near

    def reweighted(row: Mapping[str, Any], far: np.ndarray) -> np.ndarray:
        return far_reweighted_candidate_mean(
            row["candidates"],
            row["responsibilities"],
            transition_matrix,
            far,
            far_tau=args.far_tau,
        )

    def quantized(content: np.ndarray, source_label: str) -> tuple[int, ...]:
        return codec.quantize(
            codec.code_from_content(
                content, role=codec_assay.ASSISTANT_ROLE, source=source_label
            )
        )

    def condition_codes(
        rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, list[tuple[int, ...]]]:
        codes: dict[str, list[tuple[int, ...]]] = {}
        for arm in ARM_NAMES:
            codes[arm] = [
                quantized(organ_code(arm, row, reset_ledger=False), row["source_label"])
                for row in rows
            ]
        for arm in LEDGER_ARMS:
            codes[f"{arm}_reset"] = [
                quantized(organ_code(arm, row, reset_ledger=True), row["source_label"])
                for row in rows
            ]
        return codes

    def build_conditions(
        codes: Mapping[str, Sequence[tuple[int, ...]]],
        oracle_states: Sequence[tuple[int, ...]],
        shuffled_full: Sequence[tuple[int, ...]],
        flat_index: int,
    ) -> dict[str, codec_assay.RouteAssignment]:
        assignments = {
            "native": codec_assay.RouteAssignment(None, (), 0.0),
            "oracle_code_coupled": codec_assay.RouteAssignment(
                receivers["D_k4_ledger"], oracle_states[flat_index], args.gain
            ),
        }
        for arm in ARM_NAMES:
            assignments[arm] = codec_assay.RouteAssignment(
                receivers[arm], codes[arm][flat_index], args.gain
            )
        for arm in LEDGER_ARMS:
            assignments[f"{arm}_reset"] = codec_assay.RouteAssignment(
                receivers[arm], codes[f"{arm}_reset"][flat_index], args.gain
            )
        assignments["D_k4_ledger_shuffled"] = codec_assay.RouteAssignment(
            receivers["D_k4_ledger"], shuffled_full[flat_index], args.gain
        )
        if tuple(assignments) != EVAL_CONDITIONS:
            raise AssertionError("eval condition coverage changed")
        return assignments

    mode_outputs: dict[str, Any] = {}
    for mode in CONTEXT_MODES:
        rows = eval_rows(mode)
        flat_index_by_key = {
            (row["pair_id"], row["side"]): index for index, row in enumerate(rows)
        }
        oracle_states = [
            codec.quantize(
                codec.encode(
                    row["full_response"],
                    role=codec_assay.ASSISTANT_ROLE,
                    source=row["source_label"],
                )
            )
            for row in rows
        ]
        codes = condition_codes(rows)
        shuffle = codec_assay.cross_pair_shuffle_indices(len(pairs), args.control_seed)
        shuffled_full = [codes["D_k4_ledger"][int(other)] for other in shuffle]
        rows_by_condition: dict[str, list[dict[str, Any]]] = {
            condition: [] for condition in EVAL_CONDITIONS
        }
        pair_texts = []
        for pair in pairs:
            side_payloads: dict[str, dict[str, Any]] = {}
            for side_name in ("a", "b"):
                flat_index = flat_index_by_key[(pair.pair_id, side_name)]
                row = rows[flat_index]
                assignments = build_conditions(
                    codes, oracle_states, shuffled_full, flat_index
                )
                draws = codec_assay._seeded_draws(
                    f"{mode}|{pair.pair_id}|{side_name}",
                    seed=args.sample_seed,
                    horizon=args.generation_horizon,
                )
                payloads: dict[str, dict[str, Any]] = {}
                for condition, assignment in assignments.items():
                    payloads[condition] = {
                        "greedy": codec_assay.generate_branch(
                            cortex,
                            row["context"],
                            assignment,
                            horizon=args.generation_horizon,
                            residual_clip=args.residual_clip,
                            uniform_draws=None,
                        ),
                        "sampled": codec_assay.generate_branch(
                            cortex,
                            row["context"],
                            assignment,
                            horizon=args.generation_horizon,
                            residual_clip=args.residual_clip,
                            uniform_draws=draws,
                        ),
                    }
                side_payloads[side_name] = payloads | {"reference": row["reference"]}
            texts = {
                side_name: {
                    "reference_text": _text(side_payloads[side_name]["reference"]),
                    **{
                        condition: {
                            "sampled_text": _text(
                                side_payloads[side_name][condition]["sampled"].data
                            ),
                            "sampled_probability_sha256": side_payloads[side_name][
                                condition
                            ]["sampled"].probability_sha256,
                        }
                        for condition in EVAL_CONDITIONS
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
            for condition in EVAL_CONDITIONS:
                sampled_pair = _paired_reference_discrimination(
                    side_payloads["a"][condition]["sampled"].data,
                    side_payloads["b"][condition]["sampled"].data,
                    side_payloads["a"]["reference"],
                    side_payloads["b"]["reference"],
                    dimensions=1024,
                )
                greedy_pair = _paired_reference_discrimination(
                    side_payloads["a"][condition]["greedy"].data,
                    side_payloads["b"][condition]["greedy"].data,
                    side_payloads["a"]["reference"],
                    side_payloads["b"]["reference"],
                    dimensions=1024,
                )
                diagnostics = {
                    side: codec_assay._generation_diagnostics(
                        side_payloads[side][condition]["sampled"].data
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
                    seed=EVALUATION_BOOTSTRAP_SEED + updates_done,
                    resamples=args.bootstrap_resamples,
                ),
                "greedy": contrast_series(
                    partition_rows,
                    metric="greedy_margin",
                    seed=EVALUATION_BOOTSTRAP_SEED + updates_done + 5_000,
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
        "basal": transition.fingerprint(include_state=False),
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
        "experiment": "shared_goal_convergence_evaluate",
        "status": "complete_development",
        "claim_scope": {
            "development_only": True,
            "free_generation_primary": True,
            "teacher_forced_bpb_computed": False,
            "protected_test_payload_accessed": False,
            "native_checkpoint_frozen": True,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "snapshot": {
            "path": str(args.snapshot),
            "updates_done": updates_done,
            "passes": snapshot["passes"],
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": codec_assay._sha256_file(args.checkpoint),
        },
        "config": {
            "generation_horizon": args.generation_horizon,
            "gain": args.gain,
            "residual_clip": args.residual_clip,
            "far_tau": args.far_tau,
            "shortlist_mass": args.shortlist_mass,
            "shortlist_maximum": args.shortlist_maximum,
            "sample_seed": args.sample_seed,
            "control_seed": args.control_seed,
            "bootstrap_seed": EVALUATION_BOOTSTRAP_SEED,
            "bootstrap_resamples": args.bootstrap_resamples,
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
                for condition in EVAL_CONDITIONS:
                    lines.append(f"--- side {side_name} | {condition} | sampled ---")
                    lines.append(side[condition]["sampled_text"])
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = sweep_assay._parser()
    parser.description = __doc__
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--digest", type=Path, required=True)
    parser.add_argument("--generation-horizon", type=int, default=128)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--residual-clip", type=float, default=2.0)
    parser.add_argument("--far-tau", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--sample-seed", type=int, default=260_931)
    parser.add_argument("--control-seed", type=int, default=260_932)
    # --bootstrap-seed/--bootstrap-resamples/--wall-limit-seconds are
    # inherited from the chain; the margin summaries use the locked module
    # constant EVALUATION_BOOTSTRAP_SEED.
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
