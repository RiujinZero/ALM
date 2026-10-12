"""Convergence 2x2 on the inverted architecture: dense vectors prime the base.

Third convergence run. The interface is the inverted one
(vector_conditioned_base): the organ's dense predictor vector is primed into
the byte cortex directly as a canonical float32 frame -- no sparse
quantization, no residual receiver -- and the observed response bytes are
learned by the base itself, so all surprise and base plasticity live under
the primed state.

Design. Every frame in this experiment is the 64-dimensional content code
(no role/source coordinates), so the base sees one consistent frame shape.
Each arm trains its own base copy from the same initial checkpoint in two
stages:

- Stage A (interface teaching, arm-independent): one pass priming with the
  true turn's own content code and learning its response bytes. Teaches the
  base what a vector frame means before the arms diverge. Snapshotted as
  mark 0.
- Stage B (convergence, the 2x2): passes over the training conversations
  where the organ predicts the next response's content code and the base
  learns the response under that predicted vector. The organ is a linear
  near head over [fast(24); core(64)] (core zeroed for ledger-off arms),
  seeded by closed-form ridge and updated per observed turn by a local
  delta rule toward the observed code. K>1 arms add a far head trained two
  assistant turns ahead, and select among the frozen chain's shortlist
  candidates by far-horizon agreement (a_i proportional to
  resp_i * exp(-||A c_i - y_far||^2 / (2 tau^2)), A the closed-form
  assistant-to-next-assistant content map, tau locked).

Arms: A (K1, ledger off), B (K1, ledger on), C (K4, ledger off), D (K4,
ledger on). Matched budgets: same data order per pass, one organ update and
one base-learning turn per assistant turn, identical evaluation draws.

Evaluation per arm and mark: prime with the arm's causal organ code and
generate free bytes on the development pairs in both context modes, scored
by the paired byte-4gram reference discrimination margins. Controls: native
(unprimed generation from that arm's own base), the oracle content code
(the held-out answer -- leakage ceiling), reset-ledger re-predictions for
ledger arms (core zeroed), and a cross-pair shuffled control. Full text of
every condition is dumped for qualitative reading. Teacher-forced BPB is
not computed anywhere. Local delta rules only; no backpropagation; no
protected test payload.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.future_vector_core_edge import (
    mass_weighted_core_edge_split,
)
from manifoldflow.dendritic_basin.vector_conditioned_base import (
    fit_vector_conditioned_turn,
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
    DeltaOrgan,
    pass_marks,
    ridge_seed,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    _load_native,
)

CONTEXT_MODES = ("original_unanswered_prefix", "long_interruption_crossover")

CONTRAST_NAMES = (
    "arm_minus_native",
    "arm_minus_shuffled",
    "arm_minus_oracle",
    "arm_reset_minus_arm",
    "oracle_minus_native",
)

CONTRAST_CONDITIONS = {
    "arm_minus_native": ("arm", "native"),
    "arm_minus_shuffled": ("arm", "arm_shuffled"),
    "arm_minus_oracle": ("arm", "oracle"),
    "arm_reset_minus_arm": ("arm_reset", "arm"),
    "oracle_minus_native": ("oracle", "native"),
}

CROSS_ARM_NAMES = ("D_minus_A", "D_minus_B", "D_minus_C", "B_minus_A")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(
        (
            "src/manifoldflow/experiments/dendritic_basin_shared_goal_convergence_veccond.py",
            "tests/test_dendritic_basin_shared_goal_convergence_veccond.py",
            "src/manifoldflow/dendritic_basin/vector_conditioned_base.py",
            "tests/test_dendritic_basin_vector_conditioned_base.py",
        )
    ):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    for relative, record in sweep_assay.implementation_hashes().items():
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
        if positive not in values or negative not in values:
            continue
        paired = [
            a - b for a, b in zip(values[positive], values[negative], strict=True)
        ]
        contrasts[name] = codec_assay._mean_ci(
            paired, seed=seed + 10_000 + contrast_index, resamples=resamples
        )
    return {"conditions": arms, "contrasts": contrasts}


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
        positive, negative = name.split("_minus_")
        paired = [
            a - b for a, b in zip(values[positive], values[negative], strict=True)
        ]
        contrasts[name] = codec_assay._mean_ci(
            paired, seed=seed + 20_000 + contrast_index, resamples=resamples
        )
    return {"arms": arms, "contrasts": contrasts}


def generate_unprimed(
    cortex: Any,
    *,
    horizon: int,
    uniform_draws: np.ndarray | None,
) -> tuple[bytes, str]:
    """Free-run the base with no priming frame at all."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    draws = None
    if uniform_draws is not None:
        draws = np.asarray(uniform_draws, dtype=np.float64)
    cortex.reset_episode()
    generated = bytearray()
    digest = hashlib.sha256()
    for position in range(horizon):
        law = cortex.predict_next_distribution()
        law = np.asarray(law, dtype=np.float64)
        law = law / law.sum()
        digest.update(law.tobytes())
        if draws is None:
            symbol = int(np.argmax(law))
        else:
            symbol = min(
                int(np.searchsorted(np.cumsum(law), draws[position], side="right")),
                255,
            )
        generated.append(symbol)
        cortex.advance_observation(symbol, learn=False)
    cortex.reset_episode()
    return bytes(generated), digest.hexdigest()


def run_veccond(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    _manifest, verified, conversations, pairs = (
        surprise_assay.load_declared_training_and_development(args)
    )

    # -- frozen latent chain (identical to the locked sweep setup) ----------
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
    fast_states = fast_projector.transform(fast_matrix)
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
    latent_bank, persistent, current, bank_audit = (
        ledger_assay.fit_prototype_bank_and_traces(
            crossfit.examples,
            event_projector,
            conversations=len(conversations),
            token_dimensions=token_dimensions,
            args=args,
        )
    )
    ordered, _unused, overlay = surprise_assay.train_surprise_arms(
        crossfit.examples,
        fast_projector,
        conversations=len(conversations),
        args=args,
    )
    base, readouts, _global_counts, _retained, ledger_training = (
        ledger_assay.train_selective_ledger(
            crossfit.examples,
            fast_states,
            persistent,
            current,
            ordered,
            conversations=len(conversations),
            args=args,
        )
    )

    # -- per-transition rows, computed once (identical for every arm) -------
    scorer = copy.deepcopy(ordered)
    examples = crossfit.examples
    n = len(examples)
    fast_dim = int(fast_states.shape[1])
    code_dim = int(codec.content_dimensions)
    assistant_role = codec_assay.ASSISTANT_ROLE
    assistant_rows = [
        index
        for index in range(n)
        if examples[index].target_turn.role_id == assistant_role
    ]
    by_conversation_transition: dict[tuple[int, int], int] = {
        (examples[index].conversation_index, examples[index].transition_index): index
        for index in range(n)
    }
    cores = np.zeros((n, code_dim), dtype=np.float64)
    near_targets = np.stack([item.target for item in examples])
    far_targets = np.full((n, code_dim), np.nan)
    for index in assistant_rows:
        example = examples[index]
        far_index = by_conversation_transition.get(
            (example.conversation_index, example.transition_index + 2)
        )
        if (
            far_index is not None
            and examples[far_index].target_turn.role_id == assistant_role
        ):
            far_targets[index] = examples[far_index].target
    for index in range(n):
        surface = selector_assay.responsibility_surface(
            scorer,
            fast_states[index],
            examples[index].basal_prediction,
            examples[index].target,
        )
        prior = base.predict(fast_states[index], surface.mode_ids).probabilities
        probabilities = readouts["persistent"].predict(
            persistent[index], surface.mode_ids, prior
        )
        flat = selector_assay.shortlist_indices(
            probabilities, mass=args.shortlist_mass, maximum=args.shortlist_maximum
        )
        split = mass_weighted_core_edge_split(surface.means, probabilities, flat)
        cores[index] = split.core

    # The organ's delta rule is only stable on bounded features. The fast
    # projector already clips to +/-6; apply the same bound to the core so a
    # large candidate mean cannot exponential-blow the linear map.
    cores = np.clip(cores, -6.0, 6.0)

    rows_by_conversation: dict[int, list[int]] = {}
    for index in assistant_rows:
        rows_by_conversation.setdefault(examples[index].conversation_index, []).append(
            index
        )
    for rows in rows_by_conversation.values():
        rows.sort(key=lambda i: examples[i].transition_index)
    assistant_turn_by_row: dict[int, Any] = {}
    for conversation_index, rows in rows_by_conversation.items():
        conversation = conversations[conversation_index]
        if len(rows) != len(conversation.assistant_turns):
            raise AssertionError(
                "assistant-target rows do not align with assistant turns"
            )
        for row_index, turn in zip(rows, conversation.assistant_turns, strict=True):
            assistant_turn_by_row[row_index] = turn

    feature_dim = fast_dim + code_dim

    def build_organ(arm: str) -> DeltaOrgan:
        features = np.hstack(
            [fast_states, cores if arm in LEDGER_ARMS else np.zeros_like(cores)]
        )
        finite_rows = np.all(np.isfinite(features), axis=1)
        near_mask = finite_rows & np.all(np.isfinite(near_targets), axis=1)
        near_seed = ridge_seed(
            features[near_mask], near_targets[near_mask], alpha=args.ridge_alpha
        )
        far_seed = None
        if arm in FAR_ARMS:
            far_mask = finite_rows & np.all(np.isfinite(far_targets), axis=1)
            far_seed = ridge_seed(
                features[far_mask], far_targets[far_mask], alpha=args.ridge_alpha
            )
        return DeltaOrgan(
            feature_dim,
            code_dim,
            near_rate=args.near_rate,
            far_rate=args.far_rate,
            far_enabled=arm in FAR_ARMS,
            near_seed=near_seed,
            far_seed=far_seed,
        )

    # -- content-space rollout map (assistant code -> next assistant code) --
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
    rollout_mx = np.stack(rollout_x)
    rollout_my = np.stack(rollout_y)
    rollout_gram = rollout_mx.T @ rollout_mx
    rollout_gram.flat[:: rollout_gram.shape[0] + 1] += args.ridge_alpha
    rollout_map = np.linalg.solve(rollout_gram, rollout_mx.T @ rollout_my).T

    def organ_content(
        organ: DeltaOrgan,
        arm: str,
        fast: np.ndarray,
        core: np.ndarray,
        surface: Any,
        ledger_probs: np.ndarray,
    ) -> np.ndarray:
        phi = np.concatenate([fast, core])
        near = organ.predict_near(phi)
        if arm not in FAR_ARMS:
            return near
        far = organ.predict_far(phi)
        return far_reweighted_candidate_mean(
            surface.means,
            ledger_probs,
            rollout_map,
            far,
            far_tau=args.far_tau,
        )

    updates_per_pass = len(assistant_rows)
    marks = pass_marks(args.total_passes, updates_per_pass)
    marks_set = set(marks)
    snapshot_dir = args.snapshot_dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # -- training: stage A teaches the interface; stage B converges per arm -
    training_arms: dict[str, Any] = {}
    for arm in ARM_NAMES:
        organ = build_organ(arm)
        cortex = _load_native(args.checkpoint)
        if cortex is None:
            raise ValueError("a frozen native checkpoint is required")
        stage_a_surprise = 0.0
        for conversation in conversations:
            for turn in conversation.assistant_turns:
                target = turn.content[: args.response_cap]
                if not target:
                    continue
                oracle_content = codec.encode(
                    turn.content, role=codec_assay.ASSISTANT_ROLE,
                    source=turn.source_label,
                ).content
                diagnostics = fit_vector_conditioned_turn(
                    cortex, oracle_content, target
                )
                stage_a_surprise += diagnostics.total_surprise_bits
        if 0 in set(marks):
            payload = {
                "arm": arm,
                "updates_done": 0,
                "passes": 0.0,
                "organ_near": organ.near,
                "organ_far": organ.far,
                "base": cortex,
            }
            path = snapshot_dir / f"{arm}_mark_{0:08d}.pkl"
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [{arm}] snapshot updates=0", flush=True)

        updates_done = 0
        total_updates = marks[-1]
        for pass_index in range(math.ceil(args.total_passes)):
            pass_rng = np.random.default_rng(args.training_seed + pass_index)
            conversation_order = pass_rng.permutation(len(conversations))
            for conversation_index in conversation_order:
                for row_index in rows_by_conversation.get(
                    int(conversation_index), []
                ):
                    turn = assistant_turn_by_row[row_index]
                    use_core = arm in LEDGER_ARMS
                    phi = np.concatenate(
                        [fast_states[row_index], cores[row_index]]
                        if use_core
                        else [fast_states[row_index], np.zeros(code_dim)]
                    )
                    surface = selector_assay.responsibility_surface(
                        scorer,
                        fast_states[row_index],
                        examples[row_index].basal_prediction,
                        examples[row_index].target,
                    )
                    prior = base.predict(
                        fast_states[row_index], surface.mode_ids
                    ).probabilities
                    ledger_probs = readouts["persistent"].predict(
                        persistent[row_index], surface.mode_ids, prior
                    )
                    content = organ_content(
                        organ, arm, fast_states[row_index], cores[row_index]
                        if use_core else np.zeros(code_dim),
                        surface, ledger_probs,
                    )
                    if not np.all(np.isfinite(content)):
                        raise ValueError(
                            f"non-finite organ content at row {row_index} "
                            f"({arm}, pass {pass_index})"
                        )
                    target = turn.content[: args.response_cap]
                    if target:
                        fit_vector_conditioned_turn(cortex, content, target)
                    organ.update_near(phi, near_targets[row_index])
                    if organ.far is not None and np.all(
                        np.isfinite(far_targets[row_index])
                    ):
                        organ.update_far(phi, far_targets[row_index])
                    organ.near.clip(-50.0, 50.0, out=organ.near)
                    if organ.far is not None:
                        organ.far.clip(-50.0, 50.0, out=organ.far)
                    updates_done += 1
                    if updates_done in marks_set:
                        payload = {
                            "arm": arm,
                            "updates_done": updates_done,
                            "passes": updates_done / updates_per_pass,
                            "organ_near": organ.near,
                            "organ_far": organ.far,
                            "base": cortex,
                        }
                        path = snapshot_dir / f"{arm}_mark_{updates_done:08d}.pkl"
                        with path.open("wb") as handle:
                            pickle.dump(
                                payload, handle, protocol=pickle.HIGHEST_PROTOCOL
                            )
                        print(
                            f"  [{arm}] snapshot updates={updates_done}", flush=True
                        )
                    if updates_done >= total_updates:
                        break
                if updates_done >= total_updates:
                    break
            if updates_done >= total_updates:
                break
        training_arms[arm] = {
            "snapshots": sorted(marks_set),
            "stage_a_surprise_bits": stage_a_surprise,
            "near_updates": organ.near_updates,
            "far_updates": organ.far_updates,
        }
        del cortex

    elapsed_training = time.perf_counter() - started
    elapsed = elapsed_training
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached")
    result = {
        "schema_version": 1,
        "experiment": "shared_goal_convergence_veccond_train",
        "status": "complete_development_training",
        "claim_scope": {
            "development_only": True,
            "inverted_architecture": True,
            "no_sparse_quantization": True,
            "no_residual_receiver": True,
            "surprise_and_plasticity_in_base": True,
            "protected_test_payload_accessed": False,
            "backpropagation_used": False,
            "evaluation_performed_by_separate_module": True,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": codec_assay._sha256_file(args.checkpoint),
        },
        "config": {
            "total_passes": args.total_passes,
            "near_rate": args.near_rate,
            "far_rate": args.far_rate,
            "ridge_alpha": args.ridge_alpha,
            "far_tau": args.far_tau,
            "response_cap": args.response_cap,
            "shortlist_mass": args.shortlist_mass,
            "shortlist_maximum": args.shortlist_maximum,
            "training_seed": args.training_seed,
            "frame_convention": "64-dim content codes, little-endian float32",
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "training_transitions": len(examples),
            "assistant_rows_per_pass": updates_per_pass,
            "snapshot_marks_passes": [
                mark / updates_per_pass for mark in sorted(marks_set)
            ],
        },
        "prototype_bank": bank_audit,
        "surprise_population_training": overlay,
        "ledger_training": ledger_training,
        "training_arms": training_arms,
        "wall_seconds": elapsed,
    }
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = sweep_assay._parser()
    parser.description = __doc__
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--response-cap", type=int, default=128)
    parser.add_argument("--generation-horizon", type=int, default=128)
    parser.add_argument("--total-passes", type=float, default=8.0)
    parser.add_argument("--near-rate", type=float, default=0.02)
    parser.add_argument("--far-rate", type=float, default=0.02)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--far-tau", type=float, default=1.0)
    parser.add_argument("--training-seed", type=int, default=260_930)
    parser.set_defaults(wall_limit_seconds=14_400.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_veccond(args)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
