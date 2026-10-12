"""Convergence 2x2 v2: a live dendritic predictor primes the base.

The vector-conditioned run showed the priming channel carries real signal
after interruption (arm A's long-interruption curve rising to +0.0408
significant at 8 passes), with a deliberately linear predictor. This v2
replaces the linear organ with the codebase's own surprise-conditioned
dendritic grower -- SurpriseGrownFutureTransition run LIVE during the
convergence passes: per observed turn it forecasts a mixture of candidate
futures, the base learns the response bytes under the primed mixture
vector, and then the organ resolves the observed code with learning
enabled, so prediction surprise drives dendritic growth (nursery births,
promotions, consolidation, recycling) across passes.

One growing organ is shared across the four arms: its own learning is
arm-independent (it predicts the same observed codes from the same
conversation context). The 2x2 is therefore a question about how the
organ's mixture becomes a primed vector:

- Ledger off (A/C): prime with the organ's own responsibility-weighted
  mixture.
- Ledger on (B/D): the organ's candidates pass through the live ledger
  readout first (residual adjustment on the organ's affinity prior, the
  frozen chain's pipeline but learning per turn), then prime.
- K1 (A/B): prime with the responsibility-weighted mixture mean.
- K4 (C/D): prime with the far-agreement reweighted mean -- candidates
  rolled through the closed-form assistant-to-next-assistant content map
  versus a linear far head (delta rule, bounded), the second opinion that
  survived v1.

Interface unchanged from v1: canonical float32 frames, the base learns the
response bytes under the primed vector (learn=True, surprise and plasticity
in the base), generation primes from any dense vector, no sparse
quantization or residual receiver anywhere. Same marks, same seeds, same
paired-margin checker, full text digest. Local learning only; no
backpropagation; no protected test payload.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dataclasses import replace as _dataclass_replace

from manifoldflow.dendritic_basin.future_vector_core_edge import (
    mass_weighted_core_edge_split,
)
from manifoldflow.dendritic_basin.surprise_grown_future_transition import (
    SurpriseGrownFutureTransition,
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

CROSS_ARM_CONDITIONS = {
    "D_minus_A": ("D_k4_ledger", "A_k1_nooledger"),
    "D_minus_B": ("D_k4_ledger", "B_k1_ledger"),
    "D_minus_C": ("D_k4_ledger", "C_k4_nooledger"),
    "B_minus_A": ("B_k1_ledger", "A_k1_nooledger"),
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(
        (
            "src/manifoldflow/experiments/dendritic_basin_shared_goal_convergence_veccond2.py",
            "tests/test_dendritic_basin_shared_goal_convergence_veccond2.py",
            "src/manifoldflow/dendritic_basin/vector_conditioned_base.py",
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


def far_reweighted(
    means: Sequence[np.ndarray],
    probabilities: np.ndarray,
    rollout_map: np.ndarray,
    far: np.ndarray,
    *,
    far_tau: float,
) -> np.ndarray:
    return far_reweighted_candidate_mean(
        means, probabilities, rollout_map, far, far_tau=far_tau
    )


def mixture_point(
    prediction: np.ndarray,
    candidates: Sequence[Any],
    *,
    use_basal: bool,
) -> np.ndarray:
    """Responsibility-weighted mean over the mixture's own masses."""

    rows = [np.asarray(prediction, dtype=np.float64)]
    masses = [1.0]
    for candidate in candidates:
        rows.append(np.asarray(candidate.prediction, dtype=np.float64))
        masses.append(float(candidate.weight))
    if not use_basal and len(rows) > 1:
        rows = rows[1:]
        masses = masses[1:]
    matrix = np.stack(rows)
    weights = np.asarray(masses, dtype=np.float64)
    total = weights.sum()
    if total <= 0.0 or not np.all(np.isfinite(weights)):
        return matrix.mean(axis=0)
    return (weights / total) @ matrix


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


ARM_NAMES = ("A_k1_nooledger", "B_k1_ledger", "C_k4_nooledger", "D_k4_ledger")
LEDGER_ARMS = ("B_k1_ledger", "D_k4_ledger")
FAR_ARMS: tuple[()] = ()  # retained for evaluator compatibility; far_w removed from training


def run_veccond2(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    _manifest, verified, conversations, pairs = (
        surprise_assay.load_declared_training_and_development(args)
    )

    # -- frozen upstream chain (fast features, ledger machinery, codec) -----
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
    ordered, _shuffled_overlay, overlay = surprise_assay.train_surprise_arms(
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

    # -- ledger cores per transition (the ledger arms' prime bias) ----------
    scorer = copy.deepcopy(ordered)
    examples = crossfit.examples
    n = len(examples)
    assistant_role = codec_assay.ASSISTANT_ROLE
    code_dim = int(codec.content_dimensions)
    cores = np.zeros((n, code_dim), dtype=np.float64)
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
    cores = np.clip(cores, -6.0, 6.0)

    # -- assistant rows and their turns -------------------------------------
    examples = crossfit.examples
    n = len(examples)
    assistant_role = codec_assay.ASSISTANT_ROLE
    code_dim = int(codec.content_dimensions)
    assistant_rows = [
        index
        for index in range(n)
        if examples[index].target_turn.role_id == assistant_role
    ]
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

    by_conversation_transition: dict[tuple[int, int], int] = {
        (examples[index].conversation_index, examples[index].transition_index): index
        for index in range(n)
    }
    near_targets = np.stack([item.target for item in examples])

    # -- one live dendritic predictor shared by every arm -------------------
    organ = copy.deepcopy(ordered)
    capacity_overrides: dict[str, int] = {}
    if getattr(args, "max_residual_modes", 0) > 0:
        capacity_overrides["max_residual_modes"] = args.max_residual_modes
    if getattr(args, "max_context_assemblies", 0) > 0:
        capacity_overrides["max_context_assemblies"] = args.max_context_assemblies
    if args.k_inputs_per_branch > 0 or capacity_overrides:
        organ = SurpriseGrownFutureTransition(
            _dataclass_replace(
                organ.config,
                k_inputs_per_branch=args.k_inputs_per_branch,
                **capacity_overrides,
            ),
            seed=args.training_seed,
        )


    def ledger_adjusted(
        arm: str,
        surface_mode_ids: Sequence[int],
        ledger_probs: np.ndarray,
        candidates: Sequence[Any],
    ) -> np.ndarray:
        """Responsibilities after the live ledger readout's residual, as a
        weight per candidate aligned to the candidates tuple (basal first)."""

        weights = np.zeros(1 + len(candidates), dtype=np.float64)
        weights[0] = 1.0
        if ledger_probs is None:
            return weights
        position = {int(mode_id): p + 1 for p, mode_id in enumerate(surface_mode_ids)}
        for candidate_index, candidate in enumerate(candidates):
            slot = position.get(int(candidate.mode_id))
            if slot is not None and slot < len(ledger_probs):
                weights[candidate_index + 1] = float(ledger_probs[slot])
        return weights

    snapshot_dir = args.snapshot_dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    updates_per_pass = len(assistant_rows)
    marks = sorted(
        {
            int(round(mark * updates_per_pass))
            for mark in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
            if mark <= args.total_passes + 1e-9
        }
    )
    _final_update = int(round(args.total_passes * updates_per_pass))
    if not marks or marks[-1] < _final_update:
        marks.append(_final_update)
    marks_set = set(marks)
    total_updates = marks[-1]

    def save_organ_snapshot(updates_done: int) -> None:
        payload = {
            "updates_done": updates_done,
            "passes": updates_done / updates_per_pass,
            "organ": organ,
        }
        path = snapshot_dir / f"organ_mark_{updates_done:08d}.pkl"
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [organ] snapshot updates={updates_done}", flush=True)

    # Per-arm bases: each arm primes the SAME live organ's mixture, but the
    # base plasticity under that prime is arm-specific, so each arm keeps
    # its own base, loaded fresh from the shared checkpoint.
    bases: dict[str, Any] = {
        arm: _load_native(args.checkpoint) for arm in ARM_NAMES
    }
    if any(base is None for base in bases.values()):
        raise ValueError("a frozen native checkpoint is required")
    organ_growth = {"birth_eligible": 0, "nursery_increases": 0}
    updates_done = 0

    # -- resume from a previously saved mark if requested --------------------
    resume_mark = getattr(args, "resume_from_mark", 0) or 0
    if resume_mark > 0:
        organ_pkl = snapshot_dir / f"organ_mark_{resume_mark:08d}.pkl"
        if not organ_pkl.is_file():
            raise FileNotFoundError(f"organ snapshot not found: {organ_pkl}")
        with organ_pkl.open("rb") as handle:
            organ_snap = pickle.load(handle)
        organ = organ_snap["organ"]
        for arm in ARM_NAMES:
            arm_pkl = snapshot_dir / f"{arm}_mark_{resume_mark:08d}.pkl"
            if not arm_pkl.is_file():
                raise FileNotFoundError(f"arm base snapshot not found: {arm_pkl}")
            with arm_pkl.open("rb") as handle:
                arm_snap = pickle.load(handle)
            bases[arm] = arm_snap["base"]
        updates_done = resume_mark
        print(f"[resume] loaded mark {resume_mark} "
              f"(pass {resume_mark / len(assistant_rows):.2f})", flush=True)

    nursery_before = organ.nursery_count

    def snapshot_all(updates_done: int) -> None:
        save_organ_snapshot(updates_done)
        for arm in ARM_NAMES:
            payload = {"arm": arm, "updates_done": updates_done, "base": bases[arm]}
            path = snapshot_dir / f"{arm}_mark_{updates_done:08d}.pkl"
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [{arm}] base snapshot updates={updates_done}", flush=True)

    # stage A: interface teaching, arm-independent, before arms diverge
    # Skipped on resume — arm bases already include stage A.
    if resume_mark == 0:
        for conversation in conversations:
            for turn in conversation.assistant_turns:
                target = turn.content[: args.response_cap]
                if not target:
                    continue
                oracle_content = codec.encode(
                    turn.content, role=codec_assay.ASSISTANT_ROLE,
                    source=turn.source_label,
                ).content
                for arm in ARM_NAMES:
                    fit_vector_conditioned_turn(bases[arm], oracle_content, target)
        if 0 in marks_set:
            snapshot_all(0)

    for pass_index in range(math.ceil(args.total_passes)):
        pass_rng = np.random.default_rng(args.training_seed + pass_index)
        conversation_order = pass_rng.permutation(len(conversations))
        for conversation_index in conversation_order:
            for row_index in rows_by_conversation.get(int(conversation_index), []):
                if updates_done < resume_mark:
                    updates_done += 1
                    if updates_done >= total_updates:
                        break
                    continue
                example = examples[row_index]
                turn = assistant_turn_by_row[row_index]
                fast = fast_states[row_index]
                basal = example.basal_prediction
                forecast = organ.forecast(fast, basal)
                if organ.nursery_count > nursery_before:
                    organ_growth["nursery_increases"] += 1
                    nursery_before = organ.nursery_count
                target = turn.content[: args.response_cap]
                if target:
                    # Organ mixture is the state prediction; each arm primes
                    # the decoder with that predicted state (+ledger bias).
                    state_pred = mixture_point(
                        forecast.prediction, forecast.candidates, use_basal=True
                    )
                    for arm in ARM_NAMES:
                        content = state_pred + cores[row_index] if arm in LEDGER_ARMS else state_pred
                        content = np.clip(content, -6.0, 6.0)
                        if not np.all(np.isfinite(content)):
                            raise ValueError(
                                f"non-finite prime at row {row_index} ({arm}, "
                                f"pass {pass_index})"
                            )
                        fit_vector_conditioned_turn(bases[arm], content, target)
                update = organ.resolve(near_targets[row_index])
                if update.standardized_surprise >= organ.config.surprise_birth_z:
                    organ_growth["birth_eligible"] += 1
                if organ.nursery_count > nursery_before:
                    organ_growth["nursery_increases"] += 1
                    nursery_before = organ.nursery_count
                updates_done += 1
                if updates_done in marks_set:
                    snapshot_all(updates_done)
                if updates_done >= total_updates:
                    break
            if updates_done >= total_updates:
                break
        if updates_done >= total_updates:
            break

    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached")
    result = {
        "schema_version": 1,
        "experiment": "shared_goal_convergence_veccond2_train",
        "status": "complete_development_training",
        "wall_seconds": elapsed,
        "k_inputs_per_branch": args.k_inputs_per_branch,
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
    parser.add_argument("--training-seed", type=int, default=260_930)
    parser.add_argument("--k-inputs-per-branch", type=int, default=0)
    parser.set_defaults(max_residual_modes=0, max_context_assemblies=0)
    parser.add_argument("--resume-from-mark", type=int, default=0,
                        help="Resume training from this update count (requires arm+organ snapshots)")
    parser.set_defaults(wall_limit_seconds=14_400.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_veccond2(args)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
