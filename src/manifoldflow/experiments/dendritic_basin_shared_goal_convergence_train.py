"""Matched-budget convergence training for the shared-goal future organ.

Phase 1 of the convergence experiment. Four arms (the locked 2x2 over
persistent-ledger on/off x future-horizon K=1/K=4) train with identical
budgets: the same base checkpoint, one shared pass over the same native
base distributions per response (computed once -- they depend on no arm),
the same data order per pass, one receiver update and one organ update per
observed assistant turn. At geometric pass marks {0, 0.25, 0.5, 1, 2, 4, 8}
the trainer snapshots each arm's organ weights and receiver to disk; the
separate evaluator module scores those snapshots by independently checked
free generation.

Organ. Near head: y_next = W_near @ phi_t with
phi_t = [fast(24); core_t(64)], core_t the frozen chain's ledger-shortlist
mass-weighted core at turn t (zeroed for ledger-off arms). Seeded from a
closed-form ridge fit on each arm's own feature definition, then updated
per observed transition by a local delta rule toward the actual next-turn
code (stop-gradient target; no backpropagation). Far head (K>1 arms):
W_far trained toward the code two transitions ahead of the same
conversation, when that transition exists.

Causal ordering per turn: predict with current weights, update the
receiver with the predicted code, then update the organ from the observed
code. No evaluation happens here; no protected test payload is touched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import time
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
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    _load_native,
)

IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/dendritic_basin_shared_goal_convergence_train.py",
    "tests/test_dendritic_basin_shared_goal_convergence_train.py",
    *sweep_assay.IMPLEMENTATION_PATHS,
)

ARM_NAMES = ("A_k1_nooledger", "B_k1_ledger", "C_k4_nooledger", "D_k4_ledger")
LEDGER_ARMS = ("B_k1_ledger", "D_k4_ledger")
FAR_ARMS = ("C_k4_nooledger", "D_k4_ledger")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(IMPLEMENTATION_PATHS):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return result


class DeltaOrgan:
    """Near and far linear heads with local delta-rule updates."""

    def __init__(
        self,
        feature_dim: int,
        code_dim: int,
        *,
        near_rate: float,
        far_rate: float,
        far_enabled: bool,
        near_seed: np.ndarray | None,
        far_seed: np.ndarray | None,
    ):
        expected = (code_dim, feature_dim)
        if near_seed is None:
            self.near = np.zeros(expected)
        else:
            if near_seed.shape != expected:
                raise ValueError("near seed shape mismatch")
            self.near = near_seed.copy()
        if far_enabled:
            if far_seed is None:
                self.far = np.zeros(expected)
            else:
                if far_seed.shape != expected:
                    raise ValueError("far seed shape mismatch")
                self.far = far_seed.copy()
        else:
            self.far = None
        self.near_rate = near_rate
        self.far_rate = far_rate
        self.far_enabled = far_enabled
        self.near_updates = 0
        self.far_updates = 0

    def predict_near(self, phi: np.ndarray) -> np.ndarray:
        return self.near @ phi

    def predict_far(self, phi: np.ndarray) -> np.ndarray:
        if self.far is None:
            raise ValueError("far head is not enabled for this organ")
        return self.far @ phi

    def update_near(self, phi: np.ndarray, target: np.ndarray) -> float:
        error = target - self.predict_near(phi)
        self.near += self.near_rate * np.outer(error, phi)
        self.near_updates += 1
        return float(np.mean(np.square(error)))

    def update_far(self, phi: np.ndarray, target: np.ndarray) -> float:
        if self.far is None:
            raise ValueError("far head is not enabled for this organ")
        error = target - self.predict_far(phi)
        self.far += self.far_rate * np.outer(error, phi)
        self.far_updates += 1
        return float(np.mean(np.square(error)))


def ridge_seed(
    features: np.ndarray, targets: np.ndarray, *, alpha: float
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("ridge seed requires aligned feature and target matrices")
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += alpha
    return np.linalg.solve(gram, x.T @ y).T


def pass_marks(total_passes: float, updates_per_pass: int) -> list[int]:
    marks = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    schedule = sorted(
        {
            int(round(mark * updates_per_pass))
            for mark in marks
            if mark <= total_passes + 1e-9
        }
    )
    final = int(round(total_passes * updates_per_pass))
    if not schedule or schedule[-1] < final:
        schedule.append(final)
    return schedule


def snapshot_payload(
    organs: dict[str, DeltaOrgan],
    receivers: dict[str, Any],
    updates_done: int,
    updates_per_pass: int,
) -> dict[str, Any]:
    return {
        "updates_done": updates_done,
        "passes": updates_done / updates_per_pass,
        "organs": {
            arm: {
                "near": organ.near,
                "far": organ.far,
                "near_updates": organ.near_updates,
                "far_updates": organ.far_updates,
                "far_enabled": organ.far_enabled,
            }
            for arm, organ in organs.items()
        },
        "receivers": {arm: receivers[arm] for arm in ARM_NAMES},
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    _manifest, verified, conversations, _pairs = (
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
    _latent_bank, persistent, current, bank_audit = (
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
    # Examples are one per event transition. The organ predicts the next
    # ASSISTANT turn's code, so training and receiver updates run on the
    # assistant-target rows only: the rows whose target event role is
    # ASSISTANT, which are exactly the causal prediction points before each
    # assistant response.
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

    # Pair each assistant-target row with its assistant turn (the response
    # bytes the receiver trains on), in event order per conversation.
    assistant_turn_by_row: dict[int, Any] = {}
    assistant_turn_ordinal: dict[int, int] = {}
    rows_by_conversation: dict[int, list[int]] = {}
    for index in assistant_rows:
        rows_by_conversation.setdefault(examples[index].conversation_index, []).append(
            index
        )
    for conversation_index, rows in rows_by_conversation.items():
        rows.sort(key=lambda i: examples[i].transition_index)
        conversation = conversations[conversation_index]
        if len(rows) != len(conversation.assistant_turns):
            raise AssertionError(
                "assistant-target rows do not align with assistant turns"
            )
        for ordinal, (row_index, turn) in enumerate(
            zip(rows, conversation.assistant_turns, strict=True)
        ):
            assistant_turn_by_row[row_index] = turn
            assistant_turn_ordinal[row_index] = ordinal

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

    organs = {arm: build_organ(arm) for arm in ARM_NAMES}

    # -- shared native bases over training responses (computed once) --------
    cortex = _load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("a frozen native checkpoint is required")
    training_bases: dict[tuple[str, int], np.ndarray] = {}
    response_digest = hashlib.sha256()
    for conversation in conversations:
        cortex.reset_episode()
        cursor = 0
        for turn_index, turn in enumerate(conversation.assistant_turns):
            codec_assay._advance_chunk(
                cortex, conversation.payload[cursor : turn.content_start]
            )
            cursor = turn.content_start
            target = turn.content[: args.response_cap]
            bases = np.empty((len(target), 256), dtype=np.float64)
            for position, symbol in enumerate(target):
                bases[position] = codec_assay._normalized_native_distribution(
                    cortex.predict_next_distribution()
                )
                cortex.advance_observation(symbol, learn=False)
                cursor += 1
            training_bases[(conversation.episode_id, turn_index)] = bases
            response_digest.update(turn.episode_id.encode())
            response_digest.update(turn_index.to_bytes(4, "little"))
            response_digest.update(target)
        cortex.reset_episode()

    receivers = {
        arm: codec_assay.ContextCoupledByteWriter(
            "coupled",
            alphabet_size=256,
            table_bits=args.table_bits,
            learning_rate=args.learning_rate,
            local_orders=tuple(args.local_orders),
            max_persistent_features=codec.persistent_feature_count,
        )
        for arm in ARM_NAMES
    }

    def quantized(content: np.ndarray, source_label: str) -> tuple[int, ...]:
        return codec.quantize(
            codec.code_from_content(
                content, role=codec_assay.ASSISTANT_ROLE, source=source_label
            )
        )

    updates_per_pass = len(assistant_rows)
    marks = pass_marks(args.total_passes, updates_per_pass)
    marks_set = set(marks)
    snapshot_dir = args.snapshot_dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    updates_done = 0
    snapshots: list[dict[str, Any]] = []

    def take_snapshot() -> None:
        payload = snapshot_payload(organs, receivers, updates_done, updates_per_pass)
        path = snapshot_dir / f"mark_{updates_done:08d}.pkl"
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        snapshots.append({"updates_done": updates_done, "path": str(path)})
        print(f"  snapshot at updates={updates_done} -> {path.name}", flush=True)

    if 0 in marks_set:
        take_snapshot()

    total_updates = marks[-1]
    for pass_index in range(math.ceil(args.total_passes)):
        pass_rng = np.random.default_rng(args.training_seed + pass_index)
        conversation_order = pass_rng.permutation(len(conversations))
        for conversation_index in conversation_order:
            for row_index in rows_by_conversation.get(int(conversation_index), []):
                turn = assistant_turn_by_row[row_index]
                phi_full = np.concatenate([fast_states[row_index], cores[row_index]])
                phi_zero = np.concatenate(
                    [fast_states[row_index], np.zeros(code_dim)]
                )
                predictions: dict[str, np.ndarray] = {}
                for arm in ARM_NAMES:
                    phi = phi_full if arm in LEDGER_ARMS else phi_zero
                    predictions[arm] = organs[arm].predict_near(phi)
                target = turn.content[: args.response_cap]
                bases = training_bases.get(
                    (
                        conversations[int(conversation_index)].episode_id,
                        assistant_turn_ordinal[row_index],
                    )
                )
                if bases is not None and len(target) >= 8:
                    for arm in ARM_NAMES:
                        state = quantized(predictions[arm], turn.source_label)
                        receivers[arm].fit_base_anchored_episode(
                            target, bases, state, prefix=b""
                        )
                for arm in ARM_NAMES:
                    phi = phi_full if arm in LEDGER_ARMS else phi_zero
                    organs[arm].update_near(phi, near_targets[row_index])
                    if organs[arm].far is not None and np.all(
                        np.isfinite(far_targets[row_index])
                    ):
                        organs[arm].update_far(phi, far_targets[row_index])
                updates_done += 1
                if updates_done in marks_set:
                    take_snapshot()
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
        "experiment": "shared_goal_convergence_train",
        "status": "complete_development",
        "claim_scope": {
            "development_only": True,
            "no_evaluation_in_this_phase": True,
            "protected_test_payload_accessed": False,
            "backpropagation_used": False,
            "native_checkpoint_frozen": True,
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
            "shortlist_mass": args.shortlist_mass,
            "shortlist_maximum": args.shortlist_maximum,
            "table_bits": args.table_bits,
            "learning_rate": args.learning_rate,
            "local_orders": list(args.local_orders),
            "response_cap": args.response_cap,
            "training_seed": args.training_seed,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "training_transitions": len(examples),
            "updates_per_pass": updates_per_pass,
            "updates_done": updates_done,
            "snapshots": snapshots,
        },
        "prototype_bank": bank_audit,
        "surprise_population_training": overlay,
        "ledger_training": ledger_training,
        "response_sequence_sha256": response_digest.hexdigest(),
        "wall_seconds": elapsed,
    }
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = sweep_assay._parser()
    parser.description = __doc__
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--table-bits", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--local-orders", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--response-cap", type=int, default=128)
    parser.add_argument("--total-passes", type=float, default=8.0)
    parser.add_argument("--near-rate", type=float, default=0.02)
    parser.add_argument("--far-rate", type=float, default=0.02)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--training-seed", type=int, default=260_930)
    # --wall-limit-seconds is inherited from the chain; only the default is
    # re-set here.
    parser.set_defaults(wall_limit_seconds=14_400.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_training(args)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
