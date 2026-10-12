"""Development-only latent assay for surprise-grown future-vector transitions.

This experiment keeps the train-fit :class:`FutureTurnCodec` and the existing
one-step :class:`FutureTurnTrajectory` ridge transition as a fixed basal
predictor.  A bounded ``SurpriseGrownFutureTransition`` is trained on residuals
that become available only after each causal forecast.  Its training basal
forecasts are whole-conversation cross-fitted; development uses a basal model
fit on all training conversations and freezes every learned assembly.

The experiment is deliberately latent-only.  It loads schema-4 training and
development payloads, never loads a native ALM checkpoint, never generates
bytes, and makes no claim about goals or preferences.  PCA and ridge are batch
estimators.  Assembly updates are local NumPy rules, with no autograd,
backpropagation, or backpropagation through time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.future_turn_trajectory import (
    EncodedTurn,
    FutureTurnTrajectory,
)
from manifoldflow.dendritic_basin.surprise_grown_future_transition import (
    FutureModeCandidate,
    FutureTransitionForecast,
    SurpriseGrownFutureTransition,
    SurpriseGrownFutureTransitionConfig,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_recurrent_codec_ablation as recurrent_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_successor_shuffle_ablation as successor_assay,
)

TEXT_MODALITY_ID = recurrent_assay.TEXT_MODALITY_ID
KNOWN_SURPRISE_TUNING_PAIR_IDS = (
    "68903938701b8af176c324c4976b6c2997be77cf39915704e69f3b23eed0473b",
    "6ea20f05e744bf6ca9d85761f83deff238678f7af5877bf050f93557c91db255",
    "6aae93e9de1bf201cd5bd5a101b98c34f988d3a5c843750dc485b7f8196d268d",
    "df032b94e68af647a93a15ffa0a3bf19ec3cecb4ead492e746afa4a865e50391",
)
DEFAULT_EXCLUDED_PAIR_IDS = (
    *codec_assay.KNOWN_SMOKE_PAIR_IDS,
    *KNOWN_SURPRISE_TUNING_PAIR_IDS,
)
IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/"
    "dendritic_basin_surprise_future_transition_ablation.py",
    "tests/test_dendritic_basin_surprise_future_transition_ablation.py",
    "src/manifoldflow/dendritic_basin/surprise_grown_future_transition.py",
    "tests/test_dendritic_basin_surprise_grown_future_transition.py",
    "src/manifoldflow/dendritic_basin/future_turn_trajectory.py",
    "src/manifoldflow/experiments/dendritic_basin_future_turn_codec_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_future_turn_recurrent_codec_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_future_turn_successor_shuffle_ablation.py",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    """Hash this runner, focused tests, and every local mechanism it imports."""

    root = Path(__file__).resolve().parents[3]
    records: dict[str, dict[str, Any]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        records[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return records


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ContextProjector:
    """Training-only PCA whitening of an arbitrary recurrent numeric state."""

    mean: np.ndarray
    basis: np.ndarray
    scales: np.ndarray
    clip: float

    @classmethod
    def fit(
        cls,
        rows: np.ndarray,
        *,
        dimensions: int,
        scale_floor: float,
        clip: float,
    ) -> ContextProjector:
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) < 2:
            raise ValueError("context projector needs at least two matrix rows")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("context projector inputs must be finite")
        if not 1 <= dimensions <= matrix.shape[1]:
            raise ValueError("context dimensions are outside recurrent state width")
        if not math.isfinite(scale_floor) or scale_floor <= 0.0:
            raise ValueError("context scale floor must be finite and positive")
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("context clip must be finite and positive")

        mean = np.mean(matrix, axis=0)
        centered = matrix - mean
        covariance = centered.T @ centered / max(len(matrix) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues, kind="stable")[::-1][:dimensions]
        basis = np.asarray(eigenvectors[:, order], dtype=np.float64)
        for column in range(basis.shape[1]):
            pivot = int(np.argmax(np.abs(basis[:, column])))
            if basis[pivot, column] < 0.0:
                basis[:, column] *= -1.0
        scales = np.sqrt(np.maximum(eigenvalues[order], scale_floor**2))
        for value in (mean, basis, scales):
            value.setflags(write=False)
        return cls(mean, basis, scales, float(clip))

    @property
    def input_dimensions(self) -> int:
        return int(self.mean.size)

    @property
    def output_dimensions(self) -> int:
        return int(self.scales.size)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=np.float64)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.input_dimensions:
            raise ValueError("context rows have the wrong recurrent state width")
        if not np.all(np.isfinite(values)):
            raise ValueError("context rows must be finite")
        projected = ((values - self.mean) @ self.basis) / self.scales
        projected = np.clip(projected, -self.clip, self.clip)
        return projected[0] if single else projected

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"future-context-projector-v1\0")
        for array in (self.mean, self.basis, self.scales):
            digest.update(bytes.fromhex(_array_sha256(array)))
        digest.update(np.asarray([self.clip], dtype="<f8").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class TransitionExample:
    """One prefix-causal recurrent state, basal forecast, and later target."""

    conversation_index: int
    transition_index: int
    episode_id: str
    context_state: np.ndarray
    basal_prediction: np.ndarray
    target: np.ndarray
    target_turn: EncodedTurn

    def __post_init__(self) -> None:
        for name in ("context_state", "basal_prediction", "target"):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CrossFittedTransitions:
    examples: tuple[TransitionExample, ...]
    fold_assignment: np.ndarray
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        assignment = np.asarray(self.fold_assignment, dtype=np.int64).copy()
        assignment.setflags(write=False)
        object.__setattr__(self, "fold_assignment", assignment)


def conversation_fold_assignment(
    conversations: Sequence[codec_assay.TrainingConversation],
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Balanced deterministic folds whose unit is a whole conversation."""

    if folds < 2 or folds > len(conversations):
        raise ValueError("cross-fit folds are outside the conversation count")
    order = np.random.default_rng(seed).permutation(len(conversations))
    assignment = np.empty(len(conversations), dtype=np.int64)
    assignment[order] = np.arange(len(conversations), dtype=np.int64) % folds
    assignment.setflags(write=False)
    return assignment


def causal_transition_examples(
    model: FutureTurnTrajectory,
    sequence: Sequence[EncodedTurn],
    *,
    conversation_index: int,
    episode_id: str,
) -> tuple[TransitionExample, ...]:
    """Forecast every successor before revealing it to the recurrent state."""

    turns = tuple(sequence)
    if len(turns) < 2:
        raise ValueError("a causal transition sequence needs two observed events")
    model.reset()
    rows: list[TransitionExample] = []
    try:
        model.observe(turns[0])
        for transition_index, target_turn in enumerate(turns[1:]):
            snapshot = model.snapshot()
            tape = model.predicted_tape()
            if len(tape.slots) != 1:
                raise AssertionError("basal transition exposed more than one slot")
            rows.append(
                TransitionExample(
                    conversation_index=conversation_index,
                    transition_index=transition_index,
                    episode_id=episode_id,
                    context_state=snapshot.state,
                    basal_prediction=tape.slots[0].vector,
                    target=target_turn.vector,
                    target_turn=target_turn,
                )
            )
            model.observe(target_turn)
    finally:
        model.reset()
    return tuple(rows)


def cross_fitted_basal_transitions(
    conversations: Sequence[codec_assay.TrainingConversation],
    sequences: Sequence[Sequence[EncodedTurn]],
    *,
    content_dimensions: int,
    folds: int,
    fold_seed: int,
    memory_decay: float,
    ridge: float,
) -> CrossFittedTransitions:
    """Create training forecasts without fitting a basal map on their targets."""

    if len(conversations) != len(sequences):
        raise ValueError("conversations and recurrent sequences do not align")
    episode_ids = [conversation.episode_id for conversation in conversations]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("training episode IDs must be globally unique")
    assignment = conversation_fold_assignment(
        conversations, folds=folds, seed=fold_seed
    )
    by_conversation: list[tuple[TransitionExample, ...] | None] = [
        None
    ] * len(conversations)
    folds_detail: list[dict[str, Any]] = []
    for fold in range(folds):
        fit_sequences = tuple(
            sequence
            for index, sequence in enumerate(sequences)
            if int(assignment[index]) != fold
        )
        held_out = [
            index
            for index in range(len(conversations))
            if int(assignment[index]) == fold
        ]
        model = recurrent_assay.fit_recurrent_transition(
            fit_sequences,
            content_dimensions=content_dimensions,
            memory_decay=memory_decay,
            ridge=ridge,
        )
        for index in held_out:
            by_conversation[index] = causal_transition_examples(
                model,
                sequences[index],
                conversation_index=index,
                episode_id=conversations[index].episode_id,
            )
        folds_detail.append(
            {
                "fold": fold,
                "fit_conversations": len(fit_sequences),
                "held_out_conversations": len(held_out),
                "fit_episode_ids_sha256": _sha256(
                    "\n".join(
                        sorted(
                            conversations[index].episode_id
                            for index in range(len(conversations))
                            if int(assignment[index]) != fold
                        )
                    ).encode("utf-8")
                ),
                "held_out_episode_ids_sha256": _sha256(
                    "\n".join(
                        sorted(conversations[index].episode_id for index in held_out)
                    ).encode("utf-8")
                ),
                "fit_held_out_overlap": 0,
                "basal_fingerprint": model.fingerprint(include_state=False),
            }
        )
    if any(rows is None for rows in by_conversation):
        raise AssertionError("cross-fitting left a conversation unforecast")
    examples = tuple(
        row for rows in by_conversation if rows is not None for row in rows
    )
    contexts = np.stack([row.context_state for row in examples])
    basals = np.stack([row.basal_prediction for row in examples])
    targets = np.stack([row.target for row in examples])
    return CrossFittedTransitions(
        examples,
        assignment,
        {
            "unit": "whole_conversation",
            "scope": (
                "OOF ridge transition prediction conditional on one globally "
                "training-fit anchored codec"
            ),
            "codec_axes_refit_inside_each_fold": False,
            "training_episode_ids_globally_unique": True,
            "folds": folds,
            "fold_seed": fold_seed,
            "fold_counts": {
                str(fold): int(np.sum(assignment == fold)) for fold in range(folds)
            },
            "fold_assignment_sha256": _sha256(
                np.ascontiguousarray(assignment, dtype="<i8").tobytes()
            ),
            "folds_detail": folds_detail,
            "examples": len(examples),
            "context_state_sha256": _array_sha256(contexts),
            "oof_basal_sha256": _array_sha256(basals),
            "target_sha256": _array_sha256(targets),
            "oof_basal_mse": float(np.mean(np.square(targets - basals))),
            "target_role_counts": dict(
                sorted(
                    Counter(
                        str(row.target_turn.role_id) for row in examples
                    ).items()
                )
            ),
        },
    )


def _training_order(
    examples: Sequence[TransitionExample], *, conversations: int, seed: int
) -> np.ndarray:
    """Shuffle conversation blocks while preserving chronology within each block."""

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        grouped[row.conversation_index].append(index)
    if set(grouped) != set(range(conversations)):
        raise ValueError("every training conversation must contribute transitions")
    order: list[int] = []
    for conversation in np.random.default_rng(seed).permutation(conversations):
        rows = grouped[int(conversation)]
        if rows != sorted(rows, key=lambda value: examples[value].transition_index):
            raise AssertionError("cross-fitted transitions lost within-dialogue order")
        order.extend(rows)
    return np.asarray(order, dtype=np.int64)


def successor_target_permutation(
    examples: Sequence[TransitionExample], *, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derange successor content within every target role/source stratum."""

    target_turns = tuple(row.target_turn for row in examples)
    permutation, strata = successor_assay.stratified_successor_permutation(
        target_turns, seed=seed
    )
    equal = sum(
        np.array_equal(examples[index].target, examples[int(donor)].target)
        for index, donor in enumerate(permutation)
    )
    if equal:
        raise AssertionError("successor target derangement retained content")
    return permutation, {
        "kind": "role_source_stratified_successor_content_derangement",
        "seed": seed,
        "rows": len(examples),
        "strata": strata,
        "fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
        "equal_target_contents": equal,
        "permutation_sha256": _sha256(
            np.ascontiguousarray(permutation, dtype="<i8").tobytes()
        ),
        "target_content_multiset_preserved_within_strata": True,
    }


def make_surprise_transition(
    *, context_dimensions: int, target_dimensions: int, args: argparse.Namespace
) -> SurpriseGrownFutureTransition:
    """Build one arm from a common, fully reported lifecycle configuration."""

    config = SurpriseGrownFutureTransitionConfig(
        context_dim=context_dimensions,
        target_dim=target_dimensions,
        max_context_assemblies=args.max_context_assemblies,
        max_residual_modes=args.max_residual_modes,
        max_modes_per_context=args.max_modes_per_context,
        max_nursery_candidates=args.max_nursery_candidates,
        context_bandwidth=args.context_bandwidth,
        min_context_affinity=args.min_context_affinity,
        residual_floor=args.residual_floor,
        residual_scale_decay=args.residual_scale_decay,
        surprise_birth_z=args.surprise_birth_z,
        poor_representation_z=args.poor_representation_z,
        nursery_context_z=args.nursery_context_z,
        nursery_residual_z=args.nursery_residual_z,
        nursery_confirmations=args.nursery_confirmations,
        nursery_target_consistency=args.nursery_target_consistency,
        nursery_max_age=args.nursery_max_age,
        residual_learning_rate=args.residual_learning_rate,
        variance_learning_rate=args.variance_learning_rate,
        context_learning_rate=args.context_learning_rate,
        utility_decay=args.utility_decay,
        responsibility_fire_threshold=args.responsibility_fire_threshold,
        basal_mixture_weight=args.basal_mixture_weight,
        juvenile_forecast_weight=args.juvenile_forecast_weight,
        newborn_exploration_weight=args.newborn_exploration_weight,
        causal_support_prior=args.causal_support_prior,
        consolidation_min_age=args.consolidation_min_age,
        consolidation_min_fires=args.consolidation_min_fires,
        consolidation_gain_bits=args.consolidation_gain_bits,
        dormancy_staleness=args.dormancy_staleness,
        prune_gain_bits=args.prune_gain_bits,
        dormant_min_residence=args.dormant_min_residence,
        juvenile_min_recycle_age=args.juvenile_min_recycle_age,
        dormant_forecast_weight=args.dormant_forecast_weight,
    )
    return SurpriseGrownFutureTransition(config)


def _learner_fingerprint(learner: SurpriseGrownFutureTransition) -> str:
    """Hash complete Python state to prove that evaluation did not learn."""

    return _sha256(pickle.dumps(learner, protocol=5))


def _lifecycle_report(
    learner: SurpriseGrownFutureTransition,
    *,
    birth_eligible_events: int,
    nursery_increases: int,
) -> dict[str, Any]:
    states = learner.mode_states
    contexts = learner.context_states
    return {
        "population_scope": {
            "lifecycle_event_counters": "lifetime totals including recycled modes",
            "mode_and_context_state_summaries": (
                "current surviving population at the end of training"
            ),
        },
        "birth_eligible_events": birth_eligible_events,
        "nursery_candidate_increases": nursery_increases,
        "promotions": learner.births,
        "consolidations": learner.consolidations,
        "dormancies": learner.dormancies,
        "reactivations": learner.reactivations,
        "recycles": learner.recycles,
        "predictions_scored": learner.predictions_scored,
        "steps_learned": learner.step,
        "context_assemblies": learner.context_count,
        "residual_modes": len(states),
        "nursery_candidates_remaining": learner.nursery_count,
        "lifecycle_counts": dict(
            sorted(Counter(state.lifecycle for state in states).items())
        ),
        "mode_fires": {
            "total": int(sum(state.fires for state in states)),
            "minimum": int(min((state.fires for state in states), default=0)),
            "maximum": int(max((state.fires for state in states), default=0)),
        },
        "mode_causal_scores": {
            "total": int(sum(state.causal_scores for state in states)),
            "minimum": int(
                min((state.causal_scores for state in states), default=0)
            ),
            "maximum": int(
                max((state.causal_scores for state in states), default=0)
            ),
        },
        "context_causal_scores": {
            "total": int(sum(state.causal_scores for state in contexts)),
            "minimum": int(
                min((state.causal_scores for state in contexts), default=0)
            ),
            "maximum": int(
                max((state.causal_scores for state in contexts), default=0)
            ),
        },
        "predictive_gain_bits": {
            "emitted_mixture_per_coordinate": float(
                learner.mixture_predictive_gain_bits_per_coordinate
            ),
            "mean": float(
                np.mean([state.predictive_gain_bits for state in states])
                if states
                else 0.0
            ),
            "minimum": float(
                min((state.predictive_gain_bits for state in states), default=0.0)
            ),
            "maximum": float(
                max((state.predictive_gain_bits for state in states), default=0.0)
            ),
        },
        "context_predictive_gain_bits_per_coordinate": {
            "mean": float(
                np.mean([state.predictive_gain_bits for state in contexts])
                if contexts
                else 0.0
            ),
            "minimum": float(
                min((state.predictive_gain_bits for state in contexts), default=0.0)
            ),
            "maximum": float(
                max((state.predictive_gain_bits for state in contexts), default=0.0)
            ),
        },
        "frozen_fingerprint": _learner_fingerprint(learner),
    }


def train_surprise_arms(
    examples: Sequence[TransitionExample],
    projector: ContextProjector,
    *,
    conversations: int,
    args: argparse.Namespace,
) -> tuple[
    SurpriseGrownFutureTransition,
    SurpriseGrownFutureTransition,
    dict[str, Any],
]:
    """Train ordered and successor-shuffled overlays on causal OOF basals."""

    if not examples:
        raise ValueError("surprise training requires cross-fitted examples")
    order = _training_order(
        examples, conversations=conversations, seed=args.training_order_seed
    )
    permutation, shuffle_audit = successor_target_permutation(
        examples, seed=args.successor_shuffle_seed
    )
    contexts = projector.transform(
        np.stack([example.context_state for example in examples])
    )
    ordered = make_surprise_transition(
        context_dimensions=projector.output_dimensions,
        target_dimensions=examples[0].target.size,
        args=args,
    )
    shuffled = make_surprise_transition(
        context_dimensions=projector.output_dimensions,
        target_dimensions=examples[0].target.size,
        args=args,
    )
    counters = {
        "ordered": {
            "eligible": 0,
            "nursery_increases": 0,
            "basal_nll": [],
            "mixture_nll": [],
            "mixture_gain": [],
        },
        "successor_shuffled": {
            "eligible": 0,
            "nursery_increases": 0,
            "basal_nll": [],
            "mixture_nll": [],
            "mixture_gain": [],
        },
    }
    for _pass in range(args.overlay_training_passes):
        for index_value in order:
            index = int(index_value)
            example = examples[index]
            donor = examples[int(permutation[index])]
            for name, learner, target in (
                ("ordered", ordered, example.target),
                ("successor_shuffled", shuffled, donor.target),
            ):
                nursery_before = learner.nursery_count
                learner.forecast(contexts[index], example.basal_prediction)
                update = learner.resolve(target)
                if (
                    update.standardized_surprise >= learner.config.surprise_birth_z
                    and update.poor_representation
                ):
                    counters[name]["eligible"] += 1
                counters[name]["nursery_increases"] += (
                    learner.nursery_count > nursery_before
                )
                counters[name]["basal_nll"].append(
                    update.basal_nll_bits_per_coordinate
                )
                counters[name]["mixture_nll"].append(
                    update.mixture_nll_bits_per_coordinate
                )
                counters[name]["mixture_gain"].append(
                    update.mixture_gain_bits_per_coordinate
                )

    def prequential_score(name: str) -> dict[str, Any]:
        values = counters[name]
        return {
            "kind": "module_emitted_prequential_diagonal_gaussian_mixture",
            "basal_nll_bits_per_coordinate": float(np.mean(values["basal_nll"])),
            "mixture_nll_bits_per_coordinate": float(
                np.mean(values["mixture_nll"])
            ),
            "mixture_gain_bits_per_coordinate": float(
                np.mean(values["mixture_gain"])
            ),
            "lower_nll_is_better": True,
            "target_scored_before_update": True,
        }

    audit = {
        "training_order": "deterministically shuffled conversation blocks; "
        "chronology preserved within each conversation",
        "training_order_seed": args.training_order_seed,
        "training_order_sha256": _sha256(
            np.ascontiguousarray(order, dtype="<i8").tobytes()
        ),
        "training_passes": args.overlay_training_passes,
        "ordered": _lifecycle_report(
            ordered,
            birth_eligible_events=counters["ordered"]["eligible"],
            nursery_increases=counters["ordered"]["nursery_increases"],
        ),
        "successor_shuffled": _lifecycle_report(
            shuffled,
            birth_eligible_events=counters["successor_shuffled"]["eligible"],
            nursery_increases=counters["successor_shuffled"]["nursery_increases"],
        ),
        "prequential_proper_scores": {
            "ordered": prequential_score("ordered"),
            "successor_shuffled": prequential_score("successor_shuffled"),
            "cross_arm_difference_is_not_primary": (
                "the shuffled arm is scored against shuffled targets during training"
            ),
        },
        "successor_shuffle": shuffle_audit,
        "controls": {
            "same_contexts": True,
            "same_cross_fitted_basal_predictions": True,
            "same_configuration_and_capacity": True,
            "same_training_order": True,
            "realized_birth_count_forced_equal": False,
            "birth_count_matching_omitted_reason": (
                "forcing equal births would intervene on the surprise-gated "
                "structural rule being tested"
            ),
        },
    }
    return ordered, shuffled, audit


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-15 else float(np.dot(left, right) / denominator)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def gaussian_bits_per_coordinate(
    target: np.ndarray,
    means: Sequence[np.ndarray],
    weights: Sequence[float],
    variance: np.ndarray,
) -> float:
    """Proper fixed-covariance mixture log score in bits per coordinate."""

    observed = np.asarray(target, dtype=np.float64)
    diagonal = np.asarray(variance, dtype=np.float64)
    centers = tuple(np.asarray(mean, dtype=np.float64) for mean in means)
    probabilities = np.asarray(weights, dtype=np.float64)
    if not centers or probabilities.shape != (len(centers),):
        raise ValueError("mixture means and weights must be nonempty and aligned")
    if observed.ndim != 1 or diagonal.shape != observed.shape:
        raise ValueError("target and variance must be aligned vectors")
    if any(center.shape != observed.shape for center in centers):
        raise ValueError("every mixture mean must align with the target")
    if (
        not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(diagonal))
        or np.any(diagonal <= 0.0)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
    ):
        raise ValueError("mixture score inputs must be finite and positive")
    probabilities = probabilities / probabilities.sum()
    constant = float(np.sum(np.log(2.0 * math.pi * diagonal)))
    log_components = np.asarray(
        [
            math.log(float(weight))
            - 0.5
            * (
                constant
                + float(np.sum(np.square(observed - center) / diagonal))
            )
            for center, weight in zip(centers, probabilities, strict=True)
        ]
    )
    return -_logsumexp(log_components) / (observed.size * math.log(2.0))


def _mixture_surface(
    forecast: FutureTransitionForecast,
) -> tuple[tuple[np.ndarray, ...], tuple[float, ...]]:
    means = (forecast.basal_prediction,) + tuple(
        candidate.prediction for candidate in forecast.candidates
    )
    weights = np.asarray(
        [forecast.basal_weight]
        + [candidate.weight for candidate in forecast.candidates],
        dtype=np.float64,
    )
    weights = np.maximum(weights, 1e-300)
    weights /= weights.sum()
    return means, tuple(float(value) for value in weights)


def _consolidated_only_forecast(
    forecast: FutureTransitionForecast,
) -> FutureTransitionForecast:
    """Renormalize the already-emitted surface after retaining mature modes only."""

    retained = tuple(
        candidate
        for candidate in forecast.candidates
        if candidate.lifecycle == "consolidated"
    )
    raw_weights = np.asarray(
        [forecast.basal_weight] + [candidate.weight for candidate in retained],
        dtype=np.float64,
    )
    raw_weights /= raw_weights.sum()
    candidates = tuple(
        FutureModeCandidate(
            mode_id=candidate.mode_id,
            context_id=candidate.context_id,
            prediction=candidate.prediction,
            weight=float(raw_weights[index + 1]),
            lifecycle=candidate.lifecycle,
        )
        for index, candidate in enumerate(retained)
    )
    point = raw_weights[0] * forecast.basal_prediction
    for candidate in candidates:
        point = point + candidate.weight * candidate.prediction
    return FutureTransitionForecast(
        prediction=np.asarray(point, dtype=np.float64),
        basal_prediction=forecast.basal_prediction,
        basal_weight=float(raw_weights[0]),
        candidates=candidates,
    )


def _prediction_metrics(
    target: np.ndarray,
    point: np.ndarray,
    forecast: FutureTransitionForecast | None,
    *,
    proper_variance: np.ndarray,
) -> dict[str, Any]:
    if forecast is None:
        means, weights = (point,), (1.0,)
        candidate_count = 0
        best = point
        basal_weight = 1.0
    else:
        means, weights = _mixture_surface(forecast)
        candidate_count = len(forecast.candidates)
        basal_weight = float(forecast.basal_weight)
        retained = tuple(
            candidate.prediction for candidate in forecast.candidates
        ) or (forecast.basal_prediction,)
        best = min(
            retained, key=lambda mean: float(np.mean(np.square(target - mean)))
        )
    return {
        "mse": float(np.mean(np.square(target - point))),
        "cosine": _cosine(target, point),
        "proper_gaussian_mixture_bits_per_coordinate": (
            gaussian_bits_per_coordinate(
                target, means, weights, proper_variance
            )
        ),
        "candidate_count": candidate_count,
        "basal_mixture_weight": basal_weight,
        "effective_mixture_components": float(
            1.0 / np.sum(np.square(np.asarray(weights, dtype=np.float64)))
        ),
        "best_retained_mode_mse": float(np.mean(np.square(target - best))),
        "best_retained_mode_cosine": _cosine(target, best),
        "best_mode_selected_with_hindsight_nonproper": candidate_count > 0,
    }


def _development_examples(
    model: FutureTurnTrajectory,
    codec: codec_assay.FutureTurnCodec,
    pairs: Sequence[codec_assay.DevelopmentPair],
    *,
    mode: str,
) -> list[tuple[codec_assay.DevelopmentPair, str, TransitionExample]]:
    rows: list[tuple[codec_assay.DevelopmentPair, str, TransitionExample]] = []
    for pair_index, pair in enumerate(pairs):
        for side_index, (side_name, side) in enumerate(
            (("a", pair.side_a), ("b", pair.side_b))
        ):
            context = (
                side.original_context
                if mode == "original_unanswered_prefix"
                else side.interrupted_context
            )
            if context is None:
                raise ValueError("requested development context mode is unavailable")
            turns = recurrent_assay.observed_turns_from_transcript(
                context, generation_context=True
            )
            prefix = tuple(
                recurrent_assay._encoded_event(
                    codec, turn, source_label=pair.source_label
                )
                for turn in turns
            )
            target = recurrent_assay._encoded_event(
                codec,
                recurrent_assay.ObservedTurn(
                    codec_assay.ASSISTANT_ROLE, side.full_response
                ),
                source_label=pair.source_label,
            )
            model.reset()
            try:
                for event in prefix:
                    model.observe(event)
                snapshot = model.snapshot()
                basal = model.predicted_tape().slots[0].vector
            finally:
                model.reset()
            rows.append(
                (
                    pair,
                    side_name,
                    TransitionExample(
                        conversation_index=pair_index,
                        transition_index=side_index,
                        episode_id=pair.pair_id,
                        context_state=snapshot.state,
                        basal_prediction=basal,
                        target=target.vector,
                        target_turn=target,
                    ),
                )
            )
    return rows


def _mean_ci(values: Sequence[float], *, seed: int, resamples: int) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    mean = float(np.mean(data))
    if len(data) == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": 1}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(data), size=(resamples, len(data)))
    estimates = np.mean(data[draws], axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return {
        "mean": mean,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n": len(data),
    }


def _summarize_evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row["partition"] == partition]
    if not selected:
        return {"available": False, "pairs": 0, "sides": 0}
    pair_ids = sorted({str(row["pair_id"]) for row in selected})
    arms = (
        "basal",
        "ordered",
        "ordered_consolidated_only",
        "successor_shuffled",
        "successor_shuffled_consolidated_only",
    )
    metric_names = (
        "mse",
        "cosine",
        "proper_gaussian_mixture_bits_per_coordinate",
        "candidate_count",
        "basal_mixture_weight",
        "effective_mixture_components",
        "best_retained_mode_mse",
        "best_retained_mode_cosine",
    )

    def pair_values(arm: str, metric: str) -> list[float]:
        return [
            float(
                np.mean(
                    [
                        row["arms"][arm][metric]
                        for row in selected
                        if row["pair_id"] == pair_id
                    ]
                )
            )
            for pair_id in pair_ids
        ]

    arm_summaries: dict[str, Any] = {}
    for arm_index, arm in enumerate(arms):
        arm_summaries[arm] = {
            metric: _mean_ci(
                pair_values(arm, metric),
                seed=bootstrap_seed + arm_index * 1_009 + metric_index,
                resamples=bootstrap_resamples,
            )
            for metric_index, metric in enumerate(metric_names)
        }

    contrast_specs = {
        "ordered_minus_basal_mse": ("ordered", "basal", "mse"),
        "ordered_minus_basal_cosine": ("ordered", "basal", "cosine"),
        "ordered_minus_basal_proper_bits": (
            "ordered",
            "basal",
            "proper_gaussian_mixture_bits_per_coordinate",
        ),
        "ordered_minus_successor_shuffled_mse": (
            "ordered",
            "successor_shuffled",
            "mse",
        ),
        "ordered_minus_successor_shuffled_cosine": (
            "ordered",
            "successor_shuffled",
            "cosine",
        ),
        "ordered_minus_successor_shuffled_proper_bits": (
            "ordered",
            "successor_shuffled",
            "proper_gaussian_mixture_bits_per_coordinate",
        ),
        "ordered_consolidated_only_minus_basal_mse": (
            "ordered_consolidated_only",
            "basal",
            "mse",
        ),
        "ordered_consolidated_only_minus_basal_cosine": (
            "ordered_consolidated_only",
            "basal",
            "cosine",
        ),
        "ordered_consolidated_only_minus_basal_proper_bits": (
            "ordered_consolidated_only",
            "basal",
            "proper_gaussian_mixture_bits_per_coordinate",
        ),
        "ordered_consolidated_only_minus_shuffled_consolidated_only_mse": (
            "ordered_consolidated_only",
            "successor_shuffled_consolidated_only",
            "mse",
        ),
        "ordered_consolidated_only_minus_shuffled_consolidated_only_cosine": (
            "ordered_consolidated_only",
            "successor_shuffled_consolidated_only",
            "cosine",
        ),
        "ordered_consolidated_only_minus_shuffled_consolidated_only_proper_bits": (
            "ordered_consolidated_only",
            "successor_shuffled_consolidated_only",
            "proper_gaussian_mixture_bits_per_coordinate",
        ),
    }
    contrasts: dict[str, Any] = {}
    for contrast_index, (name, (left, right, metric)) in enumerate(
        contrast_specs.items()
    ):
        left_values = pair_values(left, metric)
        right_values = pair_values(right, metric)
        contrasts[name] = _mean_ci(
            [
                left_value - right_value
                for left_value, right_value in zip(
                    left_values, right_values, strict=True
                )
            ],
            seed=bootstrap_seed + 10_000 + contrast_index,
            resamples=bootstrap_resamples,
        )

    collapse_rows = [
        row
        for row in selected
        if int(row["arms"]["ordered"]["candidate_count"]) >= 2
    ]
    if collapse_rows:
        collapse_pair_ids = sorted(
            {str(row["pair_id"]) for row in collapse_rows}
        )

        def collapse_pair_values(
            left_arm: str,
            left_metric: str,
            right_arm: str,
            right_metric: str,
        ) -> list[float]:
            return [
                float(
                    np.mean(
                        [
                            float(row["arms"][left_arm][left_metric])
                            - float(row["arms"][right_arm][right_metric])
                            for row in collapse_rows
                            if row["pair_id"] == pair_id
                        ]
                    )
                )
                for pair_id in collapse_pair_ids
            ]

        collapse = {
            "available": True,
            "sides_with_two_or_more_modes": len(collapse_rows),
            "point_mse_minus_hindsight_best_mode_mse": _mean_ci(
                collapse_pair_values(
                    "ordered", "mse", "ordered", "best_retained_mode_mse"
                ),
                seed=bootstrap_seed + 20_000,
                resamples=bootstrap_resamples,
            ),
            "ordered_best_mode_minus_basal_point_mse": _mean_ci(
                collapse_pair_values(
                    "ordered",
                    "best_retained_mode_mse",
                    "basal",
                    "mse",
                ),
                seed=bootstrap_seed + 20_001,
                resamples=bootstrap_resamples,
            ),
            "ordered_best_mode_minus_shuffled_best_mode_mse": _mean_ci(
                collapse_pair_values(
                    "ordered",
                    "best_retained_mode_mse",
                    "successor_shuffled",
                    "best_retained_mode_mse",
                ),
                seed=bootstrap_seed + 20_002,
                resamples=bootstrap_resamples,
            ),
            "shuffled_best_mode_minus_basal_point_mse": _mean_ci(
                collapse_pair_values(
                    "successor_shuffled",
                    "best_retained_mode_mse",
                    "basal",
                    "mse",
                ),
                seed=bootstrap_seed + 20_003,
                resamples=bootstrap_resamples,
            ),
            "interpretation": (
                "non-proper oracle diagnostic; target selects the best retained "
                "mode, and unequal candidate counts can change oracle coverage"
            ),
        }
    else:
        collapse = {
            "available": False,
            "sides_with_two_or_more_modes": 0,
            "interpretation": "no evaluated context exposed two retained modes",
        }
    return {
        "available": True,
        "pairs": len(pair_ids),
        "sides": len(selected),
        "arms": arm_summaries,
        "paired_contrasts": contrasts,
        "point_mean_collapse_diagnostic": collapse,
    }


def evaluate_latent_arms(
    basal: FutureTurnTrajectory,
    codec: codec_assay.FutureTurnCodec,
    pairs: Sequence[codec_assay.DevelopmentPair],
    projector: ContextProjector,
    ordered: SurpriseGrownFutureTransition,
    shuffled: SurpriseGrownFutureTransition,
    *,
    proper_variance: np.ndarray,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Evaluate only declared development partitions with all learners frozen."""

    if len(pairs) < 1:
        raise ValueError("development evaluation needs at least one pair")
    fingerprints_before = {
        "basal": basal.fingerprint(include_state=False),
        "ordered": _learner_fingerprint(ordered),
        "successor_shuffled": _learner_fingerprint(shuffled),
    }
    results: dict[str, Any] = {}
    modes = ("original_unanswered_prefix", "long_interruption_crossover")
    for mode_index, mode in enumerate(modes):
        if mode == "original_unanswered_prefix" and not all(
            pair.side_a.original_context is not None
            and pair.side_b.original_context is not None
            for pair in pairs
        ):
            results[mode] = {
                "available": False,
                "reason": "original unanswered prefix metadata unavailable",
            }
            continue
        examples = _development_examples(basal, codec, pairs, mode=mode)
        rows: list[dict[str, Any]] = []
        for pair, side_name, example in examples:
            context = projector.transform(example.context_state)
            ordered_forecast = ordered.forecast(
                context, example.basal_prediction, remember=False
            )
            shuffled_forecast = shuffled.forecast(
                context, example.basal_prediction, remember=False
            )
            ordered_consolidated = _consolidated_only_forecast(
                ordered_forecast
            )
            shuffled_consolidated = _consolidated_only_forecast(
                shuffled_forecast
            )
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "side": side_name,
                    "arms": {
                        "basal": _prediction_metrics(
                            example.target,
                            example.basal_prediction,
                            None,
                            proper_variance=proper_variance,
                        ),
                        "ordered": _prediction_metrics(
                            example.target,
                            ordered_forecast.prediction,
                            ordered_forecast,
                            proper_variance=proper_variance,
                        ),
                        "ordered_consolidated_only": _prediction_metrics(
                            example.target,
                            ordered_consolidated.prediction,
                            ordered_consolidated,
                            proper_variance=proper_variance,
                        ),
                        "successor_shuffled": _prediction_metrics(
                            example.target,
                            shuffled_forecast.prediction,
                            shuffled_forecast,
                            proper_variance=proper_variance,
                        ),
                        "successor_shuffled_consolidated_only": (
                            _prediction_metrics(
                                example.target,
                                shuffled_consolidated.prediction,
                                shuffled_consolidated,
                                proper_variance=proper_variance,
                            )
                        ),
                    },
                }
            )
        results[mode] = {
            "available": True,
            "summaries": {
                partition: _summarize_evaluation_rows(
                    rows,
                    partition=partition,
                    bootstrap_seed=bootstrap_seed
                    + mode_index * 100_000
                    + (0 if partition == "selection" else 50_000),
                    bootstrap_resamples=bootstrap_resamples,
                )
                for partition in ("selection", "evaluation")
            },
            "pair_rows": rows,
        }

    fingerprints_after = {
        "basal": basal.fingerprint(include_state=False),
        "ordered": _learner_fingerprint(ordered),
        "successor_shuffled": _learner_fingerprint(shuffled),
    }
    if fingerprints_after != fingerprints_before:
        raise AssertionError("development evaluation modified fitted predictor state")
    if basal.observed_count != 0 or ordered.pending or shuffled.pending:
        raise AssertionError("development evaluation retained transient state")
    return {
        "context_modes": results,
        "fitted_fingerprints_before": fingerprints_before,
        "fitted_fingerprints_after": fingerprints_after,
        "all_predictors_frozen_and_unchanged": True,
        "development_partition_counts": {
            partition: sum(pair.partition == partition for pair in pairs)
            for partition in ("selection", "evaluation")
        },
        "proper_score": {
            "kind": "negative log2 density per coordinate",
            "family": (
                "causal predictive mixture with a common diagonal Gaussian "
                "covariance calibrated from cross-fitted training residuals"
            ),
            "lower_is_better": True,
            "development_targets_used_for_calibration": False,
        },
        "metric_roles": {
            "primary_causal_point_metrics": ["mse", "cosine"],
            "primary_predictive_density_metric": (
                "proper_gaussian_mixture_bits_per_coordinate"
            ),
            "matched_primary_comparisons": [
                "ordered versus basal",
                "ordered versus successor_shuffled",
            ],
            "lower_is_better": [
                "mse",
                "proper_gaussian_mixture_bits_per_coordinate",
            ],
            "higher_is_better": ["cosine"],
        },
        "particle_coverage": {
            "proper_score": False,
            "reason": "best mode is selected with hindsight after seeing the target",
        },
        "consolidated_only_diagnostic": {
            "primary": False,
            "posthoc": True,
            "construction": (
                "drop juvenile and dormant candidates from the frozen forecast, "
                "retain basal, and renormalize without learning"
            ),
            "purpose": "test whether lifecycle retention improves deployment",
        },
    }


def _config_record(config: SurpriseGrownFutureTransitionConfig) -> dict[str, Any]:
    return {
        name: getattr(config, name)
        for name in config.__dataclass_fields__
    }


def load_declared_training_and_development(
    args: argparse.Namespace,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, str],
    list[codec_assay.TrainingConversation],
    list[codec_assay.DevelopmentPair],
]:
    """Reach only the schema-4 training and declared development surfaces."""

    manifest = codec_assay._schema4_manifest(args.dataset_dir)
    verified = codec_assay.receiver._verify_manifest_payloads(
        args.dataset_dir, args.dataset_dir / "manifest.json", "dev"
    )
    conversations = codec_assay.load_training_conversations(
        args.dataset_dir, maximum=args.max_train_conversations
    )
    pairs = codec_assay.load_development_pairs(
        args.dataset_dir,
        maximum=args.max_dev_pairs,
        excluded_pair_ids=args.exclude_dev_pair_ids,
    )
    return manifest, verified, conversations, pairs


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    """Fit on schema-4 train only and evaluate schema-4 development only."""

    started = time.perf_counter()
    if not args.latent_only:
        raise ValueError("this assay supports only --latent-only execution")
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    if args.wall_limit_seconds <= 0.0:
        raise ValueError("wall limit must be positive")
    if (args.max_train_conversations or args.max_dev_pairs) and not args.smoke:
        raise ValueError("data slicing is permitted only with --smoke")

    manifest, verified, conversations, pairs = (
        load_declared_training_and_development(args)
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
    crossfit = cross_fitted_basal_transitions(
        conversations,
        sequences,
        content_dimensions=codec.content_dimensions,
        folds=args.crossfit_folds,
        fold_seed=args.crossfit_seed,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    context_matrix = np.stack(
        [example.context_state for example in crossfit.examples]
    )
    projector = ContextProjector.fit(
        context_matrix,
        dimensions=args.context_dimensions,
        scale_floor=args.context_scale_floor,
        clip=args.context_clip,
    )
    ordered, shuffled, overlay_training = train_surprise_arms(
        crossfit.examples,
        projector,
        conversations=len(conversations),
        args=args,
    )
    full_basal = recurrent_assay.fit_recurrent_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    assistant_examples = [
        example
        for example in crossfit.examples
        if example.target_turn.role_id == codec_assay.ASSISTANT_ROLE
    ]
    if not assistant_examples:
        raise ValueError("training transitions contain no assistant successors")
    residuals = np.stack(
        [example.target - example.basal_prediction for example in assistant_examples]
    )
    proper_variance = np.maximum(
        np.mean(np.square(residuals), axis=0), args.proper_variance_floor
    )
    evaluation = evaluate_latent_arms(
        full_basal,
        codec,
        pairs,
        projector,
        ordered,
        shuffled,
        proper_variance=proper_variance,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached after latent evaluation completed")

    result = {
        "schema_version": 1,
        "experiment": "surprise_grown_future_transition_latent_development",
        "status": "complete_smoke" if args.smoke else "complete_development",
        "claim_scope": {
            "schema4_train_and_development_only": True,
            "native_alm_loaded": False,
            "free_byte_generation_run": False,
            "latent_only": True,
            "anchored_future_turn_codec_reused": True,
            "existing_recurrent_one_step_ridge_is_basal_control": True,
            "training_basal_predictions_whole_conversation_cross_fitted": True,
            "evaluation_basal_fit_on_all_training_conversations": True,
            "surprise_overlay_target_arrives_after_forecast": True,
            "development_evaluation_frozen": True,
            "autograd_framework_used": False,
            "backpropagation_used": False,
            "backpropagation_through_time_used": False,
            "pca_and_ridge_are_batch_estimators": True,
            "assembly_updates_are_local_online_numpy_rules": True,
            "fully_online_local_learning_demonstrated": False,
            "goal_or_preference_learning": False,
            "particle_best_mode_is_nonproper_diagnostic": True,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "dataset": {
            "path": str(args.dataset_dir),
            "manifest_schema_version": int(manifest["schema_version"]),
            "addressed_splits": ["train", "dev"],
        },
        "development_protocol": {
            "confirmatory": False,
            "selection_evaluation_partition_reported_separately": True,
            "lifecycle_defaults_selected_after_small_smoke_runs": True,
            "smoke_training_conversation_counts": [32, 64],
            "smoke_development_pair_ids_excluded_from_this_run": list(
                KNOWN_SURPRISE_TUNING_PAIR_IDS
            ),
            "other_previously_known_smoke_pair_ids_also_excluded": list(
                codec_assay.KNOWN_SMOKE_PAIR_IDS
            ),
            "interpretation": (
                "exploratory development evidence; not a preregistered test result"
            ),
        },
        "config": {
            "raw_dimensions": args.raw_dimensions,
            "content_dimensions": args.content_dimensions,
            "active_content_features": args.active_content_features,
            "memory_decay": args.memory_decay,
            "transition_ridge": args.transition_ridge,
            "crossfit_folds": args.crossfit_folds,
            "crossfit_seed": args.crossfit_seed,
            "context_dimensions": args.context_dimensions,
            "context_scale_floor": args.context_scale_floor,
            "context_clip": args.context_clip,
            "overlay_training_passes": args.overlay_training_passes,
            "training_order_seed": args.training_order_seed,
            "successor_shuffle_seed": args.successor_shuffle_seed,
            "proper_variance_floor": args.proper_variance_floor,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "smoke": args.smoke,
            "latent_only": args.latent_only,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
            "surprise_transition": _config_record(ordered.config),
        },
        "counts": {
            "training_conversations": len(conversations),
            "training_role_tagged_events": sum(len(sequence) for sequence in sequences),
            "training_adjacent_transitions": len(crossfit.examples),
            "completed_training_assistant_turns": len(assistant_turns),
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
        },
        "codec": {
            "fit_inputs": "completed training assistant turns only",
            "anchored_during_all_transition_fits": True,
            "content_dimensions": codec.content_dimensions,
            "source_labels": list(codec.source_labels),
        },
        "cross_fitted_basal": crossfit.audit,
        "context_projector": {
            "fit_inputs": "causal recurrent states from training transitions only",
            "development_rows_used": 0,
            "input_dimensions": projector.input_dimensions,
            "output_dimensions": projector.output_dimensions,
            "fingerprint": projector.fingerprint(),
            "projected_training_state_sha256": _array_sha256(
                projector.transform(context_matrix)
            ),
        },
        "proper_score_calibration": {
            "fit_inputs": (
                "whole-conversation cross-fitted assistant-successor training "
                "residuals only"
            ),
            "rows": len(assistant_examples),
            "variance_sha256": _array_sha256(proper_variance),
            "minimum_variance": float(np.min(proper_variance)),
            "maximum_variance": float(np.max(proper_variance)),
            "development_rows_used": 0,
        },
        "surprise_overlay_training": overlay_training,
        "full_training_basal": {
            "kind": "shared one-step ridge over current event and decayed history",
            "fingerprint": full_basal.fingerprint(include_state=False),
            "training_objective": full_basal.training_objective,
            "trains_through_rollout": full_basal.trains_through_rollout,
        },
        "evaluation": evaluation,
        "wall_seconds": elapsed,
    }
    # Reject NaN and infinities rather than relying on non-standard JSON tokens.
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--latent-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="required; this runner has no native/free-generation path",
    )
    parser.add_argument("--raw-dimensions", type=int, default=512)
    parser.add_argument("--content-dimensions", type=int, default=64)
    parser.add_argument("--active-content-features", type=int, default=16)
    parser.add_argument("--memory-decay", type=float, default=0.8)
    parser.add_argument("--transition-ridge", type=float, default=1e-3)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--crossfit-seed", type=int, default=261_001)
    parser.add_argument("--context-dimensions", type=int, default=24)
    parser.add_argument("--context-scale-floor", type=float, default=1e-3)
    parser.add_argument("--context-clip", type=float, default=6.0)
    parser.add_argument("--overlay-training-passes", type=int, default=1)
    parser.add_argument("--training-order-seed", type=int, default=261_002)
    parser.add_argument("--successor-shuffle-seed", type=int, default=261_003)
    parser.add_argument("--max-context-assemblies", type=int, default=64)
    parser.add_argument("--max-residual-modes", type=int, default=128)
    parser.add_argument("--max-modes-per-context", type=int, default=4)
    parser.add_argument("--max-nursery-candidates", type=int, default=128)
    parser.add_argument("--context-bandwidth", type=float, default=1.0)
    parser.add_argument("--min-context-affinity", type=float, default=0.60)
    parser.add_argument("--residual-floor", type=float, default=0.75)
    parser.add_argument("--residual-scale-decay", type=float, default=0.99)
    parser.add_argument("--surprise-birth-z", type=float, default=0.75)
    parser.add_argument("--poor-representation-z", type=float, default=1.0)
    parser.add_argument("--nursery-context-z", type=float, default=1.2)
    parser.add_argument("--nursery-residual-z", type=float, default=2.0)
    parser.add_argument("--nursery-confirmations", type=int, default=3)
    parser.add_argument("--nursery-target-consistency", type=float, default=0.10)
    parser.add_argument("--nursery-max-age", type=int, default=512)
    parser.add_argument("--residual-learning-rate", type=float, default=0.10)
    parser.add_argument("--variance-learning-rate", type=float, default=0.05)
    parser.add_argument("--context-learning-rate", type=float, default=0.03)
    parser.add_argument("--utility-decay", type=float, default=0.99)
    parser.add_argument("--responsibility-fire-threshold", type=float, default=0.20)
    parser.add_argument("--basal-mixture-weight", type=float, default=1.0)
    parser.add_argument("--juvenile-forecast-weight", type=float, default=0.25)
    parser.add_argument("--newborn-exploration-weight", type=float, default=0.25)
    parser.add_argument("--causal-support-prior", type=float, default=4.0)
    parser.add_argument("--consolidation-min-age", type=int, default=128)
    parser.add_argument("--consolidation-min-fires", type=int, default=12)
    parser.add_argument("--consolidation-gain-bits", type=float, default=0.01)
    parser.add_argument("--dormancy-staleness", type=int, default=1_024)
    parser.add_argument("--prune-gain-bits", type=float, default=-0.01)
    parser.add_argument("--dormant-min-residence", type=int, default=64)
    parser.add_argument("--juvenile-min-recycle-age", type=int, default=256)
    parser.add_argument("--dormant-forecast-weight", type=float, default=0.05)
    parser.add_argument("--proper-variance-floor", type=float, default=1e-3)
    parser.add_argument("--bootstrap-seed", type=int, default=261_004)
    parser.add_argument("--bootstrap-resamples", type=int, default=4_000)
    parser.add_argument("--wall-limit-seconds", type=float, default=1_800.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-conversations", type=int, default=0)
    parser.add_argument("--max-dev-pairs", type=int, default=0)
    parser.add_argument(
        "--exclude-dev-pair-ids",
        nargs="*",
        default=DEFAULT_EXCLUDED_PAIR_IDS,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_ablation(args)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
