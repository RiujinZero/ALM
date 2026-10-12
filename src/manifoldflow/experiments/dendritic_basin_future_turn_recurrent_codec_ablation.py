"""Development-only recurrent turn-code supply for the frozen native writer.

This assay keeps the anchored turn codec and context-coupled byte receiver from
``dendritic_basin_future_turn_codec_ablation`` unchanged.  It replaces that
assay's one-shot causal supply route with a learned transition over prior
role-tagged turn vectors.  The transition is fitted only on observed adjacent
training events and is corrected by every observed event in an evaluation
prefix before it predicts the next turn vector.

The recurrent model remains a deterministic conditional-mean baseline.  It has
one future slot, is not trained through recursive rollout, and is neither a
particle future nor a goal/preference mechanism.  Actual free byte generation
is primary; dense-code similarity and teacher-forced BPB are diagnostics.

Only schema-4 training conversations and development crossover payloads are
reachable through the imported locked loader.  There is no configurable split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.future_turn_trajectory import (
    EncodedTurn,
    FutureTurnTrajectory,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)

_TOP_LEVEL_TURN_PATTERN = re.compile(rb"(?:\A|\n\n)(USER|ASSISTANT):[ \t]*")
TEXT_MODALITY_ID = "text"

CONDITION_NAMES = (
    "native",
    "oracle_code_gain0",
    "oracle_code_coupled",
    "oracle_code_paired_swapped",
    "one_shot_code_coupled",
    "one_shot_code_shuffled",
    "recurrent_code_coupled",
    "recurrent_code_additive",
    "recurrent_code_shuffled",
    "recurrent_code_paired_swapped",
    "recurrent_current_user_only",
    "recurrent_transition_shuffled",
)

CONTROLLED_CONTRASTS = {
    "oracle_coupled_minus_native": ("oracle_code_coupled", "native"),
    "oracle_coupled_minus_paired_swap": (
        "oracle_code_coupled",
        "oracle_code_paired_swapped",
    ),
    "one_shot_coupled_minus_native": ("one_shot_code_coupled", "native"),
    "one_shot_coupled_minus_shuffled": (
        "one_shot_code_coupled",
        "one_shot_code_shuffled",
    ),
    "recurrent_coupled_minus_native": ("recurrent_code_coupled", "native"),
    "recurrent_coupled_minus_additive": (
        "recurrent_code_coupled",
        "recurrent_code_additive",
    ),
    "recurrent_coupled_minus_code_shuffle": (
        "recurrent_code_coupled",
        "recurrent_code_shuffled",
    ),
    "recurrent_coupled_minus_paired_swap": (
        "recurrent_code_coupled",
        "recurrent_code_paired_swapped",
    ),
    "recurrent_coupled_minus_current_user_only": (
        "recurrent_code_coupled",
        "recurrent_current_user_only",
    ),
    "recurrent_coupled_minus_transition_shuffle": (
        "recurrent_code_coupled",
        "recurrent_transition_shuffled",
    ),
    "recurrent_coupled_minus_one_shot": (
        "recurrent_code_coupled",
        "one_shot_code_coupled",
    ),
}

IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/dendritic_basin_future_turn_recurrent_codec_ablation.py",
    "tests/test_dendritic_basin_future_turn_recurrent_codec_ablation.py",
    "src/manifoldflow/experiments/dendritic_basin_future_turn_codec_ablation.py",
    "src/manifoldflow/dendritic_basin/future_turn_trajectory.py",
    "src/manifoldflow/dendritic_basin/context_coupled_byte_writer.py",
    "src/manifoldflow/dendritic_basin/native_cortex.py",
    "src/manifoldflow/experiments/dendritic_basin_shared_goal_context_coupled_writer_ablation.py",
    "src/manifoldflow/experiments/dendritic_basin_shared_goal_crossover_ablation.py",
    "src/manifoldflow/experiments/dendritic_basin_shared_goal_generation_ablation.py",
    "src/manifoldflow/experiments/dendritic_basin_shared_goal_response_ablation.py",
    "src/manifoldflow/experiments/dendritic_basin_phase126_three_epoch_generation.py",
)


@dataclass(frozen=True)
class ObservedTurn:
    """One exact role-tagged event parsed from a serialized transcript."""

    role: str
    content: bytes


@dataclass(frozen=True)
class DenseForecast:
    """A next-event content forecast plus non-controlling role diagnostics."""

    vector: np.ndarray
    predicted_role: str
    observed_events: int


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = root / relative
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return result


def native_runtime_provenance(cortex: Any) -> dict[str, Any]:
    """Identify the exact native binary already loaded by ``cortex``."""

    library = getattr(cortex, "_library", None)
    raw_path = getattr(library, "_name", None)
    if not isinstance(raw_path, (str, bytes, os.PathLike)):
        raise ValueError("loaded native cortex did not expose its library path")
    path = Path(os.fsdecode(raw_path)).resolve(strict=True)
    raw_version = library.alm_version()
    if not isinstance(raw_version, bytes):
        raise ValueError("loaded native cortex returned a non-byte version")
    override = os.environ.get("ALM_NATIVE_DLL")
    loaded_via_override = False
    if override:
        loaded_via_override = Path(override).resolve(strict=True) == path
    return {
        "resolved_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": codec_assay._sha256_file(path),
        "version": raw_version.decode("ascii"),
        "alm_native_dll_override_set": override is not None,
        "loaded_via_override": loaded_via_override,
    }


def observed_turns_from_transcript(
    payload: bytes, *, generation_context: bool
) -> tuple[ObservedTurn, ...]:
    """Parse exact double-newline-framed turns without accepting quoted markers.

    Generation contexts must end in an empty assistant slot.  That slot is a
    boundary, not an observed event, and is removed from the returned sequence.
    All preceding content remains byte exact for the anchored codec.
    """

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("transcript must be nonempty bytes")
    matches = list(_TOP_LEVEL_TURN_PATTERN.finditer(payload))
    if not matches or matches[0].start() != 0:
        raise ValueError("transcript must begin with a top-level role marker")
    roles = [match.group(1).decode("ascii") for match in matches]
    if roles[0] != codec_assay.USER_ROLE or any(
        roles[index] == roles[index - 1] for index in range(1, len(roles))
    ):
        raise ValueError("top-level USER and ASSISTANT turns must alternate")

    turns = tuple(
        ObservedTurn(
            role,
            payload[
                match.end() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(payload)
                )
            ],
        )
        for index, (role, match) in enumerate(zip(roles, matches, strict=True))
    )
    if not generation_context:
        if not turns or any(
            turn.role == codec_assay.ASSISTANT_ROLE and not turn.content
            for turn in turns
        ):
            raise ValueError("completed transcripts need nonempty assistant turns")
        return turns

    if len(turns) < 2 or [turn.role for turn in turns[-2:]] != [
        codec_assay.USER_ROLE,
        codec_assay.ASSISTANT_ROLE,
    ]:
        raise ValueError("generation context must end in USER then ASSISTANT")
    if turns[-1].content.strip():
        raise ValueError("final ASSISTANT generation slot must be empty")
    if not turns[-2].content.strip():
        raise ValueError("immediate USER turn must be nonempty")
    return turns[:-1]


def _encoded_event(
    codec: codec_assay.FutureTurnCodec,
    turn: ObservedTurn,
    *,
    source_label: str,
) -> EncodedTurn:
    code = codec.encode(turn.content, role=turn.role, source=source_label)
    return EncodedTurn(
        code.content,
        source_id=TEXT_MODALITY_ID,
        role_id=turn.role,
    )


def encode_training_sequences(
    conversations: Sequence[codec_assay.TrainingConversation],
    codec: codec_assay.FutureTurnCodec,
) -> tuple[tuple[EncodedTurn, ...], ...]:
    """Encode every observed role-tagged event and audit assistant alignment."""

    sequences: list[tuple[EncodedTurn, ...]] = []
    for conversation in conversations:
        turns = observed_turns_from_transcript(
            conversation.payload, generation_context=False
        )
        parsed_assistants = tuple(
            turn.content for turn in turns if turn.role == codec_assay.ASSISTANT_ROLE
        )
        indexed_assistants = tuple(
            turn.content for turn in conversation.assistant_turns
        )
        if parsed_assistants != indexed_assistants:
            raise ValueError("parsed assistant events disagree with training index")
        sequences.append(
            tuple(
                _encoded_event(codec, turn, source_label=conversation.source_label)
                for turn in turns
            )
        )
    if not sequences or any(len(sequence) < 2 for sequence in sequences):
        raise ValueError("each recurrent training sequence needs adjacent events")
    return tuple(sequences)


def fit_recurrent_transition(
    sequences: Sequence[Sequence[EncodedTurn]],
    *,
    content_dimensions: int,
    memory_decay: float,
    ridge: float,
) -> FutureTurnTrajectory:
    """Fit the existing modality-neutral one-step recurrent transition."""

    return FutureTurnTrajectory(
        content_dimensions,
        source_ids=(TEXT_MODALITY_ID,),
        role_ids=codec_assay.ROLE_LABELS,
        horizon_widths=(content_dimensions,),
        memory_decay=memory_decay,
        ridge=ridge,
    ).fit(sequences)


def _forecast_encoded_prefix(
    model: FutureTurnTrajectory,
    prefix: Sequence[EncodedTurn],
    *,
    transition_matrix: np.ndarray | None = None,
) -> DenseForecast:
    if not prefix:
        raise ValueError("a recurrent forecast requires an observed event")
    model.reset()
    try:
        for turn in prefix:
            model.observe(turn)
        tape = model.predicted_tape(transition_matrix=transition_matrix)
        if len(tape.slots) != 1:
            raise AssertionError("one-step recurrent assay exposed multiple slots")
        slot = tape.slots[0]
        return DenseForecast(
            np.array(slot.vector, copy=True),
            str(slot.role_id),
            tape.observed_count,
        )
    finally:
        model.reset()


def forecast_context(
    model: FutureTurnTrajectory,
    codec: codec_assay.FutureTurnCodec,
    context: bytes,
    *,
    source_label: str,
    current_user_only: bool = False,
    transition_matrix: np.ndarray | None = None,
) -> DenseForecast:
    turns = observed_turns_from_transcript(context, generation_context=True)
    if current_user_only:
        turns = turns[-1:]
    encoded = tuple(
        _encoded_event(codec, turn, source_label=source_label) for turn in turns
    )
    return _forecast_encoded_prefix(model, encoded, transition_matrix=transition_matrix)


def fit_one_shot_predictor(
    turns: Sequence[codec_assay.AssistantTurn],
    codec: codec_assay.FutureTurnCodec,
    *,
    hash_dimensions: int,
    alpha: float,
) -> tuple[codec_assay.RidgeCodePredictor, float, int]:
    targets = np.asarray(
        [
            codec.encode(
                turn.content,
                role=codec_assay.ASSISTANT_ROLE,
                source=turn.source_label,
            ).content
            for turn in turns
        ]
    )
    features = np.asarray(
        [
            codec_assay.causal_turn_features(
                turn.prompt,
                turn.current_user_turn,
                role=codec_assay.USER_ROLE,
                source=turn.source_label,
                source_labels=codec.source_labels,
                hash_dimensions=hash_dimensions,
            )
            for turn in turns
        ]
    )
    predictor = codec_assay.RidgeCodePredictor.fit(features, targets, alpha=alpha)
    estimate = (
        features - predictor.feature_mean
    ) @ predictor.coefficients + predictor.target_mean
    rmse = float(np.sqrt(np.mean((estimate - targets) ** 2)))
    return predictor, rmse, int(features.shape[1])


def _quantized_code(
    codec: codec_assay.FutureTurnCodec,
    dense: np.ndarray,
    *,
    source_label: str,
) -> tuple[int, ...]:
    return codec.quantize(
        codec.code_from_content(
            dense,
            role=codec_assay.ASSISTANT_ROLE,
            source=source_label,
        )
    )


def _assignments(
    bank: codec_assay.DecoderBank,
    *,
    oracle: tuple[int, ...],
    oracle_swapped: tuple[int, ...],
    one_shot: tuple[int, ...],
    one_shot_shuffled: tuple[int, ...],
    recurrent: tuple[int, ...],
    recurrent_additive: tuple[int, ...],
    recurrent_shuffled: tuple[int, ...],
    recurrent_swapped: tuple[int, ...],
    recurrent_current_only: tuple[int, ...],
    recurrent_transition_shuffled: tuple[int, ...],
    gain: float,
) -> dict[str, codec_assay.RouteAssignment]:
    result = {
        "native": codec_assay.RouteAssignment(None, (), 0.0),
        "oracle_code_gain0": codec_assay.RouteAssignment(bank.coupled, oracle, 0.0),
        "oracle_code_coupled": codec_assay.RouteAssignment(bank.coupled, oracle, gain),
        "oracle_code_paired_swapped": codec_assay.RouteAssignment(
            bank.coupled, oracle_swapped, gain
        ),
        "one_shot_code_coupled": codec_assay.RouteAssignment(
            bank.coupled, one_shot, gain
        ),
        "one_shot_code_shuffled": codec_assay.RouteAssignment(
            bank.coupled, one_shot_shuffled, gain
        ),
        "recurrent_code_coupled": codec_assay.RouteAssignment(
            bank.coupled, recurrent, gain
        ),
        "recurrent_code_additive": codec_assay.RouteAssignment(
            bank.additive, recurrent_additive, gain
        ),
        "recurrent_code_shuffled": codec_assay.RouteAssignment(
            bank.coupled, recurrent_shuffled, gain
        ),
        "recurrent_code_paired_swapped": codec_assay.RouteAssignment(
            bank.coupled, recurrent_swapped, gain
        ),
        "recurrent_current_user_only": codec_assay.RouteAssignment(
            bank.coupled, recurrent_current_only, gain
        ),
        "recurrent_transition_shuffled": codec_assay.RouteAssignment(
            bank.coupled, recurrent_transition_shuffled, gain
        ),
    }
    if tuple(result) != CONDITION_NAMES:
        raise AssertionError("recurrent condition coverage changed")
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(left @ right / denominator)


def _dense_forecast_diagnostics(
    forecasts: Sequence[DenseForecast],
    oracle_targets: Sequence[np.ndarray],
    partitions: Sequence[str],
) -> dict[str, Any]:
    if not (len(forecasts) == len(oracle_targets) == len(partitions)):
        raise ValueError("forecast diagnostics are not aligned")
    cosines = [
        _cosine(forecast.vector, target)
        for forecast, target in zip(forecasts, oracle_targets, strict=True)
    ]
    pair_margins: dict[str, list[float]] = {
        "selection": [],
        "evaluation": [],
        "all": [],
    }
    for index in range(0, len(forecasts), 2):
        left = forecasts[index].vector
        right = forecasts[index + 1].vector
        left_target = oracle_targets[index]
        right_target = oracle_targets[index + 1]
        margin = (
            _cosine(left, left_target)
            + _cosine(right, right_target)
            - _cosine(left, right_target)
            - _cosine(right, left_target)
        )
        partition = partitions[index]
        pair_margins[partition].append(margin)
        pair_margins["all"].append(margin)
    return {
        "mean_oracle_cosine": float(np.mean(cosines)),
        "median_oracle_cosine": float(np.median(cosines)),
        "predicted_assistant_role_fraction": float(
            np.mean(
                [
                    forecast.predicted_role == codec_assay.ASSISTANT_ROLE
                    for forecast in forecasts
                ]
            )
        ),
        "mean_observed_prefix_events": float(
            np.mean([forecast.observed_events for forecast in forecasts])
        ),
        "dense_matched_minus_crossed_margin": {
            name: float(np.mean(values)) if values else None
            for name, values in pair_margins.items()
        },
    }


def _summarize_rows(
    rows: Mapping[str, list[dict[str, Any]]],
    *,
    partition: str,
    horizon: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    selected = {
        name: [
            row
            for row in arm_rows
            if partition == "all" or row["partition"] == partition
        ]
        for name, arm_rows in rows.items()
    }
    count = len(selected["native"])
    if count == 0:
        return {"pairs": 0, "arms": {}, "controlled_contrasts": {}}
    native = selected["native"]
    arms: dict[str, Any] = {}
    for condition_index, name in enumerate(CONDITION_NAMES):
        arm_rows = selected[name]
        greedy = [float(row["greedy_margin"]) for row in arm_rows]
        sampled = [float(row["sampled_margin"]) for row in arm_rows]
        greedy_advantage = [
            value - float(native[index]["greedy_margin"])
            for index, value in enumerate(greedy)
        ]
        sampled_advantage = [
            value - float(native[index]["sampled_margin"])
            for index, value in enumerate(sampled)
        ]
        greedy_accuracy = [
            float(
                np.mean(
                    [
                        row["sides"][side]["greedy_prefix_accuracy"]
                        for side in ("a", "b")
                    ]
                )
            )
            for row in arm_rows
        ]
        sampled_accuracy = [
            float(
                np.mean(
                    [
                        row["sides"][side]["sampled_prefix_accuracy"]
                        for side in ("a", "b")
                    ]
                )
            )
            for row in arm_rows
        ]
        divergence: list[int] = []
        degeneration: list[dict[str, Any]] = []
        for index, row in enumerate(arm_rows):
            for side in ("a", "b"):
                current = bytes.fromhex(row["sides"][side]["greedy_hex"])
                baseline = bytes.fromhex(native[index]["sides"][side]["greedy_hex"])
                position = codec_assay._first_divergence(current, baseline)
                divergence.append(horizon if position is None else position)
                degeneration.append(row["sides"][side]["sampled_diagnostics"])
        arms[name] = {
            "greedy_free_generation_margin": codec_assay._mean_ci(
                greedy,
                seed=bootstrap_seed + condition_index * 31,
                resamples=bootstrap_resamples,
            ),
            "sampled_free_generation_margin": codec_assay._mean_ci(
                sampled,
                seed=bootstrap_seed + condition_index * 31 + 1,
                resamples=bootstrap_resamples,
            ),
            "greedy_advantage_over_native": codec_assay._mean_ci(
                greedy_advantage,
                seed=bootstrap_seed + condition_index * 31 + 2,
                resamples=bootstrap_resamples,
            ),
            "sampled_advantage_over_native": codec_assay._mean_ci(
                sampled_advantage,
                seed=bootstrap_seed + condition_index * 31 + 3,
                resamples=bootstrap_resamples,
            ),
            "mean_greedy_reference_prefix_accuracy": float(np.mean(greedy_accuracy)),
            "mean_sampled_reference_prefix_accuracy": float(np.mean(sampled_accuracy)),
            "mean_teacher_forced_bpb_secondary": float(
                np.mean([row["mean_teacher_forced_bpb"] for row in arm_rows])
            ),
            "mean_first_greedy_divergence_from_native": float(np.mean(divergence)),
            "sampled_degeneration": {
                "mean_printable_fraction": float(
                    np.mean([row["printable_fraction"] for row in degeneration])
                ),
                "valid_utf8_fraction": float(
                    np.mean([row["valid_utf8"] for row in degeneration])
                ),
                "maximum_identical_byte_run": int(
                    max(row["maximum_identical_byte_run"] for row in degeneration)
                ),
                "outputs_with_run_at_least_16": int(
                    sum(row["maximum_identical_byte_run"] >= 16 for row in degeneration)
                ),
            },
        }
    contrasts: dict[str, Any] = {}
    for contrast_index, (label, (left_name, right_name)) in enumerate(
        CONTROLLED_CONTRASTS.items()
    ):
        left = selected[left_name]
        right = selected[right_name]
        contrasts[label] = {
            "greedy_free_generation_margin_difference": codec_assay._mean_ci(
                [
                    float(a["greedy_margin"]) - float(b["greedy_margin"])
                    for a, b in zip(left, right, strict=True)
                ],
                seed=bootstrap_seed + 10_000 + contrast_index * 37,
                resamples=bootstrap_resamples,
            ),
            "sampled_free_generation_margin_difference": codec_assay._mean_ci(
                [
                    float(a["sampled_margin"]) - float(b["sampled_margin"])
                    for a, b in zip(left, right, strict=True)
                ],
                seed=bootstrap_seed + 10_001 + contrast_index * 37,
                resamples=bootstrap_resamples,
            ),
            "teacher_forced_bpb_advantage_secondary": codec_assay._mean_ci(
                [
                    float(b["mean_teacher_forced_bpb"])
                    - float(a["mean_teacher_forced_bpb"])
                    for a, b in zip(left, right, strict=True)
                ],
                seed=bootstrap_seed + 10_002 + contrast_index * 37,
                resamples=bootstrap_resamples,
            ),
        }
    return {"pairs": count, "arms": arms, "controlled_contrasts": contrasts}


def evaluate_recurrent_codec(
    cortex: codec_assay.FrozenCortex,
    pairs: Sequence[codec_assay.DevelopmentPair],
    codec: codec_assay.FutureTurnCodec,
    one_shot: codec_assay.RidgeCodePredictor,
    recurrent: FutureTurnTrajectory,
    bank: codec_assay.DecoderBank,
    *,
    causal_hash_dimensions: int,
    gain: float,
    residual_clip: float,
    horizon: int,
    sample_seed: int,
    control_seed: int,
    transition_shuffle_seed: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Evaluate recurrent supply and its controls with actual generated bytes."""

    if len(pairs) < 2:
        raise ValueError("development evaluation requires at least two pairs")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("decoder gain must be finite and positive")
    writer_before = {name: writer.fingerprint() for name, writer in bank.items()}
    recurrent_before = recurrent.fingerprint(include_state=False)
    flat_sides = [
        (pair, side_name, pair.side_a if side_name == "a" else pair.side_b)
        for pair in pairs
        for side_name in ("a", "b")
    ]
    partitions = [pair.partition for pair, _, _ in flat_sides]
    oracle_dense = [
        codec.encode(
            side.full_response,
            role=codec_assay.ASSISTANT_ROLE,
            source=pair.source_label,
        ).content
        for pair, _, side in flat_sides
    ]
    oracle_codes = [
        _quantized_code(codec, dense, source_label=pair.source_label)
        for dense, (pair, _, _) in zip(oracle_dense, flat_sides, strict=True)
    ]
    shuffle = codec_assay.cross_pair_shuffle_indices(len(pairs), control_seed)
    shuffled_transition = recurrent.shuffled_transition(seed=transition_shuffle_seed)
    modes: dict[str, dict[str, Any]] = {}

    for mode in ("original_unanswered_prefix", "long_interruption_crossover"):
        available = all(side.original_context is not None for _, _, side in flat_sides)
        if mode == "original_unanswered_prefix" and not available:
            modes[mode] = {
                "available": False,
                "reason": "original unanswered prefix metadata unavailable",
            }
            continue
        contexts = [
            (
                side.original_context
                if mode == "original_unanswered_prefix"
                else side.interrupted_context
            )
            for _, _, side in flat_sides
        ]
        if any(context is None for context in contexts):
            raise AssertionError("available context mode contains a missing prefix")

        full_forecasts: list[DenseForecast] = []
        current_forecasts: list[DenseForecast] = []
        transition_control_forecasts: list[DenseForecast] = []
        one_shot_dense: list[np.ndarray] = []
        immediate_user_turns: list[bytes] = []
        for context, (pair, _, _) in zip(contexts, flat_sides, strict=True):
            if context is None:
                raise AssertionError("missing context reached recurrent forecast")
            current_user = codec_assay.last_user_turn_from_context(context)
            immediate_user_turns.append(current_user)
            full_forecasts.append(
                forecast_context(
                    recurrent,
                    codec,
                    context,
                    source_label=pair.source_label,
                )
            )
            current_forecasts.append(
                forecast_context(
                    recurrent,
                    codec,
                    context,
                    source_label=pair.source_label,
                    current_user_only=True,
                )
            )
            transition_control_forecasts.append(
                forecast_context(
                    recurrent,
                    codec,
                    context,
                    source_label=pair.source_label,
                    transition_matrix=shuffled_transition,
                )
            )
            one_shot_dense.append(
                one_shot.predict(
                    codec_assay.causal_turn_features(
                        context,
                        current_user,
                        role=codec_assay.USER_ROLE,
                        source=pair.source_label,
                        source_labels=codec.source_labels,
                        hash_dimensions=causal_hash_dimensions,
                    )
                )
            )

        recurrent_codes = [
            _quantized_code(codec, forecast.vector, source_label=pair.source_label)
            for forecast, (pair, _, _) in zip(full_forecasts, flat_sides, strict=True)
        ]
        current_codes = [
            _quantized_code(codec, forecast.vector, source_label=pair.source_label)
            for forecast, (pair, _, _) in zip(
                current_forecasts, flat_sides, strict=True
            )
        ]
        transition_control_codes = [
            _quantized_code(codec, forecast.vector, source_label=pair.source_label)
            for forecast, (pair, _, _) in zip(
                transition_control_forecasts, flat_sides, strict=True
            )
        ]
        one_shot_codes = [
            _quantized_code(codec, dense, source_label=pair.source_label)
            for dense, (pair, _, _) in zip(one_shot_dense, flat_sides, strict=True)
        ]

        collision_audit = {
            "oracle": codec_assay.code_control_collision_audit(oracle_codes, shuffle),
            "one_shot": codec_assay.code_control_collision_audit(
                one_shot_codes, shuffle
            ),
            "recurrent_full_history": codec_assay.code_control_collision_audit(
                recurrent_codes, shuffle
            ),
            "recurrent_current_user_only": codec_assay.code_control_collision_audit(
                current_codes, shuffle
            ),
            "recurrent_transition_shuffled": codec_assay.code_control_collision_audit(
                transition_control_codes, shuffle
            ),
        }
        forecast_diagnostics = {
            "one_shot": _dense_forecast_diagnostics(
                [
                    DenseForecast(value, codec_assay.ASSISTANT_ROLE, 0)
                    for value in one_shot_dense
                ],
                oracle_dense,
                partitions,
            ),
            "recurrent_full_history": _dense_forecast_diagnostics(
                full_forecasts, oracle_dense, partitions
            ),
            "recurrent_current_user_only": _dense_forecast_diagnostics(
                current_forecasts, oracle_dense, partitions
            ),
            "recurrent_transition_shuffled": _dense_forecast_diagnostics(
                transition_control_forecasts, oracle_dense, partitions
            ),
        }

        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITION_NAMES}
        for pair_index, pair in enumerate(pairs):
            pair_outputs: dict[str, dict[str, dict[str, Any]]] = {
                name: {} for name in CONDITION_NAMES
            }
            pair_bpbs: dict[str, list[float]] = {name: [] for name in CONDITION_NAMES}
            for side_offset, side_name in enumerate(("a", "b")):
                flat_index = pair_index * 2 + side_offset
                side = pair.side_a if side_name == "a" else pair.side_b
                context = contexts[flat_index]
                if context is None:
                    raise AssertionError("missing context reached generation")
                swapped_index = pair_index * 2 + (1 - side_offset)
                assignments = _assignments(
                    bank,
                    oracle=oracle_codes[flat_index],
                    oracle_swapped=oracle_codes[swapped_index],
                    one_shot=one_shot_codes[flat_index],
                    one_shot_shuffled=one_shot_codes[int(shuffle[flat_index])],
                    recurrent=recurrent_codes[flat_index],
                    recurrent_additive=recurrent_codes[flat_index],
                    recurrent_shuffled=recurrent_codes[int(shuffle[flat_index])],
                    recurrent_swapped=recurrent_codes[swapped_index],
                    recurrent_current_only=current_codes[flat_index],
                    recurrent_transition_shuffled=transition_control_codes[flat_index],
                    gain=gain,
                )
                draws = codec_assay._seeded_draws(
                    f"{mode}|{pair.pair_id}|{side_name}",
                    seed=sample_seed,
                    horizon=horizon,
                )
                reference = side.response_window
                bases = codec_assay._teacher_forced_native_bases(
                    cortex, context, reference
                )
                for name, assignment in assignments.items():
                    greedy = codec_assay.generate_branch(
                        cortex,
                        context,
                        assignment,
                        horizon=horizon,
                        residual_clip=residual_clip,
                        uniform_draws=None,
                    )
                    sampled = codec_assay.generate_branch(
                        cortex,
                        context,
                        assignment,
                        horizon=horizon,
                        residual_clip=residual_clip,
                        uniform_draws=draws,
                    )
                    pair_bpbs[name].append(
                        codec_assay.teacher_forced_bpb_from_bases(
                            bases,
                            reference,
                            assignment,
                            residual_clip=residual_clip,
                        )
                    )
                    pair_outputs[name][side_name] = {
                        "greedy": greedy,
                        "sampled": sampled,
                        "reference": reference,
                    }
            for name in CONDITION_NAMES:
                outputs = pair_outputs[name]
                greedy_pair = codec_assay._paired_reference_discrimination(
                    outputs["a"]["greedy"].data,
                    outputs["b"]["greedy"].data,
                    pair.side_a.response_window,
                    pair.side_b.response_window,
                    dimensions=1024,
                )
                sampled_pair = codec_assay._paired_reference_discrimination(
                    outputs["a"]["sampled"].data,
                    outputs["b"]["sampled"].data,
                    pair.side_a.response_window,
                    pair.side_b.response_window,
                    dimensions=1024,
                )
                rows[name].append(
                    {
                        "pair_id": pair.pair_id,
                        "partition": pair.partition,
                        "greedy_margin": greedy_pair["correct_minus_crossed_margin"],
                        "sampled_margin": sampled_pair["correct_minus_crossed_margin"],
                        "mean_teacher_forced_bpb": float(np.mean(pair_bpbs[name])),
                        "sides": {
                            side_name: {
                                "greedy_hex": outputs[side_name]["greedy"].data.hex(),
                                "greedy_text": outputs[side_name]["greedy"].data.decode(
                                    "utf-8", errors="backslashreplace"
                                ),
                                "sampled_hex": outputs[side_name]["sampled"].data.hex(),
                                "sampled_text": outputs[side_name][
                                    "sampled"
                                ].data.decode("utf-8", errors="backslashreplace"),
                                "greedy_probability_sha256": outputs[side_name][
                                    "greedy"
                                ].probability_sha256,
                                "sampled_probability_sha256": outputs[side_name][
                                    "sampled"
                                ].probability_sha256,
                                "greedy_prefix_accuracy": codec_assay._prefix_accuracy(
                                    outputs[side_name]["greedy"].data,
                                    outputs[side_name]["reference"],
                                ),
                                "sampled_prefix_accuracy": codec_assay._prefix_accuracy(
                                    outputs[side_name]["sampled"].data,
                                    outputs[side_name]["reference"],
                                ),
                                "greedy_diagnostics": (
                                    codec_assay._generation_diagnostics(
                                        outputs[side_name]["greedy"].data
                                    )
                                ),
                                "sampled_diagnostics": (
                                    codec_assay._generation_diagnostics(
                                        outputs[side_name]["sampled"].data
                                    )
                                ),
                            }
                            for side_name in ("a", "b")
                        },
                    }
                )

        for native_row, gain0_row in zip(
            rows["native"], rows["oracle_code_gain0"], strict=True
        ):
            if native_row != gain0_row:
                raise AssertionError("gain-zero oracle route is not exactly native")
        summaries = {
            partition: _summarize_rows(
                rows,
                partition=partition,
                horizon=horizon,
                bootstrap_seed=bootstrap_seed
                + (0 if mode == "original_unanswered_prefix" else 100_000),
                bootstrap_resamples=bootstrap_resamples,
            )
            for partition in ("selection", "evaluation", "all")
        }
        fixed_ids = {
            pair.pair_id for pair in sorted(pairs, key=lambda item: item.pair_id)[:5]
        }
        modes[mode] = {
            "available": True,
            "code_control_collision_audit": collision_audit,
            "forecast_diagnostics": forecast_diagnostics,
            "immediate_user_turn_sha256": _sha256(
                b"".join(
                    len(turn).to_bytes(8, "little") + turn
                    for turn in immediate_user_turns
                )
            ),
            "summaries": summaries,
            "pair_rows": rows,
            "fixed_samples": {
                name: [row for row in arm_rows if row["pair_id"] in fixed_ids]
                for name, arm_rows in rows.items()
            },
            "invariants": {
                "gain0_oracle_exactly_native": True,
                "cross_pair_code_shuffle_is_deranged": True,
                "paired_side_swap_is_separate": True,
                "actual_bytes_generated_not_candidate_reranked": True,
                "common_random_draws_shared_across_arms": True,
                "native_oracle_draw_identifier_matches_codec_ceiling_assay": True,
                "full_history_is_sequence_of_prior_role_tagged_vectors": True,
                "current_user_only_uses_same_fitted_transition": True,
                "shuffled_transition_preserves_analytic_history_equation": True,
                "assistant_role_binding_at_known_turn_boundary_is_supplied": True,
            },
        }

    if {name: writer.fingerprint() for name, writer in bank.items()} != writer_before:
        raise AssertionError("development generation modified a decoder writer")
    if recurrent.fingerprint(include_state=False) != recurrent_before:
        raise AssertionError("development evaluation modified the recurrent transition")
    if recurrent.observed_count != 0:
        raise AssertionError("development evaluation retained transient prefix state")
    return {
        "context_modes": modes,
        "writer_fingerprints_unchanged_by_evaluation": True,
        "recurrent_transition_unchanged_by_evaluation": True,
        "transient_recurrent_state_reset_after_evaluation": True,
        "development_partition_counts": {
            name: sum(pair.partition == name for pair in pairs)
            for name in ("selection", "evaluation")
        },
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    """Fit only on training conversations and evaluate development pairs."""

    started = time.perf_counter()
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    if args.wall_limit_seconds <= 0.0:
        raise ValueError("wall limit must be positive")
    if (args.max_train_conversations or args.max_dev_pairs) and not args.smoke:
        raise ValueError("data slicing is permitted only with --smoke")
    codec_assay._schema4_manifest(args.dataset_dir)
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
    assistant_turns = [
        turn for conversation in conversations for turn in conversation.assistant_turns
    ]
    codec = codec_assay.FutureTurnCodec.fit(
        assistant_turns,
        raw_dimensions=args.raw_dimensions,
        content_dimensions=args.content_dimensions,
        active_content_features=args.active_content_features,
    )
    one_shot, one_shot_rmse, one_shot_features = fit_one_shot_predictor(
        assistant_turns,
        codec,
        hash_dimensions=args.causal_hash_dimensions,
        alpha=args.one_shot_ridge_alpha,
    )
    recurrent_sequences = encode_training_sequences(conversations, codec)
    transition_started = time.perf_counter()
    recurrent = fit_recurrent_transition(
        recurrent_sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    transition_seconds = time.perf_counter() - transition_started

    cortex = codec_assay._load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("a frozen native checkpoint is required")
    native_runtime = native_runtime_provenance(cortex)
    embodied_at_load = int(getattr(cortex, "embodied_bytes", -1))
    bank = codec_assay.make_decoder_bank(
        persistent_feature_count=codec.persistent_feature_count,
        table_bits=args.table_bits,
        learning_rate=args.learning_rate,
        local_orders=args.local_orders,
    )
    training = codec_assay.fit_decoder_bank(
        cortex,
        conversations,
        codec,
        bank,
        response_cap=args.response_cap,
        training_seed=args.training_seed,
    )
    evaluation = evaluate_recurrent_codec(
        cortex,
        pairs,
        codec,
        one_shot,
        recurrent,
        bank,
        causal_hash_dimensions=args.causal_hash_dimensions,
        gain=args.gain,
        residual_clip=args.residual_clip,
        horizon=args.generation_horizon,
        sample_seed=args.sample_seed,
        control_seed=args.control_seed,
        transition_shuffle_seed=args.transition_shuffle_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached after completed recurrent evaluation")

    target_codes = np.asarray(
        [
            codec.encode(
                turn.content,
                role=codec_assay.ASSISTANT_ROLE,
                source=turn.source_label,
            ).content
            for turn in assistant_turns
        ]
    )
    return {
        "schema_version": 1,
        "experiment": "future_turn_recurrent_codec_native500_development",
        "status": "complete_smoke" if args.smoke else "complete_development",
        "claim_scope": {
            "anchored_codec_and_decoder_reused_unchanged": True,
            "recurrent_input_is_prior_role_tagged_turn_vectors": True,
            "transition_target_is_next_observed_turn_vector": True,
            "deterministic_conditional_mean_baseline": True,
            "one_future_slot_only": True,
            "trained_through_recursive_rollout": False,
            "particle_or_distributional_future": False,
            "goal_or_preference_learning": False,
            "native_checkpoint_frozen": True,
            "actual_free_bytes_generated": True,
            "candidate_reranking": False,
            "free_generation_primary": True,
            "teacher_forced_bpb_secondary_only": True,
            "text_parser_adapter_specific_transition_modality_neutral": True,
        },
        "implementation_files": implementation_hashes(),
        "native_runtime": native_runtime,
        "verified_inputs": verified,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": codec_assay._sha256_file(args.checkpoint),
            "embodied_bytes_at_load": embodied_at_load,
            "transient_embodied_bytes_at_report": int(
                getattr(cortex, "embodied_bytes", -1)
            ),
        },
        "config": {
            "raw_dimensions": args.raw_dimensions,
            "content_dimensions": args.content_dimensions,
            "active_content_features": args.active_content_features,
            "causal_hash_dimensions": args.causal_hash_dimensions,
            "one_shot_ridge_alpha": args.one_shot_ridge_alpha,
            "memory_decay": args.memory_decay,
            "transition_ridge": args.transition_ridge,
            "future_slots": 1,
            "table_bits": args.table_bits,
            "learning_rate": args.learning_rate,
            "local_orders": list(args.local_orders),
            "decoder_training_passes": 1,
            "response_cap": args.response_cap,
            "generation_horizon": args.generation_horizon,
            "gain": args.gain,
            "residual_clip": args.residual_clip,
            "training_seed": args.training_seed,
            "sample_seed": args.sample_seed,
            "control_seed": args.control_seed,
            "transition_shuffle_seed": args.transition_shuffle_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "smoke": args.smoke,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "completed_training_assistant_turns": len(assistant_turns),
            "training_role_tagged_events": sum(
                len(sequence) for sequence in recurrent_sequences
            ),
            "adjacent_training_event_transitions": sum(
                len(sequence) - 1 for sequence in recurrent_sequences
            ),
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
            "decoder_target_bytes_one_pass": sum(
                min(len(turn.content), args.response_cap) for turn in assistant_turns
            ),
        },
        "codec": {
            "fit_inputs": "completed training assistant turns only",
            "anchored_during_transition_fit": True,
            "dense_code_sha256": _sha256(target_codes.tobytes()),
            "dense_content_dimensions": codec.content_dimensions,
            "sparse_persistent_features": codec.persistent_feature_count,
        },
        "one_shot_baseline": {
            "kind": "centered_ridge_raw_history_plus_current_user",
            "training_rmse_diagnostic": one_shot_rmse,
            "feature_dimensions": one_shot_features,
            "parameter_count": int(one_shot.coefficients.size),
        },
        "recurrent_transition": {
            "kind": "shared_one_step_ridge_over_current_event_plus_decayed_history",
            "training_objective": recurrent.training_objective,
            "trains_through_rollout": recurrent.trains_through_rollout,
            "memory_decay": recurrent.memory_decay,
            "state_dimensions": recurrent.state_dim,
            "token_dimensions": recurrent.token_dim,
            "learned_next_token_parameters": recurrent.state_dim * recurrent.token_dim,
            "materialized_transition_entries": int(recurrent.transition_matrix.size),
            "fit_wall_seconds": transition_seconds,
            "fitted_fingerprint": recurrent.fingerprint(include_state=False),
            "source_ids_are_modalities_not_chat_domains": True,
            "assistant_role_at_generation_boundary_supplied_to_quantizer": True,
        },
        "decoder_allocation": codec_assay.decoder_allocation_audit(
            bank, codec.persistent_feature_count
        ),
        "decoder_training": training,
        "evaluation": evaluation,
        "wall_seconds": elapsed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--raw-dimensions", type=int, default=512)
    parser.add_argument("--content-dimensions", type=int, default=64)
    parser.add_argument("--active-content-features", type=int, default=16)
    parser.add_argument("--causal-hash-dimensions", type=int, default=128)
    parser.add_argument("--one-shot-ridge-alpha", type=float, default=4.0)
    parser.add_argument("--memory-decay", type=float, default=0.8)
    parser.add_argument("--transition-ridge", type=float, default=1e-3)
    parser.add_argument("--table-bits", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--local-orders", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--response-cap", type=int, default=128)
    parser.add_argument("--generation-horizon", type=int, default=128)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--residual-clip", type=float, default=2.0)
    parser.add_argument("--training-seed", type=int, default=260_930)
    parser.add_argument("--sample-seed", type=int, default=260_931)
    parser.add_argument("--control-seed", type=int, default=260_932)
    parser.add_argument("--transition-shuffle-seed", type=int, default=260_934)
    parser.add_argument("--bootstrap-seed", type=int, default=260_933)
    parser.add_argument("--bootstrap-resamples", type=int, default=4_000)
    parser.add_argument("--wall-limit-seconds", type=float, default=7_200.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-conversations", type=int, default=0)
    parser.add_argument("--max-dev-pairs", type=int, default=0)
    parser.add_argument(
        "--exclude-dev-pair-ids",
        nargs="*",
        default=codec_assay.KNOWN_SMOKE_PAIR_IDS,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_ablation(args)
    payload = json.dumps(report, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
