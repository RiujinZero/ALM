"""Zero-backprop receiver-convergence assay for recurrent future-turn codes.

This targeted development assay freezes the anchored turn codec, the intact and
role/source-preserving successor-shuffled turn transitions, and the native ALM
checkpoint.  It varies only cumulative local receiver exposure.  Two receiver
tracks are trained on the same target bytes and causal native base laws:

``legacy_oracle``
    The shared coupled writer receives the actual encoded assistant turn.  Its
    first pass must exactly replay the preceding successor-shuffle result.

``oof_intact`` / ``oof_successor_shuffled``
    Allocation-matched writers receive causal next-turn codes made by
    whole-conversation cross-fitted transitions.  The latter transition was fit
    after an exact successor-content derangement within target role/source
    strata.  Thus this track trains on the kind of predicted code supplied at
    inference, without fitting a transition on the conversation it forecasts.

All learning is PCA/quantization, closed-form ridge, and the bounded writer's
local normalized delta rule.  The 500 MB cortex is inference-only; there is no
autograd, backpropagation, or backpropagation through time.  Natural logs remain
teacher-forced: no generated assistant state is paired with a logged user reply.

Only schema-4 training and development payloads are addressable.  There is no
test-split switch and no candidate reranking; the primary endpoint is sampled
free generation from the native byte stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.context_coupled_byte_writer import (
    BaseAnchoredFitDiagnostics,
    ContextCoupledByteWriter,
)
from manifoldflow.dendritic_basin.future_turn_trajectory import EncodedTurn
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_recurrent_codec_ablation as recurrent_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_successor_shuffle_ablation as successor_assay,
)

TRACK_NAMES = (
    "legacy_oracle",
    "oof_intact",
    "oof_successor_shuffled",
)
DEFAULT_EXPOSURES = (1, 2, 4, 8)
IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/"
    "dendritic_basin_future_turn_decoder_curve_ablation.py",
    "tests/test_dendritic_basin_future_turn_decoder_curve_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_future_turn_successor_shuffle_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_future_turn_recurrent_codec_ablation.py",
    "src/manifoldflow/experiments/dendritic_basin_future_turn_codec_ablation.py",
    "src/manifoldflow/dendritic_basin/future_turn_trajectory.py",
    "src/manifoldflow/dendritic_basin/context_coupled_byte_writer.py",
    "src/manifoldflow/dendritic_basin/native_cortex.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_shared_goal_context_coupled_writer_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_shared_goal_crossover_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_shared_goal_generation_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_shared_goal_response_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_phase126_three_epoch_generation.py",
)


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


def _strict_exposures(values: Sequence[int]) -> tuple[int, ...]:
    exposures = tuple(int(value) for value in values)
    if not exposures or exposures[0] != 1:
        raise ValueError("receiver exposures must begin at one full pass")
    if any(value < 1 for value in exposures):
        raise ValueError("receiver exposures must be positive")
    if any(
        right <= left for left, right in zip(exposures, exposures[1:], strict=False)
    ):
        raise ValueError("receiver exposures must be strictly increasing")
    if exposures[-1] > 32:
        raise ValueError("receiver exposure ceiling is 32 passes")
    return exposures


def conversation_fold_assignment(
    conversations: Sequence[codec_assay.TrainingConversation],
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Return a balanced deterministic whole-conversation fold assignment."""

    if folds < 2 or folds > len(conversations):
        raise ValueError("cross-fit fold count is outside conversation count")
    order = np.random.default_rng(seed).permutation(len(conversations))
    assignment = np.empty(len(conversations), dtype=np.int64)
    assignment[order] = np.arange(len(conversations), dtype=np.int64) % folds
    assignment.setflags(write=False)
    return assignment


def _code_sequence_sha256(
    conversations: Sequence[codec_assay.TrainingConversation],
    codes: Sequence[Sequence[tuple[int, ...]]],
) -> str:
    if len(conversations) != len(codes):
        raise ValueError("code manifest does not align with conversations")
    digest = hashlib.sha256()
    for conversation, rows in zip(conversations, codes, strict=True):
        if len(rows) != len(conversation.assistant_turns):
            raise ValueError("code manifest does not align with assistant turns")
        episode = conversation.episode_id.encode("utf-8")
        digest.update(len(episode).to_bytes(4, "little"))
        digest.update(episode)
        for code in rows:
            array = np.ascontiguousarray(code, dtype="<u8")
            digest.update(len(array).to_bytes(4, "little"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def oracle_training_codes(
    conversations: Sequence[codec_assay.TrainingConversation],
    codec: codec_assay.FutureTurnCodec,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(
            codec.quantize(
                codec.encode(
                    turn.content,
                    role=codec_assay.ASSISTANT_ROLE,
                    source=turn.source_label,
                )
            )
            for turn in conversation.assistant_turns
        )
        for conversation in conversations
    )


@dataclass(frozen=True)
class CrossFitCodes:
    intact: tuple[tuple[tuple[int, ...], ...], ...]
    successor_shuffled: tuple[tuple[tuple[int, ...], ...], ...]
    fold_assignment: np.ndarray
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        assignment = np.array(self.fold_assignment, dtype=np.int64, copy=True)
        assignment.setflags(write=False)
        object.__setattr__(self, "fold_assignment", assignment)


def _forecast_assistant_codes(
    model: Any,
    sequence: Sequence[EncodedTurn],
    codec: codec_assay.FutureTurnCodec,
    *,
    source_label: str,
) -> tuple[tuple[int, ...], ...]:
    prefix: list[EncodedTurn] = []
    result: list[tuple[int, ...]] = []
    for event in sequence:
        if event.role_id == codec_assay.ASSISTANT_ROLE:
            forecast = recurrent_assay._forecast_encoded_prefix(model, prefix)
            result.append(
                recurrent_assay._quantized_code(
                    codec, forecast.vector, source_label=source_label
                )
            )
        prefix.append(event)
    model.reset()
    return tuple(result)


def cross_fitted_training_codes(
    conversations: Sequence[codec_assay.TrainingConversation],
    sequences: Sequence[Sequence[EncodedTurn]],
    codec: codec_assay.FutureTurnCodec,
    *,
    folds: int,
    fold_seed: int,
    successor_shuffle_seed: int,
    memory_decay: float,
    ridge: float,
) -> CrossFitCodes:
    """Forecast each training response with transitions fit on other conversations."""

    if len(conversations) != len(sequences):
        raise ValueError("training conversations and encoded sequences do not align")
    assignment = conversation_fold_assignment(
        conversations, folds=folds, seed=fold_seed
    )
    intact_rows: list[tuple[tuple[int, ...], ...] | None] = [
        None
    ] * len(conversations)
    shuffled_rows: list[tuple[tuple[int, ...], ...] | None] = [
        None
    ] * len(conversations)
    fold_audits: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(folds):
        training = tuple(
            sequence
            for index, sequence in enumerate(sequences)
            if int(assignment[index]) != fold
        )
        held_out = [
            index
            for index in range(len(conversations))
            if int(assignment[index]) == fold
        ]
        intact = recurrent_assay.fit_recurrent_transition(
            training,
            content_dimensions=codec.content_dimensions,
            memory_decay=memory_decay,
            ridge=ridge,
        )
        shuffled, shuffle_audit = successor_assay.fit_successor_shuffled_transition(
            training,
            content_dimensions=codec.content_dimensions,
            memory_decay=memory_decay,
            ridge=ridge,
            seed=successor_shuffle_seed + fold,
        )
        for index in held_out:
            intact_code = _forecast_assistant_codes(
                intact,
                sequences[index],
                codec,
                source_label=conversations[index].source_label,
            )
            shuffled_code = _forecast_assistant_codes(
                shuffled,
                sequences[index],
                codec,
                source_label=conversations[index].source_label,
            )
            expected = len(conversations[index].assistant_turns)
            if len(intact_code) != expected or len(shuffled_code) != expected:
                raise AssertionError("cross-fitted code count disagrees with index")
            intact_rows[index] = intact_code
            shuffled_rows[index] = shuffled_code
        fold_audits.append(
            {
                "fold": fold,
                "fit_conversations": len(training),
                "held_out_conversations": len(held_out),
                "intact_transition_fingerprint": intact.fingerprint(
                    include_state=False
                ),
                "successor_shuffled_transition_fingerprint": shuffled.fingerprint(
                    include_state=False
                ),
                "successor_shuffle_audit": shuffle_audit,
            }
        )
    if any(row is None for row in intact_rows + shuffled_rows):
        raise AssertionError("cross-fitting left a training conversation unforecast")
    intact_codes = tuple(row for row in intact_rows if row is not None)
    shuffled_codes = tuple(row for row in shuffled_rows if row is not None)
    oracle = oracle_training_codes(conversations, codec)
    intact_flat = [code for conversation in intact_codes for code in conversation]
    shuffled_flat = [code for conversation in shuffled_codes for code in conversation]
    oracle_flat = [code for conversation in oracle for code in conversation]
    content_width = codec.active_content_features
    oracle_features = {value for rows in oracle for code in rows for value in code}
    intact_features = {value for code in intact_flat for value in code}
    shuffled_features = {value for code in shuffled_flat for value in code}
    distinct = [
        left != right
        for left, right in zip(intact_flat, shuffled_flat, strict=True)
    ]
    overlap_counts: list[int] = []
    overlap_jaccards: list[float] = []
    exact_content_codes = 0
    unseen_occurrences = 0
    predicted_occurrences = 0
    turns_with_unseen = 0
    fold_oracle_support = {
        fold: {
            feature
            for conversation_index, rows in enumerate(oracle)
            if int(assignment[conversation_index]) != fold
            for code in rows
            for feature in code[:content_width]
        }
        for fold in range(folds)
    }
    flat_index = 0
    for conversation_index, rows in enumerate(intact_codes):
        support = fold_oracle_support[int(assignment[conversation_index])]
        for code in rows:
            oracle_code = oracle_flat[flat_index]
            predicted_content = set(code[:content_width])
            oracle_content = set(oracle_code[:content_width])
            overlap_counts.append(len(predicted_content & oracle_content))
            overlap_jaccards.append(
                len(predicted_content & oracle_content)
                / max(len(predicted_content | oracle_content), 1)
            )
            exact_content_codes += code[:content_width] == oracle_code[:content_width]
            unseen = sum(feature not in support for feature in code[:content_width])
            unseen_occurrences += unseen
            predicted_occurrences += content_width
            turns_with_unseen += unseen > 0
            flat_index += 1
    audit = {
        "folds": folds,
        "fold_seed": fold_seed,
        "fold_assignment_sha256": _sha256(
            np.ascontiguousarray(assignment, dtype="<i8").tobytes()
        ),
        "fold_counts": {
            str(fold): int(np.sum(assignment == fold)) for fold in range(folds)
        },
        "folds_detail": fold_audits,
        "assistant_codes": len(intact_flat),
        "intact_vs_successor_shuffled_distinct_fraction": float(
            np.mean(distinct)
        ),
        "intact_vs_successor_shuffled_equal_codes": int(
            len(distinct) - sum(distinct)
        ),
        "feature_support": {
            "oracle_unique": len(oracle_features),
            "oof_intact_unique": len(intact_features),
            "oof_successor_shuffled_unique": len(shuffled_features),
            "oof_intact_unseen_from_oracle": len(intact_features - oracle_features),
            "oof_successor_shuffled_unseen_from_oracle": len(
                shuffled_features - oracle_features
            ),
            "oof_intact_oracle_jaccard": len(intact_features & oracle_features)
            / max(len(intact_features | oracle_features), 1),
            "oof_successor_shuffled_oracle_jaccard": len(
                shuffled_features & oracle_features
            )
            / max(len(shuffled_features | oracle_features), 1),
            "mean_intact_oracle_shared_content_features": float(
                np.mean(overlap_counts)
            ),
            "mean_intact_oracle_content_jaccard": float(
                np.mean(overlap_jaccards)
            ),
            "exact_intact_oracle_content_code_fraction": exact_content_codes
            / max(len(intact_flat), 1),
            "intact_content_feature_occurrence_unseen_from_fold_train_fraction": (
                unseen_occurrences / max(predicted_occurrences, 1)
            ),
            "intact_turns_with_unseen_fold_train_content_feature_fraction": (
                turns_with_unseen / max(len(intact_flat), 1)
            ),
        },
        "fit_wall_seconds": time.perf_counter() - started,
    }
    return CrossFitCodes(intact_codes, shuffled_codes, assignment, audit)


def _new_coupled_writer(
    *,
    persistent_feature_count: int,
    table_bits: int,
    learning_rate: float,
    local_orders: Sequence[int],
) -> ContextCoupledByteWriter:
    return ContextCoupledByteWriter(
        "coupled",
        alphabet_size=256,
        table_bits=table_bits,
        learning_rate=learning_rate,
        local_orders=tuple(local_orders),
        max_persistent_features=persistent_feature_count,
    )


def _accumulate_fit(
    total: dict[str, float], diagnostics: BaseAnchoredFitDiagnostics
) -> None:
    total["episodes"] += 1.0
    total["symbols"] += diagnostics.symbols
    total["symbol_updates"] += diagnostics.symbol_updates
    total["raw_feature_activations"] += diagnostics.raw_feature_activations
    total["unique_feature_row_updates"] += diagnostics.unique_feature_row_updates
    total["scalar_weight_slots_touched"] += diagnostics.scalar_weight_slots_touched
    total["base_nll_nats"] += diagnostics.base_nll_nats
    total["receiver_nll_nats"] += diagnostics.anchored_nll_nats
    total["maximum_absolute_update"] = max(
        total["maximum_absolute_update"], diagnostics.maximum_absolute_update
    )


def fit_receiver_pass(
    cortex: codec_assay.FrozenCortex,
    conversations: Sequence[codec_assay.TrainingConversation],
    writers: Mapping[str, ContextCoupledByteWriter],
    codes: Mapping[str, Sequence[Sequence[tuple[int, ...]]]],
    *,
    response_cap: int,
    training_seed: int,
) -> dict[str, Any]:
    """Advance matched writers through one identical causal native-base pass."""

    if response_cap <= 0:
        raise ValueError("response cap must be positive")
    if tuple(writers) != TRACK_NAMES or tuple(codes) != TRACK_NAMES:
        raise ValueError("receiver track coverage/order changed")
    for name in TRACK_NAMES:
        if len(codes[name]) != len(conversations):
            raise ValueError(f"{name} code rows do not align with conversations")
    totals = {
        name: {
            "episodes": 0.0,
            "symbols": 0.0,
            "symbol_updates": 0.0,
            "raw_feature_activations": 0.0,
            "unique_feature_row_updates": 0.0,
            "scalar_weight_slots_touched": 0.0,
            "base_nll_nats": 0.0,
            "receiver_nll_nats": 0.0,
            "maximum_absolute_update": 0.0,
        }
        for name in TRACK_NAMES
    }
    order = np.random.default_rng(training_seed).permutation(len(conversations))
    response_digest = hashlib.sha256()
    started = time.perf_counter()
    for raw_index in order:
        conversation_index = int(raw_index)
        conversation = conversations[conversation_index]
        cortex.reset_episode()
        cursor = 0
        for turn_index, turn in enumerate(conversation.assistant_turns):
            if turn.content_start < cursor:
                raise ValueError("assistant training windows overlap")
            codec_assay._advance_chunk(
                cortex, conversation.payload[cursor : turn.content_start]
            )
            cursor = turn.content_start
            target = turn.content[:response_cap]
            bases = np.empty((len(target), 256), dtype=np.float64)
            for position, symbol in enumerate(target):
                bases[position] = codec_assay._normalized_native_distribution(
                    cortex.predict_next_distribution()
                )
                cortex.advance_observation(symbol, learn=False)
                cursor += 1
            response_digest.update(turn.episode_id.encode("utf-8"))
            response_digest.update(turn.turn_index.to_bytes(4, "little"))
            response_digest.update(target)
            for name in TRACK_NAMES:
                state = codes[name][conversation_index][turn_index]
                diagnostics = writers[name].fit_base_anchored_episode(
                    target, bases, state, prefix=b""
                )
                _accumulate_fit(totals[name], diagnostics)
        cortex.reset_episode()
    for total in totals.values():
        symbols = max(total["symbols"], 1.0)
        total["mean_base_nll_nats"] = total["base_nll_nats"] / symbols
        total["mean_receiver_nll_nats"] = total["receiver_nll_nats"] / symbols
        total["mean_residual_improvement_nats"] = (
            total["base_nll_nats"] - total["receiver_nll_nats"]
        ) / symbols
        total["episodes"] = int(total["episodes"])
        total["symbols"] = int(total["symbols"])
        total["symbol_updates"] = int(total["symbol_updates"])
        total["raw_feature_activations"] = int(total["raw_feature_activations"])
        total["unique_feature_row_updates"] = int(
            total["unique_feature_row_updates"]
        )
        total["scalar_weight_slots_touched"] = int(
            total["scalar_weight_slots_touched"]
        )
    counts = {(row["episodes"], row["symbols"]) for row in totals.values()}
    base_losses = {row["base_nll_nats"] for row in totals.values()}
    if len(counts) != 1 or len(base_losses) != 1:
        raise AssertionError("matched receiver tracks saw different byte/base streams")
    return {
        "conversation_order_sha256": _sha256(order.tobytes()),
        "response_sequence_sha256": response_digest.hexdigest(),
        "writers": totals,
        "matched_episode_symbol_and_base_stream": True,
        "wall_seconds": time.perf_counter() - started,
    }


@dataclass(frozen=True)
class ReceiverCurve:
    checkpoints: Mapping[int, Mapping[str, ContextCoupledByteWriter]]
    training: Mapping[str, Any]
    code_feature_support: Mapping[str, frozenset[int]]


def fit_receiver_curve(
    cortex: codec_assay.FrozenCortex,
    conversations: Sequence[codec_assay.TrainingConversation],
    codec: codec_assay.FutureTurnCodec,
    cross_fit: CrossFitCodes,
    *,
    exposures: Sequence[int],
    table_bits: int,
    learning_rate: float,
    local_orders: Sequence[int],
    response_cap: int,
    training_seed: int,
) -> ReceiverCurve:
    schedule = _strict_exposures(exposures)
    if response_cap <= 0:
        raise ValueError("response cap must be positive")
    oracle = oracle_training_codes(conversations, codec)
    code_rows = {
        "legacy_oracle": oracle,
        "oof_intact": cross_fit.intact,
        "oof_successor_shuffled": cross_fit.successor_shuffled,
    }
    code_feature_support = {
        name: frozenset(
            feature
            for conversation in rows
            for code in conversation
            for feature in code
        )
        for name, rows in code_rows.items()
    }
    writers = {
        name: _new_coupled_writer(
            persistent_feature_count=codec.persistent_feature_count,
            table_bits=table_bits,
            learning_rate=learning_rate,
            local_orders=local_orders,
        )
        for name in TRACK_NAMES
    }
    checkpoints: dict[int, dict[str, ContextCoupledByteWriter]] = {}
    passes: list[dict[str, Any]] = []
    maximum = schedule[-1]
    for completed in range(1, maximum + 1):
        interval = fit_receiver_pass(
            cortex,
            conversations,
            writers,
            code_rows,
            response_cap=response_cap,
            training_seed=training_seed,
        )
        interval["completed_pass"] = completed
        passes.append(interval)
        if completed in schedule:
            checkpoints[completed] = {
                name: ContextCoupledByteWriter.from_snapshot(writer.snapshot())
                for name, writer in writers.items()
            }
    expected_symbols = sum(
        min(len(turn.content), response_cap)
        for conversation in conversations
        for turn in conversation.assistant_turns
    )
    if any(
        writer.symbols_seen != expected_symbols * maximum
        or writer.episodes_seen
        != sum(len(row.assistant_turns) for row in conversations) * maximum
        for writer in writers.values()
    ):
        raise AssertionError("cumulative receiver exposure accounting diverged")
    return ReceiverCurve(
        checkpoints=checkpoints,
        training={
            "exposures": list(schedule),
            "maximum_completed_passes": maximum,
            "per_pass": passes,
            "code_manifests": {
                name: _code_sequence_sha256(conversations, rows)
                for name, rows in code_rows.items()
            },
            "code_feature_support": {
                name: {
                    "unique_features": len(support),
                    "sorted_feature_ids_sha256": _sha256(
                        np.ascontiguousarray(sorted(support), dtype="<u8").tobytes()
                    ),
                }
                for name, support in code_feature_support.items()
            },
            "checkpoint_writer_fingerprints": {
                str(exposure): {
                    name: writer.fingerprint()
                    for name, writer in checkpoints[exposure].items()
                }
                for exposure in schedule
            },
            "no_backpropagation": True,
            "native_cortex_updated": False,
        },
        code_feature_support=code_feature_support,
    )


def _flat_sides(
    pairs: Sequence[codec_assay.DevelopmentPair],
) -> list[tuple[codec_assay.DevelopmentPair, str, codec_assay.DevelopmentSide]]:
    return [
        (pair, side_name, pair.side_a if side_name == "a" else pair.side_b)
        for pair in pairs
        for side_name in ("a", "b")
    ]


def _evaluation_arm_names(exposures: Sequence[int]) -> tuple[str, ...]:
    names = ["native", "gain0"]
    for exposure in exposures:
        names.extend(
            (
                f"legacy_p{exposure}_intact",
                f"legacy_p{exposure}_successor_shuffled",
                f"oof_p{exposure}_intact",
                f"oof_p{exposure}_successor_shuffled",
                f"oof_p{exposure}_intact_paired_swap",
            )
        )
    final = exposures[-1]
    names.extend(
        (
            f"legacy_p{final}_oracle",
            f"legacy_p{final}_oracle_paired_swap",
        )
    )
    return tuple(names)


def _arm_assignments(
    checkpoints: Mapping[int, Mapping[str, ContextCoupledByteWriter]],
    exposures: Sequence[int],
    *,
    intact_code: tuple[int, ...],
    shuffled_code: tuple[int, ...],
    paired_intact_code: tuple[int, ...],
    oracle_code: tuple[int, ...],
    paired_oracle_code: tuple[int, ...],
    gain: float,
) -> dict[str, codec_assay.RouteAssignment]:
    first = exposures[0]
    assignments: dict[str, codec_assay.RouteAssignment] = {
        "native": codec_assay.RouteAssignment(None, (), 0.0),
        "gain0": codec_assay.RouteAssignment(
            checkpoints[first]["legacy_oracle"], intact_code, 0.0
        ),
    }
    for exposure in exposures:
        writers = checkpoints[exposure]
        assignments.update(
            {
                f"legacy_p{exposure}_intact": codec_assay.RouteAssignment(
                    writers["legacy_oracle"], intact_code, gain
                ),
                f"legacy_p{exposure}_successor_shuffled": (
                    codec_assay.RouteAssignment(
                        writers["legacy_oracle"], shuffled_code, gain
                    )
                ),
                f"oof_p{exposure}_intact": codec_assay.RouteAssignment(
                    writers["oof_intact"], intact_code, gain
                ),
                f"oof_p{exposure}_successor_shuffled": (
                    codec_assay.RouteAssignment(
                        writers["oof_successor_shuffled"], shuffled_code, gain
                    )
                ),
                f"oof_p{exposure}_intact_paired_swap": (
                    codec_assay.RouteAssignment(
                        writers["oof_intact"], paired_intact_code, gain
                    )
                ),
            }
        )
    final = exposures[-1]
    assignments.update(
        {
            f"legacy_p{final}_oracle": codec_assay.RouteAssignment(
                checkpoints[final]["legacy_oracle"], oracle_code, gain
            ),
            f"legacy_p{final}_oracle_paired_swap": codec_assay.RouteAssignment(
                checkpoints[final]["legacy_oracle"], paired_oracle_code, gain
            ),
        }
    )
    expected = _evaluation_arm_names(exposures)
    if tuple(assignments) != expected:
        raise AssertionError("decoder curve arm coverage changed")
    return assignments


def _selected_rows(
    rows: Mapping[str, Sequence[dict[str, Any]]], partition: str
) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            row
            for row in arm
            if partition == "all" or row["partition"] == partition
        ]
        for name, arm in rows.items()
    }


def _health(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [
        row["sides"][side]["sampled_diagnostics"]
        for row in rows
        for side in ("a", "b")
    ]
    return {
        "mean_printable_fraction": float(
            np.mean([row["printable_fraction"] for row in diagnostics])
        ),
        "valid_utf8_fraction": float(
            np.mean([row["valid_utf8"] for row in diagnostics])
        ),
        "maximum_identical_byte_run": int(
            max(row["maximum_identical_byte_run"] for row in diagnostics)
        ),
        "outputs_with_run_at_least_16": int(
            sum(row["maximum_identical_byte_run"] >= 16 for row in diagnostics)
        ),
        "mean_unique_byte_fraction": float(
            np.mean([row["unique_byte_fraction"] for row in diagnostics])
        ),
        "newline_fraction": float(
            np.mean([row["contains_newline"] for row in diagnostics])
        ),
    }


def _contrast_interval(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if [row["pair_id"] for row in left] != [row["pair_id"] for row in right]:
        raise ValueError("controlled contrast pair IDs/order do not align")
    return codec_assay._mean_ci(
        [
            float(a["sampled_margin"]) - float(b["sampled_margin"])
            for a, b in zip(left, right, strict=True)
        ],
        seed=seed,
        resamples=resamples,
    )


def _summarize_curve_rows(
    rows: Mapping[str, Sequence[dict[str, Any]]],
    *,
    exposures: Sequence[int],
    partition: str,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    selected = _selected_rows(rows, partition)
    count = len(selected["native"])
    if count == 0:
        return {"pairs": 0, "arms": {}, "controlled_contrasts": {}}
    arms: dict[str, Any] = {}
    for index, (name, arm) in enumerate(selected.items()):
        arms[name] = {
            "sampled_free_generation_margin": codec_assay._mean_ci(
                [float(row["sampled_margin"]) for row in arm],
                seed=bootstrap_seed + index * 31,
                resamples=bootstrap_resamples,
            ),
            "mean_teacher_forced_bpb_secondary": float(
                np.mean([row["mean_teacher_forced_bpb"] for row in arm])
            ),
            "sampled_degeneration": _health(arm),
        }
    contrasts: dict[str, Any] = {}
    offset = 10_000
    for exposure in exposures:
        for track_index, (label, left, right) in enumerate(
            (
                (
                    "legacy",
                    f"legacy_p{exposure}_intact",
                    f"legacy_p{exposure}_successor_shuffled",
                ),
                (
                    "oof_matched",
                    f"oof_p{exposure}_intact",
                    f"oof_p{exposure}_successor_shuffled",
                ),
            )
        ):
            contrasts[f"{label}_p{exposure}_intact_minus_successor_shuffled"] = {
                "sampled_free_generation_margin_difference": _contrast_interval(
                    selected[left],
                    selected[right],
                    seed=bootstrap_seed + offset + exposure * 101 + track_index,
                    resamples=bootstrap_resamples,
                )
            }
        contrasts[f"oof_p{exposure}_intact_minus_paired_swap"] = {
            "sampled_free_generation_margin_difference": _contrast_interval(
                selected[f"oof_p{exposure}_intact"],
                selected[f"oof_p{exposure}_intact_paired_swap"],
                seed=bootstrap_seed + offset + exposure * 101 + 2,
                resamples=bootstrap_resamples,
            )
        }
    first = exposures[0]
    final = exposures[-1]
    for track_index, (label, intact_prefix, shuffled_prefix) in enumerate(
        (
            ("legacy", "legacy", "legacy"),
            ("oof_matched", "oof", "oof"),
        )
    ):
        left_final = selected[f"{intact_prefix}_p{final}_intact"]
        right_final = selected[
            f"{shuffled_prefix}_p{final}_successor_shuffled"
        ]
        left_first = selected[f"{intact_prefix}_p{first}_intact"]
        right_first = selected[
            f"{shuffled_prefix}_p{first}_successor_shuffled"
        ]
        values = [
            (float(lf["sampled_margin"]) - float(rf["sampled_margin"]))
            - (float(l1["sampled_margin"]) - float(r1["sampled_margin"]))
            for lf, rf, l1, r1 in zip(
                left_final, right_final, left_first, right_first, strict=True
            )
        ]
        contrasts[f"{label}_p{final}_minus_p{first}_contrast_change"] = {
            "sampled_free_generation_difference_in_differences": (
                codec_assay._mean_ci(
                    values,
                    seed=bootstrap_seed + 20_000 + track_index,
                    resamples=bootstrap_resamples,
                )
            )
        }
    same_writer_specificity_change = [
        (float(lf["sampled_margin"]) - float(rf["sampled_margin"]))
        - (float(l1["sampled_margin"]) - float(r1["sampled_margin"]))
        for lf, rf, l1, r1 in zip(
            selected[f"oof_p{final}_intact"],
            selected[f"oof_p{final}_intact_paired_swap"],
            selected[f"oof_p{first}_intact"],
            selected[f"oof_p{first}_intact_paired_swap"],
            strict=True,
        )
    ]
    contrasts[f"oof_p{final}_minus_p{first}_paired_swap_contrast_change"] = {
        "sampled_free_generation_difference_in_differences": (
            codec_assay._mean_ci(
                same_writer_specificity_change,
                seed=bootstrap_seed + 30_000,
                resamples=bootstrap_resamples,
            )
        )
    }
    contrasts[f"legacy_p{final}_oracle_minus_paired_swap"] = {
        "sampled_free_generation_margin_difference": _contrast_interval(
            selected[f"legacy_p{final}_oracle"],
            selected[f"legacy_p{final}_oracle_paired_swap"],
            seed=bootstrap_seed + 30_001,
            resamples=bootstrap_resamples,
        )
    }
    contrasts[f"oof_p{final}_intact_minus_native"] = {
        "sampled_free_generation_margin_difference": _contrast_interval(
            selected[f"oof_p{final}_intact"],
            selected["native"],
            seed=bootstrap_seed + 30_002,
            resamples=bootstrap_resamples,
        )
    }
    return {"pairs": count, "arms": arms, "controlled_contrasts": contrasts}


def _codes_for_contexts(
    pairs: Sequence[codec_assay.DevelopmentPair],
    codec: codec_assay.FutureTurnCodec,
    intact: Any,
    shuffled: Any,
    contexts: Sequence[bytes],
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[str]]:
    flat = _flat_sides(pairs)
    intact_codes: list[tuple[int, ...]] = []
    shuffled_codes: list[tuple[int, ...]] = []
    state_hashes: list[str] = []
    for context, (pair, _, _) in zip(contexts, flat, strict=True):
        matched = successor_assay.matched_forecast_context(
            intact,
            shuffled,
            codec,
            context,
            source_label=pair.source_label,
        )
        intact_codes.append(
            recurrent_assay._quantized_code(
                codec, matched.intact.vector, source_label=pair.source_label
            )
        )
        shuffled_codes.append(
            recurrent_assay._quantized_code(
                codec, matched.shuffled.vector, source_label=pair.source_label
            )
        )
        state_hashes.append(matched.predictor_state_sha256)
    return intact_codes, shuffled_codes, state_hashes


def _code_distinctness(
    pairs: Sequence[codec_assay.DevelopmentPair],
    intact_codes: Sequence[tuple[int, ...]],
    shuffled_codes: Sequence[tuple[int, ...]],
    *,
    partition: str,
) -> dict[str, Any]:
    selected_pairs = [
        index
        for index, pair in enumerate(pairs)
        if partition == "all" or pair.partition == partition
    ]
    side_indexes = [
        2 * pair_index + side_offset
        for pair_index in selected_pairs
        for side_offset in (0, 1)
    ]
    intact_vs_shuffled = [
        intact_codes[index] != shuffled_codes[index] for index in side_indexes
    ]
    paired_intact = [
        intact_codes[2 * pair_index] != intact_codes[2 * pair_index + 1]
        for pair_index in selected_pairs
    ]
    intact_vs_shuffled_fraction = (
        float(np.mean(intact_vs_shuffled)) if intact_vs_shuffled else 0.0
    )
    paired_intact_fraction = float(np.mean(paired_intact)) if paired_intact else 0.0
    return {
        "pairs": len(selected_pairs),
        "sides": len(side_indexes),
        "intact_vs_successor_shuffled_code_distinct_fraction": (
            intact_vs_shuffled_fraction
        ),
        "intact_vs_successor_shuffled_equal_codes": int(
            len(intact_vs_shuffled) - sum(intact_vs_shuffled)
        ),
        "intact_paired_code_distinct_fraction": paired_intact_fraction,
        "intact_paired_equal_codes": int(len(paired_intact) - sum(paired_intact)),
    }


def _development_support(
    pairs: Sequence[codec_assay.DevelopmentPair],
    codes: Sequence[tuple[int, ...]],
    training_support: frozenset[int],
    *,
    partition: str,
) -> dict[str, Any]:
    selected = [
        code
        for pair_index, pair in enumerate(pairs)
        if partition == "all" or pair.partition == partition
        for code in (codes[2 * pair_index], codes[2 * pair_index + 1])
    ]
    occurrences = [feature for code in selected for feature in code]
    unseen_occurrences = [
        feature for feature in occurrences if feature not in training_support
    ]
    unseen_unique = set(unseen_occurrences)
    return {
        "sides": len(selected),
        "feature_occurrences": len(occurrences),
        "unique_features": len(set(occurrences)),
        "unseen_feature_occurrences": len(unseen_occurrences),
        "unseen_feature_occurrence_fraction": len(unseen_occurrences)
        / max(len(occurrences), 1),
        "unseen_unique_features": len(unseen_unique),
        "sides_with_unseen_feature": sum(
            any(feature not in training_support for feature in code)
            for code in selected
        ),
    }


def evaluate_receiver_curve(
    cortex: codec_assay.FrozenCortex,
    pairs: Sequence[codec_assay.DevelopmentPair],
    codec: codec_assay.FutureTurnCodec,
    intact: Any,
    shuffled: Any,
    curve: ReceiverCurve,
    *,
    exposures: Sequence[int],
    gain: float,
    residual_clip: float,
    horizon: int,
    sample_seed: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Generate fixed-code receiver checkpoints with common random draws."""

    schedule = _strict_exposures(exposures)
    if tuple(curve.checkpoints) != schedule:
        raise ValueError("receiver checkpoints do not match exposure schedule")
    if len(pairs) < 2:
        raise ValueError("development evaluation requires at least two pairs")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("receiver gain must be finite and positive")
    writer_before = {
        exposure: {
            name: writer.fingerprint() for name, writer in writers.items()
        }
        for exposure, writers in curve.checkpoints.items()
    }
    intact_before = intact.fingerprint(include_state=False)
    shuffled_before = shuffled.fingerprint(include_state=False)
    flat = _flat_sides(pairs)
    oracle_codes = [
        codec.quantize(
            codec.encode(
                side.full_response,
                role=codec_assay.ASSISTANT_ROLE,
                source=pair.source_label,
            )
        )
        for pair, _, side in flat
    ]
    modes: dict[str, Any] = {}
    for mode_index, mode in enumerate(
        ("original_unanswered_prefix", "long_interruption_crossover")
    ):
        available = all(side.original_context is not None for _, _, side in flat)
        if mode == "original_unanswered_prefix" and not available:
            modes[mode] = {"available": False}
            continue
        mode_exposures = schedule if mode_index == 0 else (schedule[0], schedule[-1])
        contexts = [
            side.original_context
            if mode == "original_unanswered_prefix"
            else side.interrupted_context
            for _, _, side in flat
        ]
        if any(context is None for context in contexts):
            raise AssertionError("available context mode contains no prefix")
        concrete_contexts = [context for context in contexts if context is not None]
        intact_codes, shuffled_codes, state_hashes = _codes_for_contexts(
            pairs, codec, intact, shuffled, concrete_contexts
        )
        distinctness = {
            partition: _code_distinctness(
                pairs,
                intact_codes,
                shuffled_codes,
                partition=partition,
            )
            for partition in ("selection", "evaluation", "all")
        }
        support = {
            track: {
                partition: _development_support(
                    pairs,
                    codes,
                    curve.code_feature_support[track],
                    partition=partition,
                )
                for partition in ("selection", "evaluation", "all")
            }
            for track, codes in (
                ("oof_intact", intact_codes),
                ("oof_successor_shuffled", shuffled_codes),
            )
        }
        arm_names = _evaluation_arm_names(mode_exposures)
        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in arm_names}
        for pair_index, pair in enumerate(pairs):
            pair_outputs: dict[str, dict[str, codec_assay.GeneratedBranch]] = {
                name: {} for name in arm_names
            }
            pair_bpb: dict[str, list[float]] = {name: [] for name in arm_names}
            for side_offset, side_name in enumerate(("a", "b")):
                flat_index = 2 * pair_index + side_offset
                paired_index = 2 * pair_index + (1 - side_offset)
                side = pair.side_a if side_name == "a" else pair.side_b
                context = concrete_contexts[flat_index]
                assignments = _arm_assignments(
                    curve.checkpoints,
                    mode_exposures,
                    intact_code=intact_codes[flat_index],
                    shuffled_code=shuffled_codes[flat_index],
                    paired_intact_code=intact_codes[paired_index],
                    oracle_code=oracle_codes[flat_index],
                    paired_oracle_code=oracle_codes[paired_index],
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
                    generated = codec_assay.generate_branch(
                        cortex,
                        context,
                        assignment,
                        horizon=horizon,
                        residual_clip=residual_clip,
                        uniform_draws=draws,
                    )
                    pair_outputs[name][side_name] = generated
                    pair_bpb[name].append(
                        codec_assay.teacher_forced_bpb_from_bases(
                            bases,
                            reference,
                            assignment,
                            residual_clip=residual_clip,
                        )
                    )
            for name in arm_names:
                outputs = pair_outputs[name]
                discrimination = codec_assay._paired_reference_discrimination(
                    outputs["a"].data,
                    outputs["b"].data,
                    pair.side_a.response_window,
                    pair.side_b.response_window,
                    dimensions=1024,
                )
                rows[name].append(
                    {
                        "pair_id": pair.pair_id,
                        "partition": pair.partition,
                        "sampled_margin": discrimination[
                            "correct_minus_crossed_margin"
                        ],
                        "mean_teacher_forced_bpb": float(np.mean(pair_bpb[name])),
                        "sides": {
                            side_name: {
                                "sampled_hex": outputs[side_name].data.hex(),
                                "sampled_text": outputs[side_name].data.decode(
                                    "utf-8", errors="backslashreplace"
                                ),
                                "sampled_probability_sha256": outputs[
                                    side_name
                                ].probability_sha256,
                                "sampled_diagnostics": (
                                    codec_assay._generation_diagnostics(
                                        outputs[side_name].data
                                    )
                                ),
                            }
                            for side_name in ("a", "b")
                        },
                    }
                )
        for native, gain0 in zip(rows["native"], rows["gain0"], strict=True):
            if native != gain0:
                raise AssertionError("gain-zero receiver route is not exactly native")
        modes[mode] = {
            "available": True,
            "evaluated_exposures": list(mode_exposures),
            "predictor_state_manifest_sha256": _sha256(
                "".join(state_hashes).encode("ascii")
            ),
            "intact_code_manifest_sha256": _sha256(
                repr(intact_codes).encode("ascii")
            ),
            "successor_shuffled_code_manifest_sha256": _sha256(
                repr(shuffled_codes).encode("ascii")
            ),
            "code_distinctness_by_partition": distinctness,
            "development_code_support_against_receiver_training": support,
            "summaries": {
                partition: _summarize_curve_rows(
                    rows,
                    exposures=mode_exposures,
                    partition=partition,
                    bootstrap_seed=bootstrap_seed + mode_index * 100_000,
                    bootstrap_resamples=bootstrap_resamples,
                )
                for partition in ("selection", "evaluation", "all")
            },
            "pair_rows": rows,
            "invariants": {
                "gain0_exactly_native": True,
                "actual_bytes_generated_not_candidate_reranked": True,
                "common_random_draws_shared_across_all_arms": True,
                "codes_fixed_across_receiver_checkpoints": True,
                "observed_prefix_predictor_states_identical_across_transitions": True,
            },
        }
    writer_after = {
        exposure: {
            name: writer.fingerprint() for name, writer in writers.items()
        }
        for exposure, writers in curve.checkpoints.items()
    }
    if writer_after != writer_before:
        raise AssertionError("development generation modified a receiver")
    if intact.fingerprint(include_state=False) != intact_before:
        raise AssertionError("evaluation modified intact transition")
    if shuffled.fingerprint(include_state=False) != shuffled_before:
        raise AssertionError("evaluation modified shuffled transition")
    if intact.observed_count or shuffled.observed_count:
        raise AssertionError("evaluation retained transient recurrent state")
    return {
        "context_modes": modes,
        "writer_fingerprints_unchanged_by_evaluation": True,
        "transition_fingerprints_unchanged_by_evaluation": True,
        "transient_recurrent_states_reset_after_evaluation": True,
        "development_partition_counts": {
            partition: sum(pair.partition == partition for pair in pairs)
            for partition in ("selection", "evaluation")
        },
    }


def compare_pass1_reference_replay(
    evaluation: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the unchanged legacy pass-one arms to replay exactly."""

    mapping = {
        "native": "native",
        "gain0": "recurrent_code_gain0",
        "legacy_p1_intact": "recurrent_code_coupled",
        "legacy_p1_successor_shuffled": "successor_shuffled_code_coupled",
    }
    mismatches: list[str] = []
    compared = 0
    for mode in ("original_unanswered_prefix", "long_interruption_crossover"):
        current_mode = evaluation["context_modes"][mode]
        prior_mode = reference["evaluation"]["context_modes"][mode]
        if (
            current_mode.get("available") is not True
            or prior_mode.get("available") is not True
        ):
            mismatches.append(f"{mode}: availability mismatch")
            continue
        for current_name, prior_name in mapping.items():
            current_rows = current_mode["pair_rows"][current_name]
            prior_rows = prior_mode["pair_rows"][prior_name]
            compared += len(current_rows)
            if current_rows != prior_rows:
                mismatches.append(f"{mode}/{current_name}->{prior_name}")
    return {
        "passed": not mismatches,
        "mapped_conditions": mapping,
        "compared_pair_condition_rows": compared,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
    }


def _positive_ci(record: Mapping[str, Any]) -> bool:
    low = record.get("ci95_low")
    return (
        isinstance(low, (int, float))
        and not isinstance(low, bool)
        and math.isfinite(float(low))
        and float(low) > 0.0
    )


def compute_development_gate(
    evaluation: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    final_exposure: int,
) -> dict[str, Any]:
    """Classify the locked evaluation-partition receiver-undertraining gate."""

    mode = evaluation["context_modes"]["original_unanswered_prefix"]
    summary = mode["summaries"]["evaluation"]
    contrasts = summary["controlled_contrasts"]
    effect = contrasts[
        f"oof_matched_p{final_exposure}_intact_minus_successor_shuffled"
    ]["sampled_free_generation_margin_difference"]
    improvement = contrasts[
        f"oof_p{final_exposure}_minus_p1_paired_swap_contrast_change"
    ]["sampled_free_generation_difference_in_differences"]
    specificity = contrasts[f"oof_p{final_exposure}_intact_minus_paired_swap"][
        "sampled_free_generation_margin_difference"
    ]
    native_health = summary["arms"]["native"]["sampled_degeneration"]
    intact_health = summary["arms"][f"oof_p{final_exposure}_intact"][
        "sampled_degeneration"
    ]
    printable_loss = float(native_health["mean_printable_fraction"]) - float(
        intact_health["mean_printable_fraction"]
    )
    utf8_loss = float(native_health["valid_utf8_fraction"]) - float(
        intact_health["valid_utf8_fraction"]
    )
    added_long_runs = int(
        intact_health["outputs_with_run_at_least_16"]
    ) - int(native_health["outputs_with_run_at_least_16"])
    health_passed = (
        printable_loss <= 0.05 and utf8_loss <= 0.05 and added_long_runs <= 2
    )
    distinctness = mode["code_distinctness_by_partition"]["evaluation"]
    code_distinct = float(
        distinctness["intact_vs_successor_shuffled_code_distinct_fraction"]
    )
    paired_code_distinct = float(
        distinctness["intact_paired_code_distinct_fraction"]
    )
    components = {
        "final_oof_intact_minus_shuffled_ci_strictly_positive": _positive_ci(
            effect
        ),
        "same_writer_specificity_final_minus_pass1_ci_strictly_positive": (
            _positive_ci(improvement)
        ),
        "final_intact_minus_paired_swap_ci_strictly_positive": _positive_ci(
            specificity
        ),
        "code_distinct_fraction_at_least_0_95": code_distinct >= 0.95,
        "paired_code_distinct_fraction_at_least_0_95": (
            paired_code_distinct >= 0.95
        ),
        "output_health_passed": health_passed,
        "legacy_pass1_exact_reference_replay": replay.get("passed") is True,
    }
    transmission = (
        components["final_oof_intact_minus_shuffled_ci_strictly_positive"]
        and components["final_intact_minus_paired_swap_ci_strictly_positive"]
        and components["code_distinct_fraction_at_least_0_95"]
        and components["paired_code_distinct_fraction_at_least_0_95"]
        and components["output_health_passed"]
        and components["legacy_pass1_exact_reference_replay"]
    )
    undertraining = transmission and components[
        "same_writer_specificity_final_minus_pass1_ci_strictly_positive"
    ]
    return {
        "scope": "original_unanswered_prefix/evaluation_partition",
        "final_exposure": final_exposure,
        "primary_effect": effect,
        "same_writer_specificity_exposure_difference_in_differences": improvement,
        "code_specificity": specificity,
        "code_distinct_fraction": code_distinct,
        "paired_code_distinct_fraction": paired_code_distinct,
        "output_health": {
            "native": native_health,
            "oof_intact": intact_health,
            "printable_loss": printable_loss,
            "valid_utf8_loss": utf8_loss,
            "added_outputs_with_run_at_least_16": added_long_runs,
        },
        "components": components,
        "predicted_code_receiver_transmission_demonstrated": transmission,
        "receiver_undertraining_demonstrated": undertraining,
    }


def _read_reference(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("reference result must contain a JSON object")
    if (
        value.get("experiment")
        != "future_turn_role_stratified_successor_shuffle_development"
    ):
        raise ValueError("reference result is not the successor-shuffle assay")
    return value


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    exposures = _strict_exposures(args.exposures)
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    if args.response_cap <= 0:
        raise ValueError("response cap must be positive")
    if args.generation_horizon <= 0:
        raise ValueError("generation horizon must be positive")
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
    sequences = recurrent_assay.encode_training_sequences(conversations, codec)
    transition_started = time.perf_counter()
    intact = recurrent_assay.fit_recurrent_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    shuffled, shuffle_audit = successor_assay.fit_successor_shuffled_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
        seed=args.successor_shuffle_seed,
    )
    cross_fit = cross_fitted_training_codes(
        conversations,
        sequences,
        codec,
        folds=args.cross_fit_folds,
        fold_seed=args.cross_fit_seed,
        successor_shuffle_seed=args.cross_fit_shuffle_seed,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    transition_seconds = time.perf_counter() - transition_started
    latent = successor_assay.latent_control_diagnostic(
        pairs, codec, intact, shuffled
    )
    common: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "future_turn_predicted_code_receiver_learning_curve_development",
        "status": (
            "complete_latent_smoke"
            if args.latent_only and args.smoke
            else "complete_latent_development"
            if args.latent_only
            else "running"
        ),
        "claim_scope": {
            "zero_backpropagation": True,
            "zero_backpropagation_through_time": True,
            "native_checkpoint_frozen": True,
            "codec_fit_is_batch_pca_not_online_local": True,
            "transition_fit_is_closed_form_ridge": True,
            "receiver_uses_local_normalized_delta_rule": True,
            "whole_conversation_cross_fitted_predicted_training_codes": True,
            "natural_logs_teacher_forced_not_counterfactual_rollout": True,
            "free_generation_primary": True,
            "teacher_forced_bpb_secondary_only": True,
            "goal_or_thoughtfulness_claim": False,
            "particle_or_multihorizon_claim": False,
            "surprise_grown_transition_claim": False,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "config": {
            "raw_dimensions": args.raw_dimensions,
            "content_dimensions": args.content_dimensions,
            "active_content_features": args.active_content_features,
            "memory_decay": args.memory_decay,
            "transition_ridge": args.transition_ridge,
            "successor_shuffle_seed": args.successor_shuffle_seed,
            "cross_fit_folds": args.cross_fit_folds,
            "cross_fit_seed": args.cross_fit_seed,
            "cross_fit_shuffle_seed": args.cross_fit_shuffle_seed,
            "receiver_exposures": list(exposures),
            "table_bits": args.table_bits,
            "learning_rate": args.learning_rate,
            "local_orders": list(args.local_orders),
            "response_cap": args.response_cap,
            "generation_horizon": args.generation_horizon,
            "gain": args.gain,
            "residual_clip": args.residual_clip,
            "training_seed": args.training_seed,
            "sample_seed": args.sample_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "latent_only": args.latent_only,
            "smoke": args.smoke,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "completed_training_assistant_turns": len(assistant_turns),
            "training_role_tagged_events": sum(len(row) for row in sequences),
            "adjacent_training_event_transitions": sum(
                len(row) - 1 for row in sequences
            ),
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
        },
        "codec": {
            "fit_inputs": "completed training assistant turns only",
            "anchored_throughout_curve": True,
            "dense_content_dimensions": codec.content_dimensions,
            "sparse_persistent_features": codec.persistent_feature_count,
        },
        "transitions": {
            "intact_fingerprint": intact.fingerprint(include_state=False),
            "successor_shuffled_fingerprint": shuffled.fingerprint(
                include_state=False
            ),
            "successor_shuffle_audit": shuffle_audit,
            "cross_fit": cross_fit.audit,
            "total_fit_wall_seconds": transition_seconds,
        },
        "latent_diagnostic": latent,
    }
    if args.latent_only:
        common["wall_seconds"] = time.perf_counter() - started
        return common
    if args.checkpoint is None or args.reference_result is None:
        raise ValueError("native run requires checkpoint and reference result")
    reference = _read_reference(args.reference_result)
    cortex = codec_assay._load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("could not load native checkpoint")
    embodied_at_load = int(getattr(cortex, "embodied_bytes", -1))
    runtime = recurrent_assay.native_runtime_provenance(cortex)
    curve = fit_receiver_curve(
        cortex,
        conversations,
        codec,
        cross_fit,
        exposures=exposures,
        table_bits=args.table_bits,
        learning_rate=args.learning_rate,
        local_orders=args.local_orders,
        response_cap=args.response_cap,
        training_seed=args.training_seed,
    )
    pass1 = curve.training["per_pass"][0]
    prior_training = reference["decoder_training"]
    legacy_training_replay = {
        "conversation_order_sha256": pass1["conversation_order_sha256"]
        == prior_training["conversation_order_sha256"],
        "response_sequence_sha256": pass1["response_sequence_sha256"]
        == prior_training["observed_response_sequence_sha256"],
        "episodes": pass1["writers"]["legacy_oracle"]["episodes"]
        == prior_training["writers"]["coupled"]["episodes"],
        "symbols": pass1["writers"]["legacy_oracle"]["symbols"]
        == prior_training["writers"]["coupled"]["symbols"],
        "base_nll_nats": pass1["writers"]["legacy_oracle"]["base_nll_nats"]
        == prior_training["writers"]["coupled"]["base_nll_nats"],
        "receiver_nll_nats": pass1["writers"]["legacy_oracle"][
            "receiver_nll_nats"
        ]
        == prior_training["writers"]["coupled"]["decoder_nll_nats"],
    }
    legacy_training_replay["passed"] = all(legacy_training_replay.values())
    evaluation = evaluate_receiver_curve(
        cortex,
        pairs,
        codec,
        intact,
        shuffled,
        curve,
        exposures=exposures,
        gain=args.gain,
        residual_clip=args.residual_clip,
        horizon=args.generation_horizon,
        sample_seed=args.sample_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    replay = compare_pass1_reference_replay(evaluation, reference)
    replay["legacy_training"] = legacy_training_replay
    replay["passed"] = replay["passed"] and legacy_training_replay["passed"]
    gate = compute_development_gate(
        evaluation, replay, final_exposure=exposures[-1]
    )
    allocation = {
        name: {
            "parameter_count": curve.checkpoints[exposures[-1]][
                name
            ].parameter_count,
            "allocated_bytes": curve.checkpoints[exposures[-1]][
                name
            ].allocated_bytes,
            "raw_active_features": curve.checkpoints[exposures[-1]][
                name
            ].raw_active_feature_count(codec.persistent_feature_count),
        }
        for name in TRACK_NAMES
    }
    if len(
        {
            (row["parameter_count"], row["allocated_bytes"], row["raw_active_features"])
            for row in allocation.values()
        }
    ) != 1:
        raise AssertionError("receiver track allocation diverged")
    embodied_at_report = int(getattr(cortex, "embodied_bytes", -1))
    if embodied_at_report != embodied_at_load:
        raise AssertionError("receiver assay modified the native checkpoint state")
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached after a complete checkpoint curve")
    common.update(
        {
            "status": "complete_smoke" if args.smoke else "complete_development",
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": codec_assay._sha256_file(args.checkpoint),
                "embodied_bytes_at_load": embodied_at_load,
                "transient_embodied_bytes_at_report": embodied_at_report,
            },
            "native_runtime": runtime,
            "receiver_allocation": allocation,
            "receiver_training": curve.training,
            "reference_result": {
                "path": str(args.reference_result),
                "sha256": codec_assay._sha256_file(args.reference_result),
                "pass1_exact_replay": replay,
            },
            "evaluation": evaluation,
            "development_gate": gate,
            "wall_seconds": elapsed,
        }
    )
    return common


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--latent-only", action="store_true")
    parser.add_argument("--raw-dimensions", type=int, default=512)
    parser.add_argument("--content-dimensions", type=int, default=64)
    parser.add_argument("--active-content-features", type=int, default=16)
    parser.add_argument("--memory-decay", type=float, default=0.8)
    parser.add_argument("--transition-ridge", type=float, default=1e-3)
    parser.add_argument("--successor-shuffle-seed", type=int, default=260_935)
    parser.add_argument("--cross-fit-folds", type=int, default=5)
    parser.add_argument("--cross-fit-seed", type=int, default=260_936)
    parser.add_argument("--cross-fit-shuffle-seed", type=int, default=260_940)
    parser.add_argument("--exposures", type=int, nargs="+", default=DEFAULT_EXPOSURES)
    parser.add_argument("--table-bits", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--local-orders", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--response-cap", type=int, default=128)
    parser.add_argument("--generation-horizon", type=int, default=128)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--residual-clip", type=float, default=2.0)
    parser.add_argument("--training-seed", type=int, default=260_930)
    parser.add_argument("--sample-seed", type=int, default=260_931)
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
