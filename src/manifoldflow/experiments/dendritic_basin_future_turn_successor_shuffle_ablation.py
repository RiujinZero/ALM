"""Targeted successor-shuffle control for recurrent future-turn supply.

The locked recurrent codec assay includes an output-coordinate transition
shuffle.  That control preserves the analytical history equation, but it also
scrambles role, source, constant, and content coordinates together.  This
follow-up instead holds every observed-prefix predictor state fixed and
deranges only the observed successor *content* within target-role/source
strata.  Consequently the control has the same target identity marginals and
the same ridge design matrix as the intact recurrent transition.

The native path deliberately contains only four arms: native, gain-zero,
intact recurrent coupled, and successor-shuffled recurrent coupled.  A
``--latent-only`` mode fits and audits both transitions without loading the
native checkpoint or training a byte writer.  Only schema-4 training and
development payloads are addressable; there is no configurable split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
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
from manifoldflow.experiments import (
    dendritic_basin_future_turn_recurrent_codec_ablation as recurrent_assay,
)

CONDITION_NAMES = (
    "native",
    "recurrent_code_gain0",
    "recurrent_code_coupled",
    "successor_shuffled_code_coupled",
)

IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/"
    "dendritic_basin_future_turn_successor_shuffle_ablation.py",
    "tests/test_dendritic_basin_future_turn_successor_shuffle_ablation.py",
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
    "src/manifoldflow/experiments/dendritic_basin_phase126_three_epoch_generation.py",
)


def _readonly(value: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    return _sha256(array.tobytes())


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = root / relative
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return result


@dataclass(frozen=True)
class TransitionDesign:
    """Exact teacher-forced design used by both transition fits."""

    predictor_states: np.ndarray
    target_tokens: np.ndarray
    target_turns: tuple[EncodedTurn, ...]
    conversation_indexes: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "predictor_states", _readonly(self.predictor_states)
        )
        object.__setattr__(self, "target_tokens", _readonly(self.target_tokens))
        object.__setattr__(
            self,
            "conversation_indexes",
            _readonly(self.conversation_indexes, dtype=np.int64),
        )


@dataclass(frozen=True)
class MatchedForecast:
    intact: recurrent_assay.DenseForecast
    shuffled: recurrent_assay.DenseForecast
    predictor_state_sha256: str


def build_transition_design(
    template: FutureTurnTrajectory,
    sequences: Sequence[Sequence[EncodedTurn]],
) -> TransitionDesign:
    """Reconstruct the exact adjacent-state rows used by the locked core fit."""

    predictors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_turns: list[EncodedTurn] = []
    conversation_indexes: list[int] = []
    if not sequences:
        raise ValueError("at least one recurrent sequence is required")
    for conversation_index, raw_sequence in enumerate(sequences):
        sequence = tuple(raw_sequence)
        if len(sequence) < 2:
            raise ValueError("each recurrent sequence needs adjacent events")
        state = np.zeros(template.state_dim, dtype=np.float64)
        previous: np.ndarray | None = None
        for turn in sequence:
            template._validate_turn(turn)
            state = template._advance_observed(state, turn)
            if previous is not None:
                predictors.append(previous)
                targets.append(template._token(turn))
                target_turns.append(turn)
                conversation_indexes.append(conversation_index)
            previous = state
    return TransitionDesign(
        np.stack(predictors),
        np.stack(targets),
        tuple(target_turns),
        np.asarray(conversation_indexes, dtype=np.int64),
    )


def stratified_successor_permutation(
    target_turns: Sequence[EncodedTurn], *, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Derange donor content independently inside target source/role strata.

    A row-index derangement is insufficient when two observed successors have
    byte-identical codec vectors.  Inside each identity stratum, equal-content
    rows are therefore kept contiguous and the donor ordering is rotated by
    the largest content multiplicity.  Such a content derangement exists
    exactly when no content class occupies more than half the stratum; an
    infeasible dataset fails visibly instead of silently weakening the control.
    """

    turns = tuple(target_turns)
    if len(turns) < 2:
        raise ValueError("successor shuffle requires at least two targets")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    groups: dict[tuple[str | int, str | int], list[int]] = defaultdict(list)
    for index, turn in enumerate(turns):
        groups[(turn.source_id, turn.role_id)].append(index)
    order = sorted(
        groups,
        key=lambda item: (
            type(item[0]).__name__,
            str(item[0]),
            type(item[1]).__name__,
            str(item[1]),
        ),
    )
    permutation = np.arange(len(turns), dtype=np.int64)
    counts: dict[str, int] = {}
    for stratum_index, key in enumerate(order):
        stratum_rows = groups[key]
        if len(stratum_rows) < 2:
            raise ValueError(
                "every target source/role stratum needs at least two successors"
            )
        content_groups: dict[tuple[float, ...], list[int]] = defaultdict(list)
        for index in stratum_rows:
            content_groups[
                tuple(float(value) for value in turns[index].vector)
            ].append(index)
        maximum_multiplicity = max(map(len, content_groups.values()))
        if 2 * maximum_multiplicity > len(stratum_rows):
            raise ValueError(
                "a target identity stratum has no exact content derangement"
            )
        rng = np.random.default_rng(seed + 104_729 * stratum_index)
        content_classes = list(content_groups.values())
        class_order = rng.permutation(len(content_classes))
        ordered_rows: list[int] = []
        for class_index in class_order:
            members = np.asarray(
                content_classes[int(class_index)], dtype=np.int64
            )
            ordered_rows.extend(int(value) for value in rng.permutation(members))
        recipients = np.asarray(ordered_rows, dtype=np.int64)
        donors = np.roll(recipients, -maximum_multiplicity)
        permutation[recipients] = donors
        counts[f"{key[0]!r}|{key[1]!r}"] = len(stratum_rows)
    for index, donor in enumerate(permutation):
        recipient = turns[index]
        supplied = turns[int(donor)]
        if (
            recipient.source_id != supplied.source_id
            or recipient.role_id != supplied.role_id
        ):
            raise AssertionError("successor donor crossed a role/source stratum")
        if np.array_equal(recipient.vector, supplied.vector):
            raise AssertionError(
                "stratified successor shuffle retained target content"
            )
    if np.any(permutation == np.arange(len(turns))):
        raise AssertionError("stratified successor shuffle retained a target row")
    return permutation, counts


def shuffled_target_tokens(
    template: FutureTurnTrajectory,
    design: TransitionDesign,
    permutation: np.ndarray,
) -> np.ndarray:
    """Build tokens from donor content while retaining recipient identity."""

    donors = np.asarray(permutation, dtype=np.int64)
    if donors.shape != (len(design.target_turns),) or np.any(
        (donors < 0) | (donors >= len(design.target_turns))
    ):
        raise ValueError("successor permutation does not align with target rows")
    tokens: list[np.ndarray] = []
    for index, donor_index in enumerate(donors):
        recipient = design.target_turns[index]
        donor = design.target_turns[int(donor_index)]
        if (
            recipient.source_id != donor.source_id
            or recipient.role_id != donor.role_id
        ):
            raise ValueError("successor content donor has the wrong identity")
        tokens.append(
            template._token(
                EncodedTurn(
                    donor.vector,
                    source_id=recipient.source_id,
                    role_id=recipient.role_id,
                )
            )
        )
    return np.stack(tokens)


def _fit_token_map(
    predictors: np.ndarray, targets: np.ndarray, *, ridge: float
) -> np.ndarray:
    if ridge == 0.0:
        return np.linalg.lstsq(predictors, targets, rcond=None)[0]
    gram = predictors.T @ predictors
    regularized = gram + ridge * np.eye(predictors.shape[1])
    return np.linalg.solve(regularized, predictors.T @ targets)


def fit_successor_shuffled_transition(
    sequences: Sequence[Sequence[EncodedTurn]],
    *,
    content_dimensions: int,
    memory_decay: float,
    ridge: float,
    seed: int,
) -> tuple[FutureTurnTrajectory, dict[str, Any]]:
    """Fit a matched transition whose successor content is stratified-shuffled."""

    model = recurrent_assay.fit_recurrent_transition(
        sequences,
        content_dimensions=content_dimensions,
        memory_decay=memory_decay,
        ridge=ridge,
    )
    design = build_transition_design(model, sequences)
    permutation, strata = stratified_successor_permutation(
        design.target_turns, seed=seed
    )
    shuffled = shuffled_target_tokens(model, design, permutation)

    source_slice = model._source_slice
    role_slice = model._role_slice
    identity_columns = np.r_[
        np.arange(source_slice.start, source_slice.stop),
        np.arange(role_slice.start, role_slice.stop),
        np.asarray([model.token_dim - 1]),
    ]
    if not np.array_equal(
        design.target_tokens[:, identity_columns], shuffled[:, identity_columns]
    ):
        raise AssertionError("target role/source/constant coordinates changed")

    token_map = _fit_token_map(
        design.predictor_states, shuffled, ridge=model.ridge
    )
    model._transition = model._assemble_transition(token_map)
    model._fit_residual_variance(
        design.predictor_states,
        shuffled,
        [turn.source_id for turn in design.target_turns],
    )
    model.reset()

    equal_content = sum(
        np.array_equal(
            design.target_turns[index].vector,
            design.target_turns[int(donor)].vector,
        )
        for index, donor in enumerate(permutation)
    )
    audit = {
        "control": "role_source_stratified_successor_content_derangement",
        "rows": len(design.target_turns),
        "strata": strata,
        "seed": seed,
        "predictor_state_sha256": _array_sha256(design.predictor_states),
        "original_target_token_sha256": _array_sha256(design.target_tokens),
        "shuffled_target_token_sha256": _array_sha256(shuffled),
        "permutation_sha256": _sha256(
            np.ascontiguousarray(permutation, dtype="<i8").tobytes()
        ),
        "predictor_rows_modified": 0,
        "donor_index_fixed_points": int(
            np.sum(permutation == np.arange(len(permutation)))
        ),
        "donor_content_equalities_after_derangement": equal_content,
        "donor_content_distinct_fraction": 1.0 - equal_content / len(permutation),
        "target_role_source_constant_coordinates_exactly_preserved": True,
        "target_content_multiset_preserved_within_each_stratum": True,
    }
    return model, audit


def matched_forecast_context(
    intact: FutureTurnTrajectory,
    shuffled: FutureTurnTrajectory,
    codec: codec_assay.FutureTurnCodec,
    context: bytes,
    *,
    source_label: str,
) -> MatchedForecast:
    """Forecast both arms after asserting byte-identical observed-prefix state."""

    turns = recurrent_assay.observed_turns_from_transcript(
        context, generation_context=True
    )
    encoded = tuple(
        recurrent_assay._encoded_event(codec, turn, source_label=source_label)
        for turn in turns
    )
    intact.reset()
    shuffled.reset()
    try:
        for turn in encoded:
            intact.observe(turn)
            shuffled.observe(turn)
        left_state = intact.snapshot().state
        right_state = shuffled.snapshot().state
        if not np.array_equal(left_state, right_state):
            raise AssertionError("control changed the observed-prefix predictor state")
        left_slot = intact.predicted_tape().slots[0]
        right_slot = shuffled.predicted_tape().slots[0]
        return MatchedForecast(
            recurrent_assay.DenseForecast(
                np.array(left_slot.vector, copy=True),
                str(left_slot.role_id),
                intact.observed_count,
            ),
            recurrent_assay.DenseForecast(
                np.array(right_slot.vector, copy=True),
                str(right_slot.role_id),
                shuffled.observed_count,
            ),
            _array_sha256(left_state),
        )
    finally:
        intact.reset()
        shuffled.reset()


def _flat_sides(
    pairs: Sequence[codec_assay.DevelopmentPair],
) -> list[tuple[codec_assay.DevelopmentPair, str, codec_assay.DevelopmentSide]]:
    return [
        (pair, side_name, pair.side_a if side_name == "a" else pair.side_b)
        for pair in pairs
        for side_name in ("a", "b")
    ]


def latent_control_diagnostic(
    pairs: Sequence[codec_assay.DevelopmentPair],
    codec: codec_assay.FutureTurnCodec,
    intact: FutureTurnTrajectory,
    shuffled: FutureTurnTrajectory,
) -> dict[str, Any]:
    """Evaluate only latent forecasts; never load or run the native cortex."""

    flat = _flat_sides(pairs)
    partitions = [pair.partition for pair, _, _ in flat]
    oracle = [
        codec.encode(
            side.full_response,
            role=codec_assay.ASSISTANT_ROLE,
            source=pair.source_label,
        ).content
        for pair, _, side in flat
    ]
    modes: dict[str, dict[str, Any]] = {}
    for mode in ("original_unanswered_prefix", "long_interruption_crossover"):
        available = all(side.original_context is not None for _, _, side in flat)
        if mode == "original_unanswered_prefix" and not available:
            modes[mode] = {"available": False}
            continue
        contexts = [
            side.original_context
            if mode == "original_unanswered_prefix"
            else side.interrupted_context
            for _, _, side in flat
        ]
        matched: list[MatchedForecast] = []
        for context, (pair, _, _) in zip(contexts, flat, strict=True):
            if context is None:
                raise AssertionError("available context mode contains no prefix")
            matched.append(
                matched_forecast_context(
                    intact,
                    shuffled,
                    codec,
                    context,
                    source_label=pair.source_label,
                )
            )
        intact_forecasts = [row.intact for row in matched]
        shuffled_forecasts = [row.shuffled for row in matched]
        intact_codes = [
            recurrent_assay._quantized_code(
                codec, forecast.vector, source_label=pair.source_label
            )
            for forecast, (pair, _, _) in zip(intact_forecasts, flat, strict=True)
        ]
        shuffled_codes = [
            recurrent_assay._quantized_code(
                codec, forecast.vector, source_label=pair.source_label
            )
            for forecast, (pair, _, _) in zip(shuffled_forecasts, flat, strict=True)
        ]
        intact_cosines = [
            recurrent_assay._cosine(forecast.vector, target)
            for forecast, target in zip(intact_forecasts, oracle, strict=True)
        ]
        shuffled_cosines = [
            recurrent_assay._cosine(forecast.vector, target)
            for forecast, target in zip(shuffled_forecasts, oracle, strict=True)
        ]
        distinct = [
            left != right
            for left, right in zip(intact_codes, shuffled_codes, strict=True)
        ]
        modes[mode] = {
            "available": True,
            "predictor_states_exactly_matched": True,
            "predictor_state_sequence_sha256": _sha256(
                "".join(row.predictor_state_sha256 for row in matched).encode()
            ),
            "intact": recurrent_assay._dense_forecast_diagnostics(
                intact_forecasts, oracle, partitions
            ),
            "successor_shuffled": recurrent_assay._dense_forecast_diagnostics(
                shuffled_forecasts, oracle, partitions
            ),
            "intact_minus_successor_shuffled_mean_oracle_cosine": float(
                np.mean(np.asarray(intact_cosines) - np.asarray(shuffled_cosines))
            ),
            "quantized_code_distinct_fraction": float(np.mean(distinct)),
            "quantized_code_equal_sides": int(len(distinct) - sum(distinct)),
        }
    if intact.observed_count or shuffled.observed_count:
        raise AssertionError("latent diagnostic retained a transient prefix")
    return {"context_modes": modes, "native_cortex_loaded": False}


def _summarize_rows(
    rows: Mapping[str, list[dict[str, Any]]],
    *,
    partition: str,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    selected = {
        name: [
            row
            for row in arm
            if partition == "all" or row["partition"] == partition
        ]
        for name, arm in rows.items()
    }
    count = len(selected["native"])
    if not count:
        return {"pairs": 0, "arms": {}, "controlled_contrasts": {}}
    native = selected["native"]
    arms: dict[str, Any] = {}
    for condition_index, name in enumerate(CONDITION_NAMES):
        arm = selected[name]
        margins = [float(row["sampled_margin"]) for row in arm]
        advantages = [
            value - float(native[index]["sampled_margin"])
            for index, value in enumerate(margins)
        ]
        diagnostics = [
            row["sides"][side]["sampled_diagnostics"]
            for row in arm
            for side in ("a", "b")
        ]
        arms[name] = {
            "sampled_free_generation_margin": codec_assay._mean_ci(
                margins,
                seed=bootstrap_seed + condition_index * 31,
                resamples=bootstrap_resamples,
            ),
            "sampled_advantage_over_native": codec_assay._mean_ci(
                advantages,
                seed=bootstrap_seed + condition_index * 31 + 1,
                resamples=bootstrap_resamples,
            ),
            "mean_teacher_forced_bpb_secondary": float(
                np.mean([row["mean_teacher_forced_bpb"] for row in arm])
            ),
            "sampled_degeneration": {
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
                    sum(
                        row["maximum_identical_byte_run"] >= 16
                        for row in diagnostics
                    )
                ),
            },
        }

    contrasts: dict[str, Any] = {}
    for index, (label, left_name, right_name) in enumerate(
        (
            (
                "recurrent_coupled_minus_native",
                "recurrent_code_coupled",
                "native",
            ),
            (
                "recurrent_coupled_minus_successor_shuffled",
                "recurrent_code_coupled",
                "successor_shuffled_code_coupled",
            ),
        )
    ):
        left = selected[left_name]
        right = selected[right_name]
        contrasts[label] = {
            "sampled_free_generation_margin_difference": codec_assay._mean_ci(
                [
                    float(a["sampled_margin"]) - float(b["sampled_margin"])
                    for a, b in zip(left, right, strict=True)
                ],
                seed=bootstrap_seed + 10_000 + index * 37,
                resamples=bootstrap_resamples,
            ),
            "teacher_forced_bpb_advantage_secondary": codec_assay._mean_ci(
                [
                    float(b["mean_teacher_forced_bpb"])
                    - float(a["mean_teacher_forced_bpb"])
                    for a, b in zip(left, right, strict=True)
                ],
                seed=bootstrap_seed + 10_001 + index * 37,
                resamples=bootstrap_resamples,
            ),
        }
    return {"pairs": count, "arms": arms, "controlled_contrasts": contrasts}


def evaluate_native_control(
    cortex: codec_assay.FrozenCortex,
    pairs: Sequence[codec_assay.DevelopmentPair],
    codec: codec_assay.FutureTurnCodec,
    intact: FutureTurnTrajectory,
    shuffled: FutureTurnTrajectory,
    bank: codec_assay.DecoderBank,
    *,
    gain: float,
    residual_clip: float,
    horizon: int,
    sample_seed: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Generate only the minimum sampled arms for the successor control."""

    if len(pairs) < 2:
        raise ValueError("development evaluation requires at least two pairs")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("decoder gain must be finite and positive")
    writer_before = {name: writer.fingerprint() for name, writer in bank.items()}
    intact_before = intact.fingerprint(include_state=False)
    shuffled_before = shuffled.fingerprint(include_state=False)
    flat = _flat_sides(pairs)
    modes: dict[str, dict[str, Any]] = {}
    for mode in ("original_unanswered_prefix", "long_interruption_crossover"):
        available = all(side.original_context is not None for _, _, side in flat)
        if mode == "original_unanswered_prefix" and not available:
            modes[mode] = {"available": False}
            continue
        contexts = [
            side.original_context
            if mode == "original_unanswered_prefix"
            else side.interrupted_context
            for _, _, side in flat
        ]
        matched: list[MatchedForecast] = []
        for context, (pair, _, _) in zip(contexts, flat, strict=True):
            if context is None:
                raise AssertionError("available context mode contains no prefix")
            matched.append(
                matched_forecast_context(
                    intact,
                    shuffled,
                    codec,
                    context,
                    source_label=pair.source_label,
                )
            )
        intact_codes = [
            recurrent_assay._quantized_code(
                codec, row.intact.vector, source_label=pair.source_label
            )
            for row, (pair, _, _) in zip(matched, flat, strict=True)
        ]
        shuffled_codes = [
            recurrent_assay._quantized_code(
                codec, row.shuffled.vector, source_label=pair.source_label
            )
            for row, (pair, _, _) in zip(matched, flat, strict=True)
        ]
        distinct = [
            left != right
            for left, right in zip(intact_codes, shuffled_codes, strict=True)
        ]
        rows: dict[str, list[dict[str, Any]]] = {
            name: [] for name in CONDITION_NAMES
        }
        for pair_index, pair in enumerate(pairs):
            pair_outputs: dict[str, dict[str, codec_assay.GeneratedBranch]] = {
                name: {} for name in CONDITION_NAMES
            }
            pair_bpb: dict[str, list[float]] = {
                name: [] for name in CONDITION_NAMES
            }
            for side_offset, side_name in enumerate(("a", "b")):
                flat_index = 2 * pair_index + side_offset
                side = pair.side_a if side_name == "a" else pair.side_b
                context = contexts[flat_index]
                if context is None:
                    raise AssertionError("missing context reached generation")
                assignments = {
                    "native": codec_assay.RouteAssignment(None, (), 0.0),
                    "recurrent_code_gain0": codec_assay.RouteAssignment(
                        bank.coupled, intact_codes[flat_index], 0.0
                    ),
                    "recurrent_code_coupled": codec_assay.RouteAssignment(
                        bank.coupled, intact_codes[flat_index], gain
                    ),
                    "successor_shuffled_code_coupled": codec_assay.RouteAssignment(
                        bank.coupled, shuffled_codes[flat_index], gain
                    ),
                }
                draws = codec_assay._seeded_draws(
                    f"{mode}|{pair.pair_id}|{side_name}",
                    seed=sample_seed,
                    horizon=horizon,
                )
                reference = side.response_window
                bases = codec_assay._teacher_forced_native_bases(
                    cortex, context, reference
                )
                for name in CONDITION_NAMES:
                    assignment = assignments[name]
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
            for name in CONDITION_NAMES:
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
        for native_row, gain0_row in zip(
            rows["native"], rows["recurrent_code_gain0"], strict=True
        ):
            if native_row != gain0_row:
                raise AssertionError("gain-zero recurrent route is not exactly native")
        modes[mode] = {
            "available": True,
            "predictor_states_exactly_matched": True,
            "intact_vs_successor_shuffled_quantized_code_distinct_fraction": float(
                np.mean(distinct)
            ),
            "intact_vs_successor_shuffled_quantized_code_equal_sides": int(
                len(distinct) - sum(distinct)
            ),
            "summaries": {
                partition: _summarize_rows(
                    rows,
                    partition=partition,
                    bootstrap_seed=bootstrap_seed
                    + (0 if mode == "original_unanswered_prefix" else 100_000),
                    bootstrap_resamples=bootstrap_resamples,
                )
                for partition in ("selection", "evaluation", "all")
            },
            "pair_rows": rows,
            "invariants": {
                "gain0_recurrent_exactly_native": True,
                "actual_bytes_generated_not_candidate_reranked": True,
                "common_random_draws_shared_across_arms": True,
                "observed_prefix_predictor_states_identical_across_transitions": True,
                "control_is_target_shuffle_not_output_coordinate_shuffle": True,
            },
        }
    if {name: writer.fingerprint() for name, writer in bank.items()} != writer_before:
        raise AssertionError("development generation modified a decoder writer")
    if intact.fingerprint(include_state=False) != intact_before:
        raise AssertionError("evaluation modified the intact transition")
    if shuffled.fingerprint(include_state=False) != shuffled_before:
        raise AssertionError("evaluation modified the shuffled transition")
    if intact.observed_count or shuffled.observed_count:
        raise AssertionError("evaluation retained transient recurrent state")
    return {
        "context_modes": modes,
        "writer_fingerprints_unchanged_by_evaluation": True,
        "transition_fingerprints_unchanged_by_evaluation": True,
        "transient_recurrent_states_reset_after_evaluation": True,
        "development_partition_counts": {
            name: sum(pair.partition == name for pair in pairs)
            for name in ("selection", "evaluation")
        },
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
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
    sequences = recurrent_assay.encode_training_sequences(conversations, codec)
    transition_started = time.perf_counter()
    intact = recurrent_assay.fit_recurrent_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    shuffled, shuffle_audit = fit_successor_shuffled_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
        seed=args.successor_shuffle_seed,
    )
    transition_seconds = time.perf_counter() - transition_started
    reference_design = build_transition_design(intact, sequences)
    if shuffle_audit["predictor_state_sha256"] != _array_sha256(
        reference_design.predictor_states
    ):
        raise AssertionError("intact and control fits used different predictor states")
    latent = latent_control_diagnostic(pairs, codec, intact, shuffled)

    common = {
        "schema_version": 1,
        "experiment": "future_turn_role_stratified_successor_shuffle_development",
        "status": (
            "complete_latent_smoke"
            if args.latent_only and args.smoke
            else "complete_latent_development"
            if args.latent_only
            else "complete_smoke"
            if args.smoke
            else "complete_development"
        ),
        "claim_scope": {
            "isolated_follow_up_does_not_modify_locked_recurrent_runner": True,
            "control_preserves_exact_teacher_forced_predictor_states": True,
            "control_deranges_successor_content_within_target_role_source": True,
            "target_role_source_constant_coordinates_preserved": True,
            "deterministic_one_step_conditional_mean_only": True,
            "particle_or_multihorizon_claim": False,
            "goal_or_preference_claim": False,
            "free_generation_primary_when_native_enabled": True,
            "teacher_forced_bpb_secondary_only": True,
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
            "anchored_during_both_transition_fits": True,
            "dense_content_dimensions": codec.content_dimensions,
            "sparse_persistent_features": codec.persistent_feature_count,
        },
        "transitions": {
            "intact_fingerprint": intact.fingerprint(include_state=False),
            "successor_shuffled_fingerprint": shuffled.fingerprint(
                include_state=False
            ),
            "state_dimensions": intact.state_dim,
            "token_dimensions": intact.token_dim,
            "fit_wall_seconds": transition_seconds,
            "successor_shuffle_audit": shuffle_audit,
        },
        "latent_diagnostic": latent,
    }
    if args.latent_only:
        common["wall_seconds"] = time.perf_counter() - started
        return common

    if args.checkpoint is None:
        raise ValueError("a native checkpoint is required unless --latent-only")
    cortex = codec_assay._load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("could not load the native checkpoint")
    embodied_at_load = int(getattr(cortex, "embodied_bytes", -1))
    native_runtime = recurrent_assay.native_runtime_provenance(cortex)
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
    evaluation = evaluate_native_control(
        cortex,
        pairs,
        codec,
        intact,
        shuffled,
        bank,
        gain=args.gain,
        residual_clip=args.residual_clip,
        horizon=args.generation_horizon,
        sample_seed=args.sample_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached after completed native evaluation")
    common.update(
        {
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": codec_assay._sha256_file(args.checkpoint),
                "embodied_bytes_at_load": embodied_at_load,
                "transient_embodied_bytes_at_report": int(
                    getattr(cortex, "embodied_bytes", -1)
                ),
            },
            "native_runtime": native_runtime,
            "decoder_allocation": codec_assay.decoder_allocation_audit(
                bank, codec.persistent_feature_count
            ),
            "decoder_training": training,
            "evaluation": evaluation,
            "wall_seconds": elapsed,
        }
    )
    return common


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--latent-only", action="store_true")
    parser.add_argument("--raw-dimensions", type=int, default=512)
    parser.add_argument("--content-dimensions", type=int, default=64)
    parser.add_argument("--active-content-features", type=int, default=16)
    parser.add_argument("--memory-decay", type=float, default=0.8)
    parser.add_argument("--transition-ridge", type=float, default=1e-3)
    parser.add_argument("--successor-shuffle-seed", type=int, default=260_935)
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
