"""Rerank a shared frozen-native best-of-N bank with a persistent goal state.

This is the generation-side follow-up to the whole-response trajectory
reranker.  For every crossover pair, a frozen native cortex samples one bank
of complete candidate responses from side A.  The bank is then scored under
both sides of the pair.  The construction is intentionally stronger than
generating one branch per condition:

* all conditions choose from the exact same candidate bytes;
* native mean log probability is the goal-off best-of-N score;
* the native trace is replayed under side B and must be bit-identical;
* only the inferred goal and/or the frozen action-successor map changes.

The independent lexical readout compares selected bytes with the two held-out
real responses.  A second readout compares the selected action's predicted
successor with the strictly-later natural future target.  Neither readout
changes after training.  Candidate-bank ceilings and byte-level diagnostics
make the crucial limitation explicit: a selector cannot recover useful text
that the frozen generator never placed in its bank.

The experiment only accepts ``successor`` or ``interlocutor`` targets, freezes
all native inference adaptation, never learns during prompt replay or
generation, selects gain on development pairs, and evaluates a disjoint test
slice exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    CrossoverPair,
    _load_native,
    _native_trace,
    load_crossover_pairs,
    load_training_examples,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    _hashed_byte_ngram_vector,
    _sample_with_uniform,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    DEFAULT_DECAYS,
    DEFAULT_SUCCESSOR_GAMMAS,
    RidgeMap,
    hashed_text_vector,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation import (  # noqa: E501
    FittedReranker,
    PairVectors,
    _cosine_rows,
    _pair_goal_predictions,
    _pair_vectors,
    _predict_clamped,
    _sattolo_permutation,
    action_features,
    fit_reranker,
    summarize,
    validate_dataset_semantics,
)

CONDITION_NAMES = (
    "goal_off",
    "oracle_future_target",
    "causal_full_history",
    "causal_pending_user_only",
    "swapped_oracle",
    "swapped_causal_full_history",
    "shuffled_predictor_target",
    "shuffled_action_target_mapping",
    "shuffled_candidate_action_mapping",
    "shuffled_episode_goal",
    "candidate_blind_action",
)


@dataclass(frozen=True)
class GeneratedCandidate:
    """One native sample plus frozen representations and audit evidence."""

    data: bytes
    native_mean_log2_a: float
    native_mean_log2_b: float
    action_vector: np.ndarray
    lexical_vector: np.ndarray
    diagnostics: dict[str, Any]
    sample_seed: int
    uniform_draw_sha256: str
    native_trace_a_equals_b: bool
    origin: str = "native_sample"


@dataclass(frozen=True)
class SharedCandidateBank:
    pair_id: str
    candidates: tuple[GeneratedCandidate, ...]


@dataclass(frozen=True)
class PreparedSplit:
    raw_pairs: tuple[CrossoverPair, ...]
    pairs: tuple[PairVectors, ...]
    banks: tuple[SharedCandidateBank, ...]
    predictions: dict[str, np.ndarray]
    shuffled_episode_goals: np.ndarray


def _stable_sample_seed(
    sampling_seed: int, split: str, pair_id: str, candidate_index: int
) -> int:
    """Derive a process-independent uint64 seed from explicit identifiers."""

    payload = (f"{sampling_seed}\x00{split}\x00{pair_id}\x00{candidate_index}").encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"alm-bank").digest()
    return int.from_bytes(digest, "little", signed=False)


def _generate_with_trace(
    cortex: object,
    prompt: bytes,
    uniform_draws: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    stop_at_story_boundary: bool,
) -> tuple[bytes, np.ndarray]:
    """Generate one read-only candidate and retain its exact native trace."""

    draws = np.asarray(uniform_draws, dtype=np.float64)
    if draws.ndim != 1 or not len(draws):
        raise ValueError("uniform_draws must be a nonempty vector")
    cortex.reset_episode()
    if prompt:
        cortex.observe_chunk_probabilities(
            np.frombuffer(prompt, dtype=np.uint8), learn=False
        )
    boundary = getattr(cortex, "story_boundary", b"<|endoftext|>\n")
    if not isinstance(boundary, bytes):
        boundary = bytes(boundary)
    generated = bytearray()
    trace = np.empty((len(draws), 256), dtype=np.float64)
    used = 0
    for draw in draws:
        distribution = np.asarray(
            cortex.predict_next_distribution(), dtype=np.float64
        ).reshape(-1)
        if distribution.shape != (256,):
            raise ValueError("native cortex distribution must have shape (256,)")
        trace[used] = distribution
        byte_id = _sample_with_uniform(
            distribution,
            float(draw),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(byte_id)
        cortex.advance_observation(byte_id, learn=False)
        used += 1
        if stop_at_story_boundary and generated.endswith(boundary):
            break
    return bytes(generated), trace[:used].copy()


def _mean_log2_from_trace(trace: np.ndarray, candidate: bytes) -> float:
    targets = np.frombuffer(candidate, dtype=np.uint8)
    if trace.shape != (len(targets), 256) or not len(targets):
        raise ValueError("trace must align with a nonempty candidate")
    probability = trace[np.arange(len(targets)), targets]
    return float(np.mean(np.log2(np.maximum(probability, 1e-300))))


def build_shared_candidate_bank(
    cortex: object,
    pair: CrossoverPair,
    *,
    split: str,
    bank_size: int,
    max_new_bytes: int,
    temperature: float,
    top_k: int,
    top_p: float,
    sampling_seed: int,
    feature_dimensions: int,
    lexical_dimensions: int,
    stop_at_story_boundary: bool,
) -> SharedCandidateBank:
    """Sample once from A and verify the same trace under B for every item."""

    if bank_size < 2:
        raise ValueError("bank_size must be at least two")
    boundary = getattr(cortex, "story_boundary", b"<|endoftext|>\n")
    if not isinstance(boundary, bytes):
        boundary = bytes(boundary)
    candidates = []
    for candidate_index in range(bank_size):
        sample_seed = _stable_sample_seed(
            sampling_seed, split, pair.pair_id, candidate_index
        )
        draws = np.random.default_rng(sample_seed).random(max_new_bytes)
        data, trace_a = _generate_with_trace(
            cortex,
            pair.context_a,
            draws,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_at_story_boundary=stop_at_story_boundary,
        )
        trace_b = _native_trace(cortex, pair.context_b, data)
        score_a = _mean_log2_from_trace(trace_a, data)
        score_b = _mean_log2_from_trace(trace_b, data)
        candidates.append(
            GeneratedCandidate(
                data=data,
                native_mean_log2_a=score_a,
                native_mean_log2_b=score_b,
                action_vector=hashed_text_vector(
                    data.decode("utf-8", errors="replace"), feature_dimensions
                ),
                lexical_vector=_hashed_byte_ngram_vector(data, lexical_dimensions),
                diagnostics=_generation_diagnostics(data, boundary),
                sample_seed=sample_seed,
                uniform_draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),
                native_trace_a_equals_b=bool(np.array_equal(trace_a, trace_b)),
            )
        )
    return SharedCandidateBank(pair.pair_id, tuple(candidates))


def _teacher_forced_candidate(
    cortex: object,
    pair: CrossoverPair,
    data: bytes,
    *,
    origin: str,
    feature_dimensions: int,
    lexical_dimensions: int,
) -> GeneratedCandidate:
    """Represent and score a diagnostic candidate without learning from it."""

    if not data:
        raise ValueError("a diagnostic candidate cannot be empty")
    trace_a = _native_trace(cortex, pair.context_a, data)
    trace_b = _native_trace(cortex, pair.context_b, data)
    boundary = getattr(cortex, "story_boundary", b"<|endoftext|>\n")
    if not isinstance(boundary, bytes):
        boundary = bytes(boundary)
    return GeneratedCandidate(
        data=data,
        native_mean_log2_a=_mean_log2_from_trace(trace_a, data),
        native_mean_log2_b=_mean_log2_from_trace(trace_b, data),
        action_vector=hashed_text_vector(
            data.decode("utf-8", errors="replace"), feature_dimensions
        ),
        lexical_vector=_hashed_byte_ngram_vector(data, lexical_dimensions),
        diagnostics=_generation_diagnostics(data, boundary),
        sample_seed=-1,
        uniform_draw_sha256="",
        native_trace_a_equals_b=bool(np.array_equal(trace_a, trace_b)),
        origin=origin,
    )


def build_reference_injected_bank(
    cortex: object,
    pair: CrossoverPair,
    generated_bank: SharedCandidateBank,
    *,
    feature_dimensions: int,
    lexical_dimensions: int,
) -> SharedCandidateBank:
    """Append both matched responses as an explicitly offline diagnostic pool."""

    reference_a = _teacher_forced_candidate(
        cortex,
        pair,
        pair.response_a,
        origin="injected_held_out_response_a",
        feature_dimensions=feature_dimensions,
        lexical_dimensions=lexical_dimensions,
    )
    reference_b = _teacher_forced_candidate(
        cortex,
        pair,
        pair.response_b,
        origin="injected_held_out_response_b",
        feature_dimensions=feature_dimensions,
        lexical_dimensions=lexical_dimensions,
    )
    return SharedCandidateBank(
        pair.pair_id, generated_bank.candidates + (reference_a, reference_b)
    )


def _candidate_alignment(
    model: FittedReranker,
    pending: np.ndarray,
    candidates: np.ndarray,
    goal: np.ndarray,
    *,
    action_model: RidgeMap | None = None,
    candidate_blind: bool = False,
    interaction: bool,
) -> np.ndarray:
    count = len(candidates)
    pending_rows = np.repeat(np.asarray(pending)[None, :], count, axis=0)
    if candidate_blind:
        successor = _predict_clamped(model.candidate_blind_successor, pending_rows)
    else:
        selected = model.action_successor if action_model is None else action_model
        successor = _predict_clamped(
            selected,
            action_features(pending_rows, candidates, interaction=interaction),
        )
    goals = np.repeat(np.asarray(goal)[None, :], count, axis=0)
    return _cosine_rows(successor, goals)


def _condition_specifications(
    pair_index: int,
    pair: PairVectors,
    prepared: PreparedSplit,
    model: FittedReranker,
) -> dict[str, tuple[np.ndarray, np.ndarray, RidgeMap | None, bool]]:
    offset = 2 * pair_index
    full = prepared.predictions["causal_full_history"]
    pending = prepared.predictions["causal_pending_user_only"]
    shuffled = prepared.predictions["shuffled_predictor_target"]
    shuffled_episode = prepared.shuffled_episode_goals
    default = None
    return {
        "oracle_future_target": (pair.target_a, pair.target_b, default, False),
        "causal_full_history": (full[offset], full[offset + 1], default, False),
        "causal_pending_user_only": (
            pending[offset],
            pending[offset + 1],
            default,
            False,
        ),
        "swapped_oracle": (pair.target_b, pair.target_a, default, False),
        "swapped_causal_full_history": (
            full[offset + 1],
            full[offset],
            default,
            False,
        ),
        "shuffled_predictor_target": (
            shuffled[offset],
            shuffled[offset + 1],
            default,
            False,
        ),
        "shuffled_action_target_mapping": (
            full[offset],
            full[offset + 1],
            model.shuffled_action_successor,
            False,
        ),
        "shuffled_candidate_action_mapping": (
            full[offset],
            full[offset + 1],
            model.shuffled_candidate_action_successor,
            False,
        ),
        "shuffled_episode_goal": (
            shuffled_episode[offset],
            shuffled_episode[offset + 1],
            default,
            False,
        ),
        "candidate_blind_action": (
            full[offset],
            full[offset + 1],
            default,
            True,
        ),
    }


def select_from_bank(
    pair_index: int,
    prepared: PreparedSplit,
    model: FittedReranker,
    *,
    gain: float,
    interaction: bool,
) -> dict[str, dict[str, Any]]:
    """Return deterministic A/B selections for every intervention."""

    pair = prepared.pairs[pair_index]
    bank = prepared.banks[pair_index]
    candidates = np.asarray([row.action_vector for row in bank.candidates])
    base_a = np.asarray([row.native_mean_log2_a for row in bank.candidates])
    base_b = np.asarray([row.native_mean_log2_b for row in bank.candidates])
    base_index_a = int(np.argmax(base_a))
    base_index_b = int(np.argmax(base_b))
    result: dict[str, dict[str, Any]] = {
        "goal_off": {
            "selected_index_a": base_index_a,
            "selected_index_b": base_index_b,
            "selected_score_a": float(base_a[base_index_a]),
            "selected_score_b": float(base_b[base_index_b]),
            "selected_alignment_a": 0.0,
            "selected_alignment_b": 0.0,
        }
    }
    specifications = _condition_specifications(pair_index, pair, prepared, model)
    for name, (goal_a, goal_b, action_model, candidate_blind) in specifications.items():
        alignment_a = _candidate_alignment(
            model,
            pair.pending_a,
            candidates,
            goal_a,
            action_model=action_model,
            candidate_blind=candidate_blind,
            interaction=interaction,
        )
        alignment_b = _candidate_alignment(
            model,
            pair.pending_b,
            candidates,
            goal_b,
            action_model=action_model,
            candidate_blind=candidate_blind,
            interaction=interaction,
        )
        score_a = base_a + gain * alignment_a
        score_b = base_b + gain * alignment_b
        index_a = int(np.argmax(score_a))
        index_b = int(np.argmax(score_b))
        result[name] = {
            "selected_index_a": index_a,
            "selected_index_b": index_b,
            "selected_score_a": float(score_a[index_a]),
            "selected_score_b": float(score_b[index_b]),
            "selected_alignment_a": float(alignment_a[index_a]),
            "selected_alignment_b": float(alignment_b[index_b]),
            "alignment_range_a": float(np.ptp(alignment_a)),
            "alignment_range_b": float(np.ptp(alignment_b)),
        }
    return result


def _byte_difference_fraction(left: bytes, right: bytes) -> float:
    width = max(len(left), len(right))
    if width == 0:
        return 0.0
    mismatch = abs(len(left) - len(right))
    mismatch += sum(a != b for a, b in zip(left, right, strict=False))
    return mismatch / width


def _bank_readouts(
    raw_pair: CrossoverPair,
    pair: PairVectors,
    bank: SharedCandidateBank,
    model: FittedReranker,
    *,
    lexical_dimensions: int,
    interaction: bool,
) -> dict[str, Any]:
    candidate_action = np.asarray([row.action_vector for row in bank.candidates])
    candidate_lexical = np.asarray([row.lexical_vector for row in bank.candidates])
    references = np.asarray(
        [
            _hashed_byte_ngram_vector(raw_pair.response_a, lexical_dimensions),
            _hashed_byte_ngram_vector(raw_pair.response_b, lexical_dimensions),
        ]
    )
    reference_similarity = candidate_lexical @ references.T
    count = len(bank.candidates)
    pending_a = np.repeat(pair.pending_a[None, :], count, axis=0)
    pending_b = np.repeat(pair.pending_b[None, :], count, axis=0)
    successor_a = _predict_clamped(
        model.action_successor,
        action_features(pending_a, candidate_action, interaction=interaction),
    )
    successor_b = _predict_clamped(
        model.action_successor,
        action_features(pending_b, candidate_action, interaction=interaction),
    )
    targets = np.asarray([pair.target_a, pair.target_b])
    future_a = _cosine_rows(
        np.repeat(successor_a[:, None, :], 2, axis=1).reshape(-1, targets.shape[1]),
        np.tile(targets, (count, 1)),
    ).reshape(count, 2)
    future_b = _cosine_rows(
        np.repeat(successor_b[:, None, :], 2, axis=1).reshape(-1, targets.shape[1]),
        np.tile(targets, (count, 1)),
    ).reshape(count, 2)
    return {
        "reference_similarity": reference_similarity,
        "future_similarity_a": future_a,
        "future_similarity_b": future_b,
    }


def _selection_readout(
    selected: dict[str, Any], readouts: dict[str, Any]
) -> dict[str, float | int | bool]:
    index_a = int(selected["selected_index_a"])
    index_b = int(selected["selected_index_b"])
    reference = readouts["reference_similarity"]
    future_a = readouts["future_similarity_a"]
    future_b = readouts["future_similarity_b"]
    ref_aa = float(reference[index_a, 0])
    ref_ab = float(reference[index_a, 1])
    ref_ba = float(reference[index_b, 0])
    ref_bb = float(reference[index_b, 1])
    future_aa = float(future_a[index_a, 0])
    future_ab = float(future_a[index_a, 1])
    future_ba = float(future_b[index_b, 0])
    future_bb = float(future_b[index_b, 1])
    return {
        "selected_index_a": index_a,
        "selected_index_b": index_b,
        "selection_differs_between_sides": index_a != index_b,
        "matched_reference_mean_cosine": 0.5 * (ref_aa + ref_bb),
        "crossed_reference_mean_cosine": 0.5 * (ref_ab + ref_ba),
        "reference_symmetric_pair_margin": (ref_aa - ref_ab) + (ref_bb - ref_ba),
        "matched_future_mean_alignment": 0.5 * (future_aa + future_bb),
        "crossed_future_mean_alignment": 0.5 * (future_ab + future_ba),
        "future_symmetric_pair_margin": (future_aa - future_ab)
        + (future_bb - future_ba),
    }


def _injected_reference_indices(
    bank: SharedCandidateBank,
) -> tuple[int, int] | None:
    origins = [row.origin for row in bank.candidates]
    label_a = "injected_held_out_response_a"
    label_b = "injected_held_out_response_b"
    if label_a not in origins and label_b not in origins:
        return None
    if origins.count(label_a) != 1 or origins.count(label_b) != 1:
        raise ValueError(
            "diagnostic pool must contain exactly one A and one B response"
        )
    return origins.index(label_a), origins.index(label_b)


def _bank_quality(
    raw_pair: CrossoverPair,
    bank: SharedCandidateBank,
    readouts: dict[str, Any],
    *,
    min_printable_fraction: float,
    useful_reference_cosine: float,
) -> dict[str, Any]:
    candidates = bank.candidates
    reference = readouts["reference_similarity"]
    future_a = readouts["future_similarity_a"]
    future_b = readouts["future_similarity_b"]
    ref_index_a = int(np.argmax(reference[:, 0] - reference[:, 1]))
    ref_index_b = int(np.argmax(reference[:, 1] - reference[:, 0]))
    future_index_a = int(np.argmax(future_a[:, 0] - future_a[:, 1]))
    future_index_b = int(np.argmax(future_b[:, 1] - future_b[:, 0]))
    readable = np.asarray(
        [
            bool(row.diagnostics["valid_utf8"])
            and float(row.diagnostics["printable_ascii_fraction"])
            >= min_printable_fraction
            for row in candidates
        ]
    )
    useful_a = readable & (reference[:, 0] >= useful_reference_cosine)
    useful_b = readable & (reference[:, 1] >= useful_reference_cosine)
    pairwise = list(combinations(range(len(candidates)), 2))
    byte_difference = [
        _byte_difference_fraction(candidates[a].data, candidates[b].data)
        for a, b in pairwise
    ]
    lexical_cosine = [
        float(candidates[a].lexical_vector @ candidates[b].lexical_vector)
        for a, b in pairwise
    ]
    unique = len({row.data for row in candidates})
    reference_ceiling = _selection_readout(
        {"selected_index_a": ref_index_a, "selected_index_b": ref_index_b},
        readouts,
    )
    future_ceiling = _selection_readout(
        {
            "selected_index_a": future_index_a,
            "selected_index_b": future_index_b,
        },
        readouts,
    )
    return {
        "candidate_count": len(candidates),
        "unique_candidate_count": unique,
        "unique_candidate_fraction": unique / len(candidates),
        "mean_pairwise_byte_difference_fraction": float(np.mean(byte_difference)),
        "mean_pairwise_lexical_cosine": float(np.mean(lexical_cosine)),
        "all_native_context_traces_bit_identical": all(
            row.native_trace_a_equals_b for row in candidates
        ),
        "max_abs_native_mean_log2_a_minus_b": float(
            max(
                abs(row.native_mean_log2_a - row.native_mean_log2_b)
                for row in candidates
            )
        ),
        "valid_utf8_fraction": float(
            np.mean([row.diagnostics["valid_utf8"] for row in candidates])
        ),
        "mean_printable_ascii_fraction": float(
            np.mean([row.diagnostics["printable_ascii_fraction"] for row in candidates])
        ),
        "mean_distinct_4gram_fraction": float(
            np.mean(
                [row.diagnostics["distinct_ngram_fraction"]["4"] for row in candidates]
            )
        ),
        "exact_held_out_response_present_a": any(
            row.data == raw_pair.response_a for row in candidates
        ),
        "exact_held_out_response_present_b": any(
            row.data == raw_pair.response_b for row in candidates
        ),
        "reference_proxy_useful_candidate_present_a": bool(np.any(useful_a)),
        "reference_proxy_useful_candidate_present_b": bool(np.any(useful_b)),
        "best_available_reference_pairing_index_a": ref_index_a,
        "best_available_reference_pairing_index_b": ref_index_b,
        "best_available_future_pairing_index_a": future_index_a,
        "best_available_future_pairing_index_b": future_index_b,
        "reference_oracle_ceiling": reference_ceiling,
        "future_oracle_ceiling": future_ceiling,
    }


def _prepare_split(
    raw_pairs: list[CrossoverPair],
    banks: list[SharedCandidateBank],
    model: FittedReranker,
    *,
    seed: int,
) -> PreparedSplit:
    pairs = tuple(_pair_vectors(raw_pairs, model))
    predictions = _pair_goal_predictions(raw_pairs, model)
    rng = np.random.default_rng(seed + 8_291)
    permutation = _sattolo_permutation(len(raw_pairs) * 2, rng)
    shuffled_episode = predictions["causal_full_history"][permutation]
    return PreparedSplit(
        tuple(raw_pairs), pairs, tuple(banks), predictions, shuffled_episode
    )


def _summarize_condition(
    rows: list[dict[str, Any]],
    goal_off: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    numeric = (
        "matched_reference_mean_cosine",
        "reference_symmetric_pair_margin",
        "matched_future_mean_alignment",
        "future_symmetric_pair_margin",
    )
    report: dict[str, Any] = {}
    for index, name in enumerate(numeric):
        values = np.asarray([float(row[name]) for row in rows])
        report[name] = summarize(values, seed=seed + 103 * index, resamples=resamples)
        baseline = np.asarray([float(row[name]) for row in goal_off])
        report[f"{name}_delta_vs_goal_off"] = summarize(
            values - baseline,
            seed=seed + 103 * index + 41,
            resamples=resamples,
        )
    report["selection_differs_between_sides_fraction"] = float(
        np.mean([row["selection_differs_between_sides"] for row in rows])
    )
    report["selected_side_fraction_changed_vs_goal_off"] = float(
        np.mean(
            [
                int(row["selected_index_a"] != base["selected_index_a"])
                for row, base in zip(rows, goal_off, strict=True)
            ]
            + [
                int(row["selected_index_b"] != base["selected_index_b"])
                for row, base in zip(rows, goal_off, strict=True)
            ]
        )
    )
    if "available_matched_response_recovery" in rows[0]:
        report["available_matched_response_recovery"] = summarize(
            np.asarray(
                [float(row["available_matched_response_recovery"]) for row in rows]
            ),
            seed=seed + 701,
            resamples=resamples,
        )
        report["pair_fraction_recovering_both_matched_responses"] = float(
            np.mean([row["recovered_both_matched_responses"] for row in rows])
        )
    return report


def evaluate_prepared_split(
    prepared: PreparedSplit,
    model: FittedReranker,
    *,
    gain: float,
    lexical_dimensions: int,
    interaction: bool,
    min_printable_fraction: float,
    useful_reference_cosine: float,
    seed: int,
    resamples: int,
    include_pair_audit: bool,
) -> dict[str, Any]:
    condition_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in CONDITION_NAMES
    }
    qualities = []
    pair_audit = []
    for index, (raw_pair, pair, bank) in enumerate(
        zip(prepared.raw_pairs, prepared.pairs, prepared.banks, strict=True)
    ):
        selections = select_from_bank(
            index, prepared, model, gain=gain, interaction=interaction
        )
        readouts = _bank_readouts(
            raw_pair,
            pair,
            bank,
            model,
            lexical_dimensions=lexical_dimensions,
            interaction=interaction,
        )
        quality = _bank_quality(
            raw_pair,
            bank,
            readouts,
            min_printable_fraction=min_printable_fraction,
            useful_reference_cosine=useful_reference_cosine,
        )
        qualities.append(quality)
        records = {}
        injected_indices = _injected_reference_indices(bank)
        for name in CONDITION_NAMES:
            record = _selection_readout(selections[name], readouts)
            if injected_indices is not None:
                recovered_a = (
                    bank.candidates[int(record["selected_index_a"])].data
                    == raw_pair.response_a
                )
                recovered_b = (
                    bank.candidates[int(record["selected_index_b"])].data
                    == raw_pair.response_b
                )
                record["available_matched_response_recovery"] = 0.5 * (
                    int(recovered_a) + int(recovered_b)
                )
                record["recovered_both_matched_responses"] = recovered_a and recovered_b
            condition_rows[name].append(record)
            records[name] = {**selections[name], **record}
        if include_pair_audit:
            pair_audit.append(
                {
                    "pair_id": bank.pair_id,
                    "source_label": raw_pair.source_label,
                    "response_a_text": raw_pair.response_a.decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "response_b_text": raw_pair.response_b.decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "bank_quality": quality,
                    "candidates": [
                        {
                            "index": candidate_index,
                            "origin": row.origin,
                            "sample_seed": row.sample_seed,
                            "uniform_draw_sha256": row.uniform_draw_sha256,
                            "generated_hex": row.data.hex(),
                            "generated_text": row.data.decode(
                                "utf-8", errors="backslashreplace"
                            ),
                            "native_mean_log2": row.native_mean_log2_a,
                            "diagnostics": row.diagnostics,
                        }
                        for candidate_index, row in enumerate(bank.candidates)
                    ],
                    "conditions": records,
                }
            )

    goal_off = condition_rows["goal_off"]
    summaries = {
        name: _summarize_condition(
            rows,
            goal_off,
            seed=seed + condition_index * 2_003,
            resamples=resamples,
        )
        for condition_index, (name, rows) in enumerate(condition_rows.items())
    }
    quality_summary = {
        "banks": len(qualities),
        "mean_unique_candidate_fraction": float(
            np.mean([row["unique_candidate_fraction"] for row in qualities])
        ),
        "mean_pairwise_byte_difference_fraction": float(
            np.mean(
                [row["mean_pairwise_byte_difference_fraction"] for row in qualities]
            )
        ),
        "mean_pairwise_lexical_cosine": float(
            np.mean([row["mean_pairwise_lexical_cosine"] for row in qualities])
        ),
        "all_native_context_traces_bit_identical": all(
            row["all_native_context_traces_bit_identical"] for row in qualities
        ),
        "max_abs_native_mean_log2_a_minus_b": float(
            max(row["max_abs_native_mean_log2_a_minus_b"] for row in qualities)
        ),
        "mean_valid_utf8_fraction": float(
            np.mean([row["valid_utf8_fraction"] for row in qualities])
        ),
        "mean_printable_ascii_fraction": float(
            np.mean([row["mean_printable_ascii_fraction"] for row in qualities])
        ),
        "mean_distinct_4gram_fraction": float(
            np.mean([row["mean_distinct_4gram_fraction"] for row in qualities])
        ),
        "pair_side_fraction_with_reference_proxy_useful_candidate": float(
            np.mean(
                [
                    row[f"reference_proxy_useful_candidate_present_{side}"]
                    for row in qualities
                    for side in ("a", "b")
                ]
            )
        ),
        "pair_side_fraction_with_exact_held_out_response": float(
            np.mean(
                [
                    row[f"exact_held_out_response_present_{side}"]
                    for row in qualities
                    for side in ("a", "b")
                ]
            )
        ),
        "reference_oracle_ceiling_pair_margin": summarize(
            np.asarray(
                [
                    row["reference_oracle_ceiling"]["reference_symmetric_pair_margin"]
                    for row in qualities
                ]
            ),
            seed=seed + 77_101,
            resamples=resamples,
        ),
        "future_oracle_ceiling_pair_margin": summarize(
            np.asarray(
                [
                    row["future_oracle_ceiling"]["future_symmetric_pair_margin"]
                    for row in qualities
                ]
            ),
            seed=seed + 77_201,
            resamples=resamples,
        ),
    }
    result: dict[str, Any] = {
        "gain": gain,
        "conditions": summaries,
        "candidate_bank_quality": quality_summary,
        "protocol_invariants": {
            "goal_off_selects_same_candidate_for_both_sides": all(
                row["selected_index_a"] == row["selected_index_b"]
                for row in condition_rows["goal_off"]
            ),
            "goal_off_reference_pair_margin_exactly_zero": all(
                row["reference_symmetric_pair_margin"] == 0.0
                for row in condition_rows["goal_off"]
            ),
            "candidate_blind_selection_equals_goal_off": all(
                blind["selected_index_a"] == off["selected_index_a"]
                and blind["selected_index_b"] == off["selected_index_b"]
                for blind, off in zip(
                    condition_rows["candidate_blind_action"],
                    condition_rows["goal_off"],
                    strict=True,
                )
            ),
            "all_native_context_traces_bit_identical": quality_summary[
                "all_native_context_traces_bit_identical"
            ],
        },
    }
    if include_pair_audit:
        result["pair_audit"] = pair_audit
    return result


def _select_gain_from_generated_dev(
    prepared: PreparedSplit,
    model: FittedReranker,
    *,
    gains: tuple[float, ...],
    lexical_dimensions: int,
    interaction: bool,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Select by held-out lexical pairing on development candidate banks."""

    curve: dict[str, dict[str, float]] = {}
    best: tuple[float, float, float] | None = None
    selected = gains[0]
    for gain in gains:
        margins = []
        changed = []
        for index, (raw_pair, pair, bank) in enumerate(
            zip(prepared.raw_pairs, prepared.pairs, prepared.banks, strict=True)
        ):
            selections = select_from_bank(
                index, prepared, model, gain=gain, interaction=interaction
            )
            readouts = _bank_readouts(
                raw_pair,
                pair,
                bank,
                model,
                lexical_dimensions=lexical_dimensions,
                interaction=interaction,
            )
            causal = _selection_readout(selections["causal_full_history"], readouts)
            goal_off = _selection_readout(selections["goal_off"], readouts)
            margins.append(float(causal["reference_symmetric_pair_margin"]))
            changed.extend(
                (
                    causal["selected_index_a"] != goal_off["selected_index_a"],
                    causal["selected_index_b"] != goal_off["selected_index_b"],
                )
            )
        values = np.asarray(margins)
        mean_margin = float(np.mean(values))
        win_fraction = float(np.mean(values > 0.0))
        curve[str(gain)] = {
            "causal_mean_reference_symmetric_pair_margin": mean_margin,
            "causal_reference_pair_win_fraction": win_fraction,
            "causal_selected_side_fraction_changed_vs_goal_off": float(
                np.mean(changed)
            ),
        }
        # The lexical reference readout is independent of the trained action
        # map.  Prefer mean paired discrimination, then consistency, then the
        # least intervention.  Test banks are never inspected here.
        key = (mean_margin, win_fraction, -gain)
        if best is None or key > best:
            best = key
            selected = gain
    return selected, curve


def _build_banks(
    cortex: object,
    pairs: list[CrossoverPair],
    *,
    split: str,
    args: argparse.Namespace,
) -> list[SharedCandidateBank]:
    return [
        build_shared_candidate_bank(
            cortex,
            pair,
            split=split,
            bank_size=args.bank_size,
            max_new_bytes=args.max_new_bytes,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sampling_seed=args.sampling_seed,
            feature_dimensions=args.feature_dimensions,
            lexical_dimensions=args.lexical_dimensions,
            stop_at_story_boundary=args.stop_at_story_boundary,
        )
        for pair in pairs
    ]


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    target_kind = validate_dataset_semantics(args.dataset_dir)
    decays = tuple(float(value) for value in args.state_decays)
    gammas = tuple(float(value) for value in args.successor_gammas)
    examples = load_training_examples(
        args.dataset_dir,
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=gammas,
    )
    dev_pairs = load_crossover_pairs(
        args.dataset_dir,
        "dev",
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=gammas,
    )[: args.max_dev_pairs]
    test_pairs = load_crossover_pairs(
        args.dataset_dir,
        "test",
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=gammas,
    )[: args.max_test_pairs]
    if not dev_pairs or not test_pairs:
        raise ValueError("development and test slices must both be nonempty")
    model = fit_reranker(
        examples,
        feature_dimensions=args.feature_dimensions,
        latent_dimensions=args.latent_dimensions,
        goal_ridge_alpha=args.goal_ridge_alpha,
        action_ridge_alpha=args.action_ridge_alpha,
        plausibility_ridge_alpha=args.plausibility_ridge_alpha,
        seed=args.mechanism_seed,
        interaction=not args.no_interaction,
    )
    cortex = _load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("a frozen native checkpoint is required")
    dev_banks = _build_banks(cortex, dev_pairs, split="dev", args=args)
    test_banks = _build_banks(cortex, test_pairs, split="test", args=args)
    diagnostic_dev_banks = [
        build_reference_injected_bank(
            cortex,
            pair,
            bank,
            feature_dimensions=args.feature_dimensions,
            lexical_dimensions=args.lexical_dimensions,
        )
        for pair, bank in zip(dev_pairs, dev_banks, strict=True)
    ]
    diagnostic_test_banks = [
        build_reference_injected_bank(
            cortex,
            pair,
            bank,
            feature_dimensions=args.feature_dimensions,
            lexical_dimensions=args.lexical_dimensions,
        )
        for pair, bank in zip(test_pairs, test_banks, strict=True)
    ]
    prepared_dev = _prepare_split(dev_pairs, dev_banks, model, seed=args.mechanism_seed)
    prepared_test = _prepare_split(
        test_pairs, test_banks, model, seed=args.mechanism_seed
    )
    prepared_diagnostic_dev = _prepare_split(
        dev_pairs, diagnostic_dev_banks, model, seed=args.mechanism_seed
    )
    prepared_diagnostic_test = _prepare_split(
        test_pairs, diagnostic_test_banks, model, seed=args.mechanism_seed
    )
    selected_gain, gain_curve = _select_gain_from_generated_dev(
        prepared_dev,
        model,
        gains=tuple(args.gain_values),
        lexical_dimensions=args.lexical_dimensions,
        interaction=not args.no_interaction,
    )
    diagnostic_gain, diagnostic_gain_curve = _select_gain_from_generated_dev(
        prepared_diagnostic_dev,
        model,
        gains=tuple(args.gain_values),
        lexical_dimensions=args.lexical_dimensions,
        interaction=not args.no_interaction,
    )
    dev = evaluate_prepared_split(
        prepared_dev,
        model,
        gain=selected_gain,
        lexical_dimensions=args.lexical_dimensions,
        interaction=not args.no_interaction,
        min_printable_fraction=args.min_printable_fraction,
        useful_reference_cosine=args.useful_reference_cosine,
        seed=args.mechanism_seed + 101,
        resamples=args.resamples,
        include_pair_audit=False,
    )
    test = evaluate_prepared_split(
        prepared_test,
        model,
        gain=selected_gain,
        lexical_dimensions=args.lexical_dimensions,
        interaction=not args.no_interaction,
        min_printable_fraction=args.min_printable_fraction,
        useful_reference_cosine=args.useful_reference_cosine,
        seed=args.mechanism_seed + 211,
        resamples=args.resamples,
        include_pair_audit=args.include_pair_audit,
    )
    diagnostic_dev = evaluate_prepared_split(
        prepared_diagnostic_dev,
        model,
        gain=diagnostic_gain,
        lexical_dimensions=args.lexical_dimensions,
        interaction=not args.no_interaction,
        min_printable_fraction=args.min_printable_fraction,
        useful_reference_cosine=args.useful_reference_cosine,
        seed=args.mechanism_seed + 307,
        resamples=args.resamples,
        include_pair_audit=False,
    )
    diagnostic_test = evaluate_prepared_split(
        prepared_diagnostic_test,
        model,
        gain=diagnostic_gain,
        lexical_dimensions=args.lexical_dimensions,
        interaction=not args.no_interaction,
        min_printable_fraction=args.min_printable_fraction,
        useful_reference_cosine=args.useful_reference_cosine,
        seed=args.mechanism_seed + 401,
        resamples=args.resamples,
        include_pair_audit=args.include_pair_audit,
    )
    diagnostic_test_at_generated_gain = evaluate_prepared_split(
        prepared_diagnostic_test,
        model,
        gain=selected_gain,
        lexical_dimensions=args.lexical_dimensions,
        interaction=not args.no_interaction,
        min_printable_fraction=args.min_printable_fraction,
        useful_reference_cosine=args.useful_reference_cosine,
        seed=args.mechanism_seed + 503,
        resamples=args.resamples,
        include_pair_audit=False,
    )
    invariant_values = [
        bool(value)
        for split in (
            dev,
            test,
            diagnostic_dev,
            diagnostic_test,
            diagnostic_test_at_generated_gain,
        )
        for value in split["protocol_invariants"].values()
    ]
    protocol_valid = all(invariant_values)
    useful_fraction = float(
        test["candidate_bank_quality"][
            "pair_side_fraction_with_reference_proxy_useful_candidate"
        ]
    )
    return {
        "experiment": "shared_goal_frozen_native_best_of_n_reranking",
        "status": "complete" if protocol_valid else "invalid_protocol_invariant",
        "target_kind": target_kind,
        "checkpoint": str(args.checkpoint),
        "frozen_native_protocol": {
            "one_shared_candidate_bank_per_pair": True,
            "bank_sampled_once_from_context_a": True,
            "every_candidate_replayed_under_context_b": True,
            "native_context_trace_equality_required": True,
            "reset_before_every_generation_or_replay": True,
            "learn_during_prompt_generation_or_replay": False,
            "episodic_layer_enabled": False,
            "inference_adaptation_frozen": True,
            "deterministic_identifier_derived_sampling_seeds": True,
        },
        "mechanism": {
            "base_score": "frozen native mean log2 probability per candidate byte",
            "goal_score": (
                "cosine between frozen candidate-conditioned successor prediction "
                "and inferred continuous goal state"
            ),
            "selection": "argmax(base_score + development_selected_gain * goal_score)",
            "goal_state": (
                "train-only PCA of strictly-later targets with causal ridge inference"
            ),
        },
        "evaluation": {
            "lexical_response_readout": (
                "frozen signed raw-byte 2/3/4-gram hash cosine against held-out "
                "real responses; separate from the action map"
            ),
            "future_readout": (
                "frozen real action-successor prediction cosine against the true "
                "strictly-later latent target"
            ),
            "primary_pairing_metric": (
                "A-selected/A-reference + B-selected/B-reference minus crossed "
                "assignments"
            ),
        },
        "config": {
            "dataset_dir": str(args.dataset_dir),
            "feature_dimensions": args.feature_dimensions,
            "latent_dimensions": args.latent_dimensions,
            "lexical_dimensions": args.lexical_dimensions,
            "state_decays": list(decays),
            "successor_gammas": list(gammas),
            "goal_ridge_alpha": args.goal_ridge_alpha,
            "action_ridge_alpha": args.action_ridge_alpha,
            "plausibility_ridge_alpha": args.plausibility_ridge_alpha,
            "mechanism_seed": args.mechanism_seed,
            "sampling_seed": args.sampling_seed,
            "bank_size": args.bank_size,
            "max_new_bytes": args.max_new_bytes,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "stop_at_story_boundary": args.stop_at_story_boundary,
            "gain_values": args.gain_values,
            "min_printable_fraction": args.min_printable_fraction,
            "useful_reference_cosine": args.useful_reference_cosine,
            "use_bounded_interaction": not args.no_interaction,
            "resamples": args.resamples,
            "include_pair_audit": args.include_pair_audit,
        },
        "counts": {
            "train_episodes": len(examples),
            "dev_pairs": len(dev_pairs),
            "test_pairs": len(test_pairs),
            "candidate_banks": len(dev_banks) + len(test_banks),
            "native_candidates": (len(dev_banks) + len(test_banks)) * args.bank_size,
            "offline_injected_candidates": 2
            * (len(diagnostic_dev_banks) + len(diagnostic_test_banks)),
        },
        "selected_gain_from_generated_dev": selected_gain,
        "dev_gain_curve": gain_curve,
        "reference_injected_diagnostic": {
            "availability": "offline_only; held-out responses are unavailable online",
            "pool": "same generated bank plus response_a and response_b",
            "selection_gain_selected_only_on_injected_development_pools": (
                diagnostic_gain
            ),
            "dev_gain_curve": diagnostic_gain_curve,
            "dev": diagnostic_dev,
            "test": diagnostic_test,
            "test_at_generated_pool_gain": diagnostic_test_at_generated_gain,
        },
        "conditions": list(CONDITION_NAMES),
        "dev": dev,
        "test": test,
        "protocol_valid": protocol_valid,
        "candidate_availability_conclusion": {
            "reference_proxy_useful_pair_side_fraction": useful_fraction,
            "all_pair_sides_have_reference_proxy_useful_candidate": (
                useful_fraction == 1.0
            ),
            "candidate_limited": useful_fraction < 1.0,
            "interpretation": (
                "A failed selector comparison is candidate-limited wherever no bank "
                "member clears the declared readability and held-out-reference "
                "cosine floor. Passing this heuristic is not evidence of coherent "
                "or thoughtful language."
            ),
        },
        "claim_limits": [
            (
                "The selector can only choose content already sampled by the frozen "
                "500MB generator; it cannot repair or invent an absent response."
            ),
            (
                "The reference-proxy useful-candidate flag is an explicit lexical "
                "and byte-readability heuristic, not a human coherence judgment."
            ),
            (
                "The future readout shares the train-fitted action representation "
                "with the selector and is mechanistic evidence; the byte-ngram "
                "response readout is the more independent outcome."
            ),
            (
                "Oracle future targets are diagnostic and unavailable online; only "
                "causal_full_history is deployable."
            ),
            (
                "The reference-injected pool is an offline selector-competence "
                "diagnostic: it includes both held-out answers and has its own "
                "development-calibrated gain, so it cannot be deployed."
            ),
            (
                "Natural-dialogue future compatibility does not uniquely identify "
                "goals or intent, and hashed text adapters do not establish semantic "
                "modality generality."
            ),
        ],
        "wall_seconds": time.perf_counter() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-dev-pairs", type=int, default=16)
    parser.add_argument("--max-test-pairs", type=int, default=32)
    parser.add_argument("--bank-size", type=int, default=16)
    parser.add_argument("--max-new-bytes", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--feature-dimensions", type=int, default=32)
    parser.add_argument("--latent-dimensions", type=int, default=32)
    parser.add_argument("--lexical-dimensions", type=int, default=2048)
    parser.add_argument("--state-decays", type=float, nargs="+", default=DEFAULT_DECAYS)
    parser.add_argument(
        "--successor-gammas",
        type=float,
        nargs="+",
        default=DEFAULT_SUCCESSOR_GAMMAS,
    )
    parser.add_argument("--goal-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--action-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--plausibility-ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--gain-values",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    )
    parser.add_argument("--mechanism-seed", type=int, default=3)
    parser.add_argument("--sampling-seed", type=int, default=20_260_914)
    parser.add_argument("--min-printable-fraction", type=float, default=0.8)
    parser.add_argument("--useful-reference-cosine", type=float, default=0.05)
    parser.add_argument("--resamples", type=int, default=4_000)
    parser.add_argument("--stop-at-story-boundary", action="store_true")
    parser.add_argument("--no-interaction", action="store_true")
    parser.add_argument("--include-pair-audit", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_dev_pairs < 1 or args.max_test_pairs < 1:
        raise ValueError("max-dev-pairs and max-test-pairs must be positive")
    if args.bank_size < 2 or args.max_new_bytes < 1:
        raise ValueError("bank-size must be at least two and max-new-bytes positive")
    if args.feature_dimensions < 8 or args.latent_dimensions < 1:
        raise ValueError(
            "feature dimensions must be >= 8 and latent dimensions positive"
        )
    if args.lexical_dimensions < 16:
        raise ValueError("lexical-dimensions must be at least 16")
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if not 0 <= args.top_k <= 256 or not 0.0 < args.top_p <= 1.0:
        raise ValueError("top-k/top-p lie outside their valid ranges")
    if any(not 0.0 < value < 1.0 for value in args.state_decays):
        raise ValueError("state decays must lie in (0, 1)")
    if any(not 0.0 < value < 1.0 for value in args.successor_gammas):
        raise ValueError("successor gammas must lie in (0, 1)")
    if any(
        value <= 0.0
        for value in (
            args.goal_ridge_alpha,
            args.action_ridge_alpha,
            args.plausibility_ridge_alpha,
        )
    ):
        raise ValueError("ridge penalties must be positive")
    if not args.gain_values or any(value < 0.0 for value in args.gain_values):
        raise ValueError("gain-values must be a nonempty non-negative list")
    if not 0.0 <= args.min_printable_fraction <= 1.0:
        raise ValueError("min-printable-fraction must lie in [0, 1]")
    if not -1.0 <= args.useful_reference_cosine <= 1.0:
        raise ValueError("useful-reference-cosine must lie in [-1, 1]")
    if args.resamples < 100:
        raise ValueError("resamples must be at least 100")
    report = run_ablation(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
