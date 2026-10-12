"""Test successor guidance inside frozen-native proposal search.

The whole-candidate trajectory reranker can identify an appropriate supplied
response, but it cannot create a response that is absent from a fixed sample
bank.  This ablation moves the same frozen score one step earlier by using a
two-stage receding-horizon rollout:

1. sample ``K`` complete native rollouts and select one with the declared
   condition's native-plus-successor score;
2. commit only the first ``C`` bytes, resample ``K`` suffixes from that prefix,
   and select the resulting complete response with the same score.

Consequently the final response need not occur in the initial complete bank.
The successor representation and action map still see complete, fixed-length
candidates, avoiding an unvalidated partial-response action model.

All branches use a frozen native checkpoint, ``learn=False``, deterministic
identifier- and prefix-keyed draws, and exact A/B replay checks.  A flat
terminal bank has the same conceptual native prediction-byte budget as one
two-stage decision.  Gain is selected only with development references; test
trees are not even sampled until that choice is frozen.  Held-out responses
are evaluation targets only and are never inserted into a proposal bank.

Above the byte generator and text event adapter, the search consumes only
dense action vectors, a dense trajectory state, and an action-successor map;
the search rule is therefore reusable with other modality-specific proposal
and event encoders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
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
    _paired_reference_discrimination,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_reranker_ablation import (  # noqa: E501
    CONDITION_NAMES as RERANKER_CONDITIONS,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_reranker_ablation import (  # noqa: E501
    GeneratedCandidate,
    PreparedSplit,
    SharedCandidateBank,
    _bank_readouts,
    _candidate_alignment,
    _condition_specifications,
    _generate_with_trace,
    _mean_log2_from_trace,
    _prepare_split,
    _selection_readout,
    select_from_bank,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    DEFAULT_DECAYS,
    DEFAULT_SUCCESSOR_GAMMAS,
    RidgeMap,
    hashed_text_vector,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation import (  # noqa: E501
    FittedReranker,
    fit_reranker,
    summarize,
    validate_dataset_semantics,
)

FLAT_GOAL_OFF = "flat_terminal_goal_off"
FLAT_CAUSAL = "flat_terminal_causal_full_history"
CONDITION_NAMES = (*RERANKER_CONDITIONS, FLAT_GOAL_OFF, FLAT_CAUSAL)


@dataclass(frozen=True)
class PairRolloutTree:
    """A label-free frozen proposal tree for one crossover pair."""

    pair_id: str
    horizon_bytes: int
    commit_bytes: int
    branch_factor: int
    flat_bank: SharedCandidateBank
    continuation_banks: dict[bytes, SharedCandidateBank]

    @property
    def root_bank(self) -> SharedCandidateBank:
        return SharedCandidateBank(
            self.pair_id, self.flat_bank.candidates[: self.branch_factor]
        )


@dataclass(frozen=True)
class SelectedPath:
    """One pair of outputs plus the two search decisions that produced it."""

    candidate_a: GeneratedCandidate
    candidate_b: GeneratedCandidate
    reached_bank_a: SharedCandidateBank
    reached_bank_b: SharedCandidateBank
    root_index_a: int
    root_index_b: int
    final_index_a: int
    final_index_b: int
    selected_score_a: float
    selected_score_b: float
    selected_alignment_a: float
    selected_alignment_b: float


def compute_matched_flat_bank_size(
    branch_factor: int, horizon_bytes: int, commit_bytes: int
) -> int:
    """Return the full-rollout count matching one two-stage prediction budget."""

    if branch_factor < 2:
        raise ValueError("branch_factor must be at least two")
    if horizon_bytes < 2 or not 0 < commit_bytes < horizon_bytes:
        raise ValueError("commit_bytes must lie strictly inside the horizon")
    # Search predicts K*H bytes at the root and K*(H-C) after committing C.
    numerator = branch_factor * (2 * horizon_bytes - commit_bytes)
    count, remainder = divmod(numerator, horizon_bytes)
    if remainder:
        raise ValueError(
            "the configured search budget is not an integer number of flat "
            "full-horizon candidates"
        )
    return count


def stable_rollout_seed(
    sampling_seed: int,
    split: str,
    pair_id: str,
    stage: int,
    prefix: bytes,
    rollout_index: int,
) -> int:
    """Derive a stable seed whose proposal identity includes its exact prefix."""

    if stage < 0 or rollout_index < 0:
        raise ValueError("stage and rollout_index must be non-negative")
    prefix_digest = hashlib.sha256(prefix).digest()
    payload = b"\x00".join(
        (
            str(sampling_seed).encode("ascii"),
            split.encode("utf-8"),
            pair_id.encode("utf-8"),
            str(stage).encode("ascii"),
            str(rollout_index).encode("ascii"),
            prefix_digest,
        )
    )
    digest = hashlib.blake2b(payload, digest_size=8, person=b"alm-mpc").digest()
    return int.from_bytes(digest, "little", signed=False)


def _empty_trace() -> np.ndarray:
    return np.empty((0, 256), dtype=np.float64)


def build_rollout_bank(
    cortex: object,
    pair: CrossoverPair,
    *,
    split: str,
    stage: int,
    prefix: bytes,
    candidate_count: int,
    horizon_bytes: int,
    temperature: float,
    top_k: int,
    top_p: float,
    sampling_seed: int,
    feature_dimensions: int,
    lexical_dimensions: int,
) -> SharedCandidateBank:
    """Generate fixed-horizon candidates and audit exact crossover replay."""

    prefix = bytes(prefix)
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    if len(prefix) >= horizon_bytes:
        raise ValueError("rollout prefix must be shorter than the horizon")
    remaining = horizon_bytes - len(prefix)
    if prefix:
        prefix_trace_a = _native_trace(cortex, pair.context_a, prefix)
        prefix_trace_b = _native_trace(cortex, pair.context_b, prefix)
    else:
        prefix_trace_a = _empty_trace()
        prefix_trace_b = _empty_trace()
    prefix_exact = bool(np.array_equal(prefix_trace_a, prefix_trace_b))
    boundary = getattr(cortex, "story_boundary", b"<|endoftext|>\n")
    if not isinstance(boundary, bytes):
        boundary = bytes(boundary)

    candidates = []
    for rollout_index in range(candidate_count):
        seed = stable_rollout_seed(
            sampling_seed,
            split,
            pair.pair_id,
            stage,
            prefix,
            rollout_index,
        )
        draws = np.random.default_rng(seed).random(remaining)
        suffix, suffix_trace_a = _generate_with_trace(
            cortex,
            pair.context_a + prefix,
            draws,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_at_story_boundary=False,
        )
        if len(suffix) != remaining:
            raise AssertionError("fixed-horizon rollout ended at the wrong length")
        suffix_trace_b = _native_trace(cortex, pair.context_b + prefix, suffix)
        data = prefix + suffix
        trace_a = np.concatenate((prefix_trace_a, suffix_trace_a), axis=0)
        trace_b = np.concatenate((prefix_trace_b, suffix_trace_b), axis=0)
        trace_exact = prefix_exact and bool(np.array_equal(trace_a, trace_b))
        candidates.append(
            GeneratedCandidate(
                data=data,
                native_mean_log2_a=_mean_log2_from_trace(trace_a, data),
                native_mean_log2_b=_mean_log2_from_trace(trace_b, data),
                action_vector=hashed_text_vector(
                    data.decode("utf-8", errors="replace"), feature_dimensions
                ),
                lexical_vector=_hashed_byte_ngram_vector(data, lexical_dimensions),
                diagnostics=_generation_diagnostics(data, boundary),
                sample_seed=seed,
                uniform_draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),
                native_trace_a_equals_b=trace_exact,
                origin=f"rollout_stage_{stage}",
            )
        )
    return SharedCandidateBank(pair.pair_id, tuple(candidates))


def build_pair_rollout_tree(
    cortex: object,
    pair: CrossoverPair,
    *,
    split: str,
    branch_factor: int,
    horizon_bytes: int,
    commit_bytes: int,
    temperature: float,
    top_k: int,
    top_p: float,
    sampling_seed: int,
    feature_dimensions: int,
    lexical_dimensions: int,
) -> PairRolloutTree:
    """Sample the complete label-free tree before any condition is evaluated."""

    flat_count = compute_matched_flat_bank_size(
        branch_factor, horizon_bytes, commit_bytes
    )
    flat_bank = build_rollout_bank(
        cortex,
        pair,
        split=split,
        stage=0,
        prefix=b"",
        candidate_count=flat_count,
        horizon_bytes=horizon_bytes,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        sampling_seed=sampling_seed,
        feature_dimensions=feature_dimensions,
        lexical_dimensions=lexical_dimensions,
    )
    prefixes = {row.data[:commit_bytes] for row in flat_bank.candidates[:branch_factor]}
    continuation_banks = {
        prefix: build_rollout_bank(
            cortex,
            pair,
            split=split,
            stage=1,
            prefix=prefix,
            candidate_count=branch_factor,
            horizon_bytes=horizon_bytes,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sampling_seed=sampling_seed,
            feature_dimensions=feature_dimensions,
            lexical_dimensions=lexical_dimensions,
        )
        for prefix in sorted(prefixes)
    }
    return PairRolloutTree(
        pair_id=pair.pair_id,
        horizon_bytes=horizon_bytes,
        commit_bytes=commit_bytes,
        branch_factor=branch_factor,
        flat_bank=flat_bank,
        continuation_banks=continuation_banks,
    )


def _select_side(
    bank: SharedCandidateBank,
    pair: Any,
    model: FittedReranker,
    *,
    side: str,
    goal: np.ndarray | None,
    action_model: RidgeMap | None,
    candidate_blind: bool,
    gain: float,
    interaction: bool,
) -> tuple[int, float, float]:
    if side not in {"a", "b"}:
        raise ValueError("side must be 'a' or 'b'")
    base = np.asarray(
        [
            row.native_mean_log2_a if side == "a" else row.native_mean_log2_b
            for row in bank.candidates
        ],
        dtype=np.float64,
    )
    if goal is None:
        alignment = np.zeros(len(bank.candidates), dtype=np.float64)
    else:
        pending = pair.pending_a if side == "a" else pair.pending_b
        candidates = np.asarray([row.action_vector for row in bank.candidates])
        alignment = _candidate_alignment(
            model,
            pending,
            candidates,
            goal,
            action_model=action_model,
            candidate_blind=candidate_blind,
            interaction=interaction,
        )
    scores = base + gain * alignment
    selected = int(np.argmax(scores))
    return selected, float(scores[selected]), float(alignment[selected])


def _condition_parameters(
    name: str,
    pair_index: int,
    prepared: PreparedSplit,
    model: FittedReranker,
) -> tuple[np.ndarray | None, np.ndarray | None, RidgeMap | None, bool]:
    if name == "goal_off":
        return None, None, None, False
    specs = _condition_specifications(
        pair_index, prepared.pairs[pair_index], prepared, model
    )
    goal_a, goal_b, action_model, blind = specs[name]
    return goal_a, goal_b, action_model, blind


def select_search_paths(
    pair_index: int,
    prepared: PreparedSplit,
    tree: PairRolloutTree,
    model: FittedReranker,
    *,
    gain: float,
    interaction: bool,
) -> dict[str, SelectedPath]:
    """Select every controlled two-stage path and both flat baselines."""

    pair = prepared.pairs[pair_index]
    prepared_bank = prepared.banks[pair_index]
    root_bank = tree.root_bank
    if (
        tree.pair_id != pair.pair_id
        or prepared_bank.pair_id != root_bank.pair_id
        or tuple(row.data for row in prepared_bank.candidates)
        != tuple(row.data for row in root_bank.candidates)
    ):
        raise ValueError("prepared split and rollout tree are not aligned")
    first = select_from_bank(
        pair_index, prepared, model, gain=gain, interaction=interaction
    )
    result: dict[str, SelectedPath] = {}
    for name in RERANKER_CONDITIONS:
        goal_a, goal_b, action_model, blind = _condition_parameters(
            name, pair_index, prepared, model
        )
        root_a = int(first[name]["selected_index_a"])
        root_b = int(first[name]["selected_index_b"])
        prefix_a = tree.root_bank.candidates[root_a].data[: tree.commit_bytes]
        prefix_b = tree.root_bank.candidates[root_b].data[: tree.commit_bytes]
        bank_a = tree.continuation_banks[prefix_a]
        bank_b = tree.continuation_banks[prefix_b]
        final_a, score_a, alignment_a = _select_side(
            bank_a,
            pair,
            model,
            side="a",
            goal=goal_a,
            action_model=action_model,
            candidate_blind=blind,
            gain=gain,
            interaction=interaction,
        )
        final_b, score_b, alignment_b = _select_side(
            bank_b,
            pair,
            model,
            side="b",
            goal=goal_b,
            action_model=action_model,
            candidate_blind=blind,
            gain=gain,
            interaction=interaction,
        )
        result[name] = SelectedPath(
            candidate_a=bank_a.candidates[final_a],
            candidate_b=bank_b.candidates[final_b],
            reached_bank_a=bank_a,
            reached_bank_b=bank_b,
            root_index_a=root_a,
            root_index_b=root_b,
            final_index_a=final_a,
            final_index_b=final_b,
            selected_score_a=score_a,
            selected_score_b=score_b,
            selected_alignment_a=alignment_a,
            selected_alignment_b=alignment_b,
        )

    flat_specs = {
        FLAT_GOAL_OFF: (None, None, None, False),
        FLAT_CAUSAL: _condition_parameters(
            "causal_full_history", pair_index, prepared, model
        ),
    }
    for name, (goal_a, goal_b, action_model, blind) in flat_specs.items():
        index_a, score_a, alignment_a = _select_side(
            tree.flat_bank,
            pair,
            model,
            side="a",
            goal=goal_a,
            action_model=action_model,
            candidate_blind=blind,
            gain=gain,
            interaction=interaction,
        )
        index_b, score_b, alignment_b = _select_side(
            tree.flat_bank,
            pair,
            model,
            side="b",
            goal=goal_b,
            action_model=action_model,
            candidate_blind=blind,
            gain=gain,
            interaction=interaction,
        )
        result[name] = SelectedPath(
            candidate_a=tree.flat_bank.candidates[index_a],
            candidate_b=tree.flat_bank.candidates[index_b],
            reached_bank_a=tree.flat_bank,
            reached_bank_b=tree.flat_bank,
            root_index_a=index_a,
            root_index_b=index_b,
            final_index_a=index_a,
            final_index_b=index_b,
            selected_score_a=score_a,
            selected_score_b=score_b,
            selected_alignment_a=alignment_a,
            selected_alignment_b=alignment_b,
        )
    return result


def _tree_candidates(tree: PairRolloutTree) -> list[GeneratedCandidate]:
    return list(tree.flat_bank.candidates) + [
        row for bank in tree.continuation_banks.values() for row in bank.candidates
    ]


def _pair_outcome(
    raw_pair: CrossoverPair,
    pair: Any,
    selected: SelectedPath,
    model: FittedReranker,
    *,
    lexical_dimensions: int,
    interaction: bool,
) -> dict[str, Any]:
    independent = _paired_reference_discrimination(
        selected.candidate_a.data,
        selected.candidate_b.data,
        raw_pair.response_a,
        raw_pair.response_b,
        dimensions=lexical_dimensions,
    )
    selected_bank = SharedCandidateBank(
        raw_pair.pair_id, (selected.candidate_a, selected.candidate_b)
    )
    readouts = _bank_readouts(
        raw_pair,
        pair,
        selected_bank,
        model,
        lexical_dimensions=lexical_dimensions,
        interaction=interaction,
    )
    mechanistic = _selection_readout(
        {"selected_index_a": 0, "selected_index_b": 1}, readouts
    )
    references = np.asarray(
        [
            _hashed_byte_ngram_vector(raw_pair.response_a, lexical_dimensions),
            _hashed_byte_ngram_vector(raw_pair.response_b, lexical_dimensions),
        ]
    )
    reached_a = np.asarray(
        [row.lexical_vector for row in selected.reached_bank_a.candidates]
    )
    reached_b = np.asarray(
        [row.lexical_vector for row in selected.reached_bank_b.candidates]
    )
    # This is an evaluation-only availability ceiling. It asks how well the
    # two reached proposal pools could be paired if their held-out references
    # were revealed after search; it never participates in path selection.
    oracle_delta_a = reached_a @ (references[0] - references[1])
    oracle_delta_b = reached_b @ (references[1] - references[0])
    reached_pool_ceiling = float(np.max(oracle_delta_a) + np.max(oracle_delta_b))
    return {
        "independent_reference_margin": float(
            independent["correct_minus_crossed_margin"]
        ),
        "action_shared_future_margin": float(
            mechanistic["future_symmetric_pair_margin"]
        ),
        "reached_pool_reference_oracle_ceiling": reached_pool_ceiling,
        "valid_utf8_side_fraction": float(
            np.mean(
                [
                    selected.candidate_a.diagnostics["valid_utf8"],
                    selected.candidate_b.diagnostics["valid_utf8"],
                ]
            )
        ),
        "printable_ascii_fraction": float(
            np.mean(
                [
                    selected.candidate_a.diagnostics["printable_ascii_fraction"],
                    selected.candidate_b.diagnostics["printable_ascii_fraction"],
                ]
            )
        ),
    }


def _summarize_vector(
    values: list[float] | np.ndarray, *, seed: int, resamples: int
) -> dict[str, float | int]:
    return summarize(
        np.asarray(values, dtype=np.float64), seed=seed, resamples=resamples
    )


def evaluate_search_split(
    prepared: PreparedSplit,
    trees: list[PairRolloutTree],
    model: FittedReranker,
    *,
    gain: float,
    lexical_dimensions: int,
    interaction: bool,
    seed: int,
    resamples: int,
    include_pair_audit: bool,
) -> dict[str, Any]:
    """Evaluate selected bytes; references enter only after every path is fixed."""

    if len(prepared.raw_pairs) != len(trees):
        raise ValueError("prepared pairs and rollout trees must have equal lengths")
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITION_NAMES}
    selected_rows: list[dict[str, SelectedPath]] = []
    pair_audit = []
    for index, (raw_pair, pair, tree) in enumerate(
        zip(prepared.raw_pairs, prepared.pairs, trees, strict=True)
    ):
        selected = select_search_paths(
            index, prepared, tree, model, gain=gain, interaction=interaction
        )
        selected_rows.append(selected)
        for name in CONDITION_NAMES:
            rows[name].append(
                _pair_outcome(
                    raw_pair,
                    pair,
                    selected[name],
                    model,
                    lexical_dimensions=lexical_dimensions,
                    interaction=interaction,
                )
            )
        if include_pair_audit:
            pair_audit.append(
                {
                    "pair_id": raw_pair.pair_id,
                    "conditions": {
                        name: {
                            "root_index_a": value.root_index_a,
                            "root_index_b": value.root_index_b,
                            "final_index_a": value.final_index_a,
                            "final_index_b": value.final_index_b,
                            "generated_a_hex": value.candidate_a.data.hex(),
                            "generated_b_hex": value.candidate_b.data.hex(),
                            "generated_a_text": value.candidate_a.data.decode(
                                "utf-8", errors="backslashreplace"
                            ),
                            "generated_b_text": value.candidate_b.data.decode(
                                "utf-8", errors="backslashreplace"
                            ),
                            "reached_pool_reference_oracle_ceiling": rows[name][-1][
                                "reached_pool_reference_oracle_ceiling"
                            ],
                        }
                        for name, value in selected.items()
                    },
                }
            )

    summaries: dict[str, Any] = {}
    for condition_index, name in enumerate(CONDITION_NAMES):
        independent = [row["independent_reference_margin"] for row in rows[name]]
        future = [row["action_shared_future_margin"] for row in rows[name]]
        ceiling = [row["reached_pool_reference_oracle_ceiling"] for row in rows[name]]
        baseline = [row["independent_reference_margin"] for row in rows["goal_off"]]
        selected_changed = [
            float(
                selected[name].candidate_a.data != selected["goal_off"].candidate_a.data
            )
            for selected in selected_rows
        ] + [
            float(
                selected[name].candidate_b.data != selected["goal_off"].candidate_b.data
            )
            for selected in selected_rows
        ]
        summaries[name] = {
            "independent_reference_pairing": _summarize_vector(
                independent,
                seed=seed + 101 * condition_index,
                resamples=resamples,
            ),
            "independent_delta_vs_receding_goal_off": _summarize_vector(
                np.asarray(independent) - np.asarray(baseline),
                seed=seed + 101 * condition_index + 1,
                resamples=resamples,
            ),
            "action_shared_future_pairing": _summarize_vector(
                future,
                seed=seed + 101 * condition_index + 2,
                resamples=resamples,
            ),
            "reached_pool_reference_oracle_ceiling": _summarize_vector(
                ceiling,
                seed=seed + 101 * condition_index + 3,
                resamples=resamples,
            ),
            "selected_side_fraction_changed_vs_receding_goal_off": float(
                np.mean(selected_changed)
            ),
            "valid_utf8_side_fraction": float(
                np.mean([row["valid_utf8_side_fraction"] for row in rows[name]])
            ),
            "mean_printable_ascii_fraction": float(
                np.mean([row["printable_ascii_fraction"] for row in rows[name]])
            ),
        }

    causal = np.asarray(
        [row["independent_reference_margin"] for row in rows["causal_full_history"]]
    )
    contrasts = {}
    for contrast_index, comparator in enumerate(
        (
            "goal_off",
            FLAT_CAUSAL,
            "causal_pending_user_only",
            "shuffled_predictor_target",
            "shuffled_action_target_mapping",
            "shuffled_candidate_action_mapping",
            "shuffled_episode_goal",
        )
    ):
        other = np.asarray(
            [row["independent_reference_margin"] for row in rows[comparator]]
        )
        contrasts[f"causal_receding_minus_{comparator}"] = _summarize_vector(
            causal - other,
            seed=seed + 2_003 + contrast_index,
            resamples=resamples,
        )

    all_candidates = [row for tree in trees for row in _tree_candidates(tree)]
    exact_trace = all(row.native_trace_a_equals_b for row in all_candidates)
    exact_score = all(
        row.native_mean_log2_a == row.native_mean_log2_b for row in all_candidates
    )
    goal_off_same = all(
        selected["goal_off"].candidate_a.data == selected["goal_off"].candidate_b.data
        for selected in selected_rows
    )
    blind_equals_off = all(
        selected["candidate_blind_action"].candidate_a.data
        == selected["goal_off"].candidate_a.data
        and selected["candidate_blind_action"].candidate_b.data
        == selected["goal_off"].candidate_b.data
        for selected in selected_rows
    )
    fixed_length = all(
        len(row.data) == tree.horizon_bytes
        for tree in trees
        for row in _tree_candidates(tree)
    )
    compute_match = all(
        len(tree.flat_bank.candidates) * tree.horizon_bytes
        == tree.branch_factor * tree.horizon_bytes
        + tree.branch_factor * (tree.horizon_bytes - tree.commit_bytes)
        for tree in trees
    )
    result = {
        "conditions": summaries,
        "predeclared_primary_contrasts": contrasts,
        "protocol_invariants": {
            "all_native_crossover_traces_bit_identical": exact_trace,
            "all_native_crossover_scores_bit_identical": exact_score,
            "receding_goal_off_selects_same_bytes_for_both_sides": goal_off_same,
            "candidate_blind_search_equals_receding_goal_off": blind_equals_off,
            "all_rollouts_have_fixed_horizon": fixed_length,
            "flat_and_two_stage_prediction_byte_budgets_match": compute_match,
        },
        "tree_summary": {
            "pairs": len(trees),
            "flat_candidates_per_pair": len(trees[0].flat_bank.candidates),
            "root_search_candidates_per_pair": trees[0].branch_factor,
            "mean_unique_committed_prefixes": float(
                np.mean([len(tree.continuation_banks) for tree in trees])
            ),
            "stored_native_candidates": int(
                sum(len(_tree_candidates(tree)) for tree in trees)
            ),
            "precomputed_native_prediction_bytes": int(
                sum(
                    len(tree.flat_bank.candidates) * tree.horizon_bytes
                    + len(tree.continuation_banks)
                    * tree.branch_factor
                    * (tree.horizon_bytes - tree.commit_bytes)
                    for tree in trees
                )
            ),
            "cost_accounting_note": (
                "the flat match is a per-condition conceptual native prediction-"
                "byte budget; the precomputed experiment stores every unique root "
                "continuation bank once so all controls can reuse label-free proposals"
            ),
        },
    }
    if include_pair_audit:
        result["pair_audit"] = pair_audit
    return result


def select_gain_on_development(
    prepared: PreparedSplit,
    trees: list[PairRolloutTree],
    model: FittedReranker,
    *,
    gains: tuple[float, ...],
    lexical_dimensions: int,
    interaction: bool,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Select only from development references with a least-gain tie break."""

    if not gains:
        raise ValueError("at least one gain is required")
    curve = {}
    best: tuple[float, float, float] | None = None
    selected_gain = gains[0]
    for gain in gains:
        margins = []
        changed = []
        for index, (raw_pair, tree) in enumerate(
            zip(prepared.raw_pairs, trees, strict=True)
        ):
            selected = select_search_paths(
                index, prepared, tree, model, gain=gain, interaction=interaction
            )
            causal = selected["causal_full_history"]
            goal_off = selected["goal_off"]
            outcome = _paired_reference_discrimination(
                causal.candidate_a.data,
                causal.candidate_b.data,
                raw_pair.response_a,
                raw_pair.response_b,
                dimensions=lexical_dimensions,
            )
            margins.append(float(outcome["correct_minus_crossed_margin"]))
            changed.extend(
                (
                    causal.candidate_a.data != goal_off.candidate_a.data,
                    causal.candidate_b.data != goal_off.candidate_b.data,
                )
            )
        values = np.asarray(margins, dtype=np.float64)
        mean_margin = float(values.mean())
        win_fraction = float(np.mean(values > 0.0))
        curve[str(gain)] = {
            "causal_mean_independent_reference_margin": mean_margin,
            "causal_independent_reference_win_fraction": win_fraction,
            "causal_selected_side_fraction_changed_vs_goal_off": float(
                np.mean(changed)
            ),
        }
        key = (mean_margin, win_fraction, -gain)
        if best is None or key > best:
            best = key
            selected_gain = gain
    return selected_gain, curve


def _build_trees(
    cortex: object,
    pairs: list[CrossoverPair],
    *,
    split: str,
    args: argparse.Namespace,
) -> list[PairRolloutTree]:
    return [
        build_pair_rollout_tree(
            cortex,
            pair,
            split=split,
            branch_factor=args.branch_factor,
            horizon_bytes=args.horizon_bytes,
            commit_bytes=args.commit_bytes,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sampling_seed=args.sampling_seed,
            feature_dimensions=args.feature_dimensions,
            lexical_dimensions=args.lexical_dimensions,
        )
        for pair in pairs
    ]


def _prepare_trees(
    pairs: list[CrossoverPair],
    trees: list[PairRolloutTree],
    model: FittedReranker,
    *,
    seed: int,
) -> PreparedSplit:
    return _prepare_split(pairs, [tree.root_bank for tree in trees], model, seed=seed)


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
        interaction=args.use_interaction,
    )
    cortex = _load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("a frozen native checkpoint is required")

    # Development proposals and labels are fully processed before test
    # proposal sampling begins.  This makes the no-test-label tuning boundary
    # visible in control flow rather than only asserted in report prose.
    dev_trees = _build_trees(cortex, dev_pairs, split="dev", args=args)
    prepared_dev = _prepare_trees(dev_pairs, dev_trees, model, seed=args.mechanism_seed)
    selected_gain, gain_curve = select_gain_on_development(
        prepared_dev,
        dev_trees,
        model,
        gains=tuple(args.gain_values),
        lexical_dimensions=args.lexical_dimensions,
        interaction=args.use_interaction,
    )
    dev = evaluate_search_split(
        prepared_dev,
        dev_trees,
        model,
        gain=selected_gain,
        lexical_dimensions=args.lexical_dimensions,
        interaction=args.use_interaction,
        seed=args.mechanism_seed + 101,
        resamples=args.resamples,
        include_pair_audit=False,
    )

    test_trees = _build_trees(cortex, test_pairs, split="test", args=args)
    prepared_test = _prepare_trees(
        test_pairs, test_trees, model, seed=args.mechanism_seed
    )
    test = evaluate_search_split(
        prepared_test,
        test_trees,
        model,
        gain=selected_gain,
        lexical_dimensions=args.lexical_dimensions,
        interaction=args.use_interaction,
        seed=args.mechanism_seed + 211,
        resamples=args.resamples,
        include_pair_audit=args.include_pair_audit,
    )
    invariants = [
        bool(value)
        for split in (dev, test)
        for value in split["protocol_invariants"].values()
    ]
    protocol_valid = all(invariants)
    flat_count = compute_matched_flat_bank_size(
        args.branch_factor, args.horizon_bytes, args.commit_bytes
    )
    return {
        "experiment": "shared_goal_frozen_native_two_stage_rollout_search",
        "status": "complete" if protocol_valid else "invalid_protocol_invariant",
        "target_kind": target_kind,
        "checkpoint": str(args.checkpoint),
        "mechanism": {
            "search": (
                "score K full-horizon rollouts, commit C bytes, then score K "
                "resampled suffix rollouts"
            ),
            "score": (
                "frozen native mean log2 probability plus gain times cosine of "
                "candidate-conditioned successor and inferred trajectory"
            ),
            "candidate_embedding_scope": (
                "complete fixed-horizon response at both stages"
            ),
            "flat_baseline": (
                "terminal reranking of equally many native prediction bytes"
            ),
            "precomputation_cost_scope": (
                "all K root-prefix continuation banks are sampled once and shared "
                "across conditions; this stored-proposal total is larger than one "
                "condition's compute-matched flat budget"
            ),
        },
        "frozen_native_protocol": {
            "reset_and_replay_every_rollout": True,
            "learn_during_prompt_or_rollout": False,
            "episodic_layer_enabled": False,
            "inference_adaptation_frozen": True,
            "deterministic_prefix_keyed_draws": True,
            "exact_cross_context_trace_equality_required": True,
            "test_trees_sampled_only_after_dev_gain_frozen": True,
            "held_out_responses_used_only_after_path_selection": True,
        },
        "config": {
            "dataset_dir": str(args.dataset_dir),
            "max_dev_pairs": args.max_dev_pairs,
            "max_test_pairs": args.max_test_pairs,
            "branch_factor": args.branch_factor,
            "horizon_bytes": args.horizon_bytes,
            "commit_bytes": args.commit_bytes,
            "flat_bank_size": flat_count,
            "conceptual_prediction_bytes_per_search_condition": (
                args.branch_factor * (2 * args.horizon_bytes - args.commit_bytes)
            ),
            "conceptual_prediction_bytes_flat_terminal": (
                flat_count * args.horizon_bytes
            ),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
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
            "gain_values": list(args.gain_values),
            "use_interaction": args.use_interaction,
            "resamples": args.resamples,
        },
        "counts": {
            "train_episodes": len(examples),
            "dev_pairs": len(dev_pairs),
            "test_pairs": len(test_pairs),
        },
        "selected_gain_from_development_only": selected_gain,
        "development_gain_curve": gain_curve,
        "condition_roles": {
            "causal_full_history": "deployable condition used for dev-only gain tuning",
            "oracle_future_target": (
                "evaluation-only diagnostic using unavailable future; never used "
                "for gain tuning"
            ),
            "reached_pool_reference_oracle_ceiling": (
                "evaluation-only availability bound computed after path selection"
            ),
        },
        "dev": dev,
        "test": test,
        "protocol_valid": protocol_valid,
        "primary_endpoint": (
            "pair-clustered correct-minus-crossed held-out raw-byte n-gram margin"
        ),
        "claim_limits": [
            (
                "The independent endpoint is a byte-ngram proxy, not a human "
                "coherence or thoughtfulness judgment."
            ),
            (
                "The action-shared future endpoint reuses the trained successor "
                "map and is mechanistic evidence only."
            ),
            (
                "Receding-horizon search can recombine sampled paths but cannot "
                "supply concepts absent from the frozen native proposal process."
            ),
            (
                "Natural-dialogue successor compatibility does not uniquely "
                "identify an autonomous goal or intent."
            ),
        ],
        "wall_seconds": time.perf_counter() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-dev-pairs", type=int, default=32)
    parser.add_argument("--max-test-pairs", type=int, default=64)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument("--horizon-bytes", type=int, default=96)
    parser.add_argument("--commit-bytes", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--feature-dimensions", type=int, default=96)
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
    parser.add_argument("--mechanism-seed", type=int, default=3)
    parser.add_argument("--sampling-seed", type=int, default=20260914)
    parser.add_argument(
        "--gain-values",
        type=float,
        nargs="+",
        default=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    )
    parser.add_argument("--use-interaction", action="store_true")
    parser.add_argument("--resamples", type=int, default=4_000)
    parser.add_argument("--include-pair-audit", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_dev_pairs < 1 or args.max_test_pairs < 1:
        raise ValueError("pair limits must be positive")
    compute_matched_flat_bank_size(
        args.branch_factor, args.horizon_bytes, args.commit_bytes
    )
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if not 0 <= args.top_k <= 256 or not 0.0 < args.top_p <= 1.0:
        raise ValueError("top-k/top-p configuration is invalid")
    if args.feature_dimensions < 8 or args.lexical_dimensions < 16:
        raise ValueError("feature dimensions are below their adapter minima")
    if args.latent_dimensions < 1:
        raise ValueError("latent_dimensions must be positive")
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (
            args.goal_ridge_alpha,
            args.action_ridge_alpha,
            args.plausibility_ridge_alpha,
            *args.gain_values,
        )
    ):
        raise ValueError("ridge alphas and gains must be finite and non-negative")
    if args.resamples < 100:
        raise ValueError("resamples must be at least 100")
    report = run_ablation(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
