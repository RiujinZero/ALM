from __future__ import annotations

import numpy as np

from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    CrossoverPair,
    TrainingExample,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    _hashed_byte_ngram_vector,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_reranker_ablation import (  # noqa: E501
    GeneratedCandidate,
    SharedCandidateBank,
    _prepare_split,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    hashed_text_vector,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_rollout_search_ablation import (  # noqa: E501
    FLAT_CAUSAL,
    PairRolloutTree,
    build_pair_rollout_tree,
    compute_matched_flat_bank_size,
    evaluate_search_split,
    select_search_paths,
    stable_rollout_seed,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation import (  # noqa: E501
    fit_reranker,
)


class _PrefixTreeFrozenCortex:
    story_boundary = b"<END>"

    def __init__(self) -> None:
        self.state = bytearray()
        self.learn_flags: list[bool] = []
        self.reset_calls = 0

    def reset_episode(self) -> None:
        self.state.clear()
        self.reset_calls += 1

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> np.ndarray:
        self.learn_flags.append(learn)
        self.state.extend(byte_ids.tolist())
        return np.zeros(len(byte_ids), dtype=np.float64)

    def predict_next_distribution(self) -> np.ndarray:
        suffix = bytes(self.state).rsplit(b">", 1)[-1]
        probability = np.zeros(256, dtype=np.float64)
        if not suffix:
            probability[ord("A")] = 0.5
            probability[ord("B")] = 0.5
        elif suffix in {b"A", b"B"}:
            probability[ord("0")] = 1.0
        elif suffix in {b"A0", b"B0"}:
            probability[ord("g")] = 0.25
            probability[ord("x")] = 0.75
        else:
            probability[ord(".")] = 1.0
        return probability

    def advance_observation(self, byte_id: int, *, learn: bool) -> float:
        self.learn_flags.append(learn)
        self.state.append(byte_id)
        return 0.0


def _raw_pair() -> CrossoverPair:
    dimensions = 8
    pending = np.eye(1, dimensions, 0)[0]
    persistent = np.eye(1, dimensions, 1)[0]
    return CrossoverPair(
        pair_id="pair-1",
        context_a=b"red history>",
        context_b=b"blue history>",
        response_a=b"alphaaaa",
        response_b=b"betabbbb",
        evidence_a=np.concatenate((pending, persistent)),
        evidence_b=np.concatenate((pending, -persistent)),
        source_label="synthetic",
        goal_target_a=b"later alpha",
        goal_target_b=b"later beta",
        goal_target_kind="successor",
        goal_target_vector_a=np.asarray([1.0, 0.0, 0.0, 0.0]),
        goal_target_vector_b=np.asarray([-1.0, 0.0, 0.0, 0.0]),
    )


def _model() -> object:
    dimensions = 8
    response_a = b"alphaaaa"
    response_b = b"betabbbb"
    vector_a = hashed_text_vector(response_a.decode(), dimensions)
    vector_b = hashed_text_vector(response_b.decode(), dimensions)
    pending = np.eye(1, dimensions, 0)[0]
    examples = []
    for index in range(80):
        side_a = index % 2 == 0
        persistent = np.eye(1, dimensions, 1)[0]
        if not side_a:
            persistent *= -1.0
        examples.append(
            TrainingExample(
                episode_id=f"episode-{index}",
                response=response_a if side_a else response_b,
                prefix=b"ASSISTANT: ",
                evidence=np.concatenate((pending, persistent)),
                response_vector=vector_a if side_a else vector_b,
                goal_target_vector=np.asarray([1.0 if side_a else -1.0, 0.0, 0.0, 0.0]),
            )
        )
    return fit_reranker(
        examples,
        feature_dimensions=dimensions,
        latent_dimensions=2,
        goal_ridge_alpha=1e-4,
        action_ridge_alpha=1e-4,
        plausibility_ridge_alpha=1e-4,
        seed=7,
        interaction=False,
    )


def _candidate(data: bytes, base: float) -> GeneratedCandidate:
    return GeneratedCandidate(
        data=data,
        native_mean_log2_a=base,
        native_mean_log2_b=base,
        action_vector=hashed_text_vector(data.decode(), 8),
        lexical_vector=_hashed_byte_ngram_vector(data, 128),
        diagnostics={
            "valid_utf8": True,
            "printable_ascii_fraction": 1.0,
            "distinct_ngram_fraction": {"4": 1.0},
        },
        sample_seed=1,
        uniform_draw_sha256="digest",
        native_trace_a_equals_b=True,
    )


def _manual_tree() -> PairRolloutTree:
    pair_id = "pair-1"
    root_a = _candidate(b"alphaaaa", -2.0)
    root_b = _candidate(b"betabbbb", -2.0)
    flat = SharedCandidateBank(pair_id, (root_a, root_b, _candidate(b"otherxxx", -3.0)))
    return PairRolloutTree(
        pair_id=pair_id,
        horizon_bytes=8,
        commit_bytes=4,
        branch_factor=2,
        flat_bank=flat,
        continuation_banks={
            b"alph": SharedCandidateBank(
                pair_id,
                (
                    _candidate(b"alphaaaa", -2.0),
                    _candidate(b"alphxxxx", -1.0),
                ),
            ),
            b"beta": SharedCandidateBank(
                pair_id,
                (
                    _candidate(b"betabbbb", -2.0),
                    _candidate(b"betaxxxx", -1.0),
                ),
            ),
        },
    )


def test_default_search_and_flat_terminal_budgets_match_exactly() -> None:
    assert compute_matched_flat_bank_size(8, 96, 48) == 12
    assert 8 * 96 + 8 * 48 == 12 * 96
    try:
        compute_matched_flat_bank_size(3, 10, 4)
    except ValueError as exc:
        assert "not an integer" in str(exc)
    else:
        raise AssertionError("an inexact flat budget must be rejected")


def test_prefix_keyed_seed_is_stable_and_changes_with_prefix() -> None:
    first = stable_rollout_seed(7, "test", "pair", 1, b"prefix-a", 2)
    assert first == stable_rollout_seed(7, "test", "pair", 1, b"prefix-a", 2)
    assert first != stable_rollout_seed(7, "test", "pair", 1, b"prefix-b", 2)


def test_rollout_tree_is_reproducible_read_only_fixed_and_cross_context_exact() -> None:
    pair = _raw_pair()
    cortex = _PrefixTreeFrozenCortex()
    kwargs = dict(
        split="test",
        branch_factor=2,
        horizon_bytes=4,
        commit_bytes=2,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        sampling_seed=2,
        feature_dimensions=8,
        lexical_dimensions=128,
    )
    first = build_pair_rollout_tree(cortex, pair, **kwargs)
    second = build_pair_rollout_tree(cortex, pair, **kwargs)

    assert [row.data for row in first.flat_bank.candidates] == [
        row.data for row in second.flat_bank.candidates
    ]
    assert first.continuation_banks.keys() == second.continuation_banks.keys()
    assert all(not flag for flag in cortex.learn_flags)
    candidates = list(first.flat_bank.candidates) + [
        row for bank in first.continuation_banks.values() for row in bank.candidates
    ]
    assert candidates and all(len(row.data) == 4 for row in candidates)
    assert all(row.native_trace_a_equals_b for row in candidates)
    assert all(row.native_mean_log2_a == row.native_mean_log2_b for row in candidates)


def test_two_stage_controls_and_independent_evaluation_are_wired() -> None:
    raw_pair = _raw_pair()
    model = _model()
    tree = _manual_tree()
    prepared = _prepare_split([raw_pair], [tree.root_bank], model, seed=7)
    selected = select_search_paths(
        0, prepared, tree, model, gain=100.0, interaction=False
    )

    assert selected["causal_full_history"].candidate_a.data == raw_pair.response_a
    assert selected["causal_full_history"].candidate_b.data == raw_pair.response_b
    assert selected["swapped_causal_full_history"].candidate_a.data.startswith(b"beta")
    assert selected["swapped_causal_full_history"].candidate_b.data.startswith(b"alph")
    assert (
        selected["candidate_blind_action"].candidate_a.data
        == selected["goal_off"].candidate_a.data
    )
    assert (
        selected["candidate_blind_action"].candidate_b.data
        == selected["goal_off"].candidate_b.data
    )
    assert FLAT_CAUSAL in selected

    report = evaluate_search_split(
        prepared,
        [tree],
        model,
        gain=100.0,
        lexical_dimensions=128,
        interaction=False,
        seed=11,
        resamples=100,
        include_pair_audit=False,
    )
    assert all(report["protocol_invariants"].values())
    causal = report["conditions"]["causal_full_history"]
    assert causal["independent_reference_pairing"]["mean"] > 0.0
    assert causal["reached_pool_reference_oracle_ceiling"]["mean"] > 0.0
    assert (
        "causal_receding_minus_flat_terminal_causal_full_history"
        in report["predeclared_primary_contrasts"]
    )
