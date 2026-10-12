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
    _bank_quality,
    _bank_readouts,
    _prepare_split,
    _selection_readout,
    _stable_sample_seed,
    build_reference_injected_bank,
    build_shared_candidate_bank,
    select_from_bank,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    hashed_text_vector,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation import (  # noqa: E501
    _pair_vectors,
    fit_reranker,
)


class _FrozenSuffixCortex:
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
        probability = np.ones(256, dtype=np.float64)
        previous = self.state[-1] if self.state else 0
        probability[(previous + 1) % 256] = 100.0
        return probability / probability.sum()

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
        context_a=b"different history A; same suffix>",
        context_b=b"other history B; same suffix>",
        response_a=b"alpha deliberate trajectory",
        response_b=b"beta deliberate trajectory",
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
    response_a = b"alpha deliberate trajectory"
    response_b = b"beta deliberate trajectory"
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
    )


def _candidate(data: bytes, base: float, *, dimensions: int = 8) -> GeneratedCandidate:
    diagnostics = {
        "valid_utf8": True,
        "printable_ascii_fraction": 1.0,
        "distinct_ngram_fraction": {"4": 1.0},
    }
    return GeneratedCandidate(
        data=data,
        native_mean_log2_a=base,
        native_mean_log2_b=base,
        action_vector=hashed_text_vector(data.decode(), dimensions),
        lexical_vector=_hashed_byte_ngram_vector(data, 128),
        diagnostics=diagnostics,
        sample_seed=1,
        uniform_draw_sha256="digest",
        native_trace_a_equals_b=True,
    )


def test_shared_bank_generation_is_reproducible_read_only_and_cross_context_exact() -> (  # noqa: E501
    None
):
    pair = _raw_pair()
    cortex = _FrozenSuffixCortex()
    first = build_shared_candidate_bank(
        cortex,
        pair,
        split="test",
        bank_size=4,
        max_new_bytes=12,
        temperature=0.8,
        top_k=16,
        top_p=0.9,
        sampling_seed=19,
        feature_dimensions=8,
        lexical_dimensions=128,
        stop_at_story_boundary=False,
    )
    second = build_shared_candidate_bank(
        cortex,
        pair,
        split="test",
        bank_size=4,
        max_new_bytes=12,
        temperature=0.8,
        top_k=16,
        top_p=0.9,
        sampling_seed=19,
        feature_dimensions=8,
        lexical_dimensions=128,
        stop_at_story_boundary=False,
    )

    assert [row.data for row in first.candidates] == [
        row.data for row in second.candidates
    ]
    assert all(row.native_trace_a_equals_b for row in first.candidates)
    assert all(
        row.native_mean_log2_a == row.native_mean_log2_b for row in first.candidates
    )
    assert cortex.learn_flags and not any(cortex.learn_flags)
    assert _stable_sample_seed(19, "test", pair.pair_id, 2) == _stable_sample_seed(
        19, "test", pair.pair_id, 2
    )


def test_goal_reranking_selects_matched_actions_while_blind_equals_base() -> None:
    model = _model()
    raw_pair = _raw_pair()
    bank = SharedCandidateBank(
        raw_pair.pair_id,
        (
            _candidate(raw_pair.response_a, -2.0),
            _candidate(raw_pair.response_b, -2.0),
        ),
    )
    prepared = _prepare_split([raw_pair], [bank], model, seed=7)
    selected = select_from_bank(0, prepared, model, gain=10.0, interaction=True)

    assert selected["goal_off"]["selected_index_a"] == 0
    assert selected["goal_off"]["selected_index_b"] == 0
    assert selected["causal_full_history"]["selected_index_a"] == 0
    assert selected["causal_full_history"]["selected_index_b"] == 1
    assert selected["swapped_causal_full_history"]["selected_index_a"] == 1
    assert selected["swapped_causal_full_history"]["selected_index_b"] == 0
    assert selected["candidate_blind_action"]["selected_index_a"] == 0
    assert selected["candidate_blind_action"]["selected_index_b"] == 0


def test_reference_injection_exposes_selector_competence_separately_from_coverage() -> (  # noqa: E501
    None
):
    raw_pair = _raw_pair()
    cortex = _FrozenSuffixCortex()
    generated = SharedCandidateBank(
        raw_pair.pair_id,
        (_candidate(b"unrelated one", -3.0), _candidate(b"unrelated two", -3.1)),
    )
    injected = build_reference_injected_bank(
        cortex,
        raw_pair,
        generated,
        feature_dimensions=8,
        lexical_dimensions=128,
    )

    assert len(injected.candidates) == 4
    assert injected.candidates[-2].origin == "injected_held_out_response_a"
    assert injected.candidates[-1].origin == "injected_held_out_response_b"
    assert all(not value for value in cortex.learn_flags)


def test_independent_readout_has_exact_zero_for_one_shared_selection_and_a_ceiling() -> (  # noqa: E501
    None
):
    model = _model()
    raw_pair = _raw_pair()
    bank = SharedCandidateBank(
        raw_pair.pair_id,
        (
            _candidate(b"unrelated response", -1.0),
            _candidate(raw_pair.response_a, -2.0),
            _candidate(raw_pair.response_b, -2.0),
        ),
    )
    pair = _pair_vectors([raw_pair], model)[0]
    readouts = _bank_readouts(
        raw_pair,
        pair,
        bank,
        model,
        lexical_dimensions=128,
        interaction=True,
    )
    goal_off = _selection_readout(
        {"selected_index_a": 0, "selected_index_b": 0}, readouts
    )
    quality = _bank_quality(
        raw_pair,
        bank,
        readouts,
        min_printable_fraction=0.8,
        useful_reference_cosine=0.05,
    )

    assert goal_off["reference_symmetric_pair_margin"] == 0.0
    assert quality["reference_oracle_ceiling"]["reference_symmetric_pair_margin"] > 0.0
    assert quality["reference_proxy_useful_candidate_present_a"] is True
    assert quality["reference_proxy_useful_candidate_present_b"] is True
