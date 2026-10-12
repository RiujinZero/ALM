from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np

from manifoldflow.dendritic_basin.future_turn_trajectory import EncodedTurn
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_recurrent_codec_ablation as recurrent_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_successor_shuffle_ablation as assay,
)


@dataclass
class FakeCortex:
    observed: list[int] = field(default_factory=list)
    learn_flags: list[bool] = field(default_factory=list)

    def reset_episode(self) -> None:
        self.observed.clear()

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> None:
        self.learn_flags.append(learn)
        self.observed.extend(int(value) for value in byte_ids)

    def predict_next_distribution(self) -> np.ndarray:
        result = np.full(256, 1.0, dtype=np.float64)
        preferred = 32 if not self.observed else (self.observed[-1] + 1) % 128
        result[preferred] = 4.0
        return result / result.sum()

    def advance_observation(self, byte_id: int, *, learn: bool) -> None:
        self.learn_flags.append(learn)
        self.observed.append(int(byte_id))


def _conversation(episode_id: str, label: bytes) -> codec_assay.TrainingConversation:
    payload = (
        b"USER: choose "
        + label
        + b"\n\nASSISTANT: "
        + label
        + b" response one"
        + b"\n\nUSER: continue "
        + label
        + b"\n\nASSISTANT: "
        + label
        + b" response two"
    )
    parsed = recurrent_assay.observed_turns_from_transcript(
        payload, generation_context=False
    )
    matches = list(recurrent_assay._TOP_LEVEL_TURN_PATTERN.finditer(payload))
    assistants: list[codec_assay.AssistantTurn] = []
    for index, (turn, match) in enumerate(zip(parsed, matches, strict=True)):
        if turn.role != codec_assay.ASSISTANT_ROLE:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(payload)
        assistants.append(
            codec_assay.AssistantTurn(
                episode_id=episode_id,
                source_label="fixture",
                turn_index=index,
                prompt=payload[: match.end()],
                current_user_turn=parsed[index - 1].content,
                content=turn.content,
                content_start=match.end(),
                content_end=end,
            )
        )
    return codec_assay.TrainingConversation(
        episode_id,
        "fixture",
        payload,
        tuple(assistants),
    )


def _side(label: bytes) -> codec_assay.DevelopmentSide:
    original = b"USER: choose " + label + codec_assay.ASSISTANT_PROMPT
    interrupted = (
        b"USER: choose "
        + label
        + b"\n\nASSISTANT: remembered "
        + label
        + b"\n\nUSER: resume"
        + codec_assay.ASSISTANT_PROMPT
    )
    response = label + b" response one"
    return codec_assay.DevelopmentSide(
        interrupted_context=interrupted,
        original_context=original,
        request=b"choose " + label,
        response_window=response,
        full_response=response,
    )


def _fit_components():
    conversations = tuple(
        _conversation(f"episode-{index}", label)
        for index, label in enumerate((b"alpha", b"beta", b"gamma", b"delta"))
    )
    assistant_turns = [
        turn for conversation in conversations for turn in conversation.assistant_turns
    ]
    codec = codec_assay.FutureTurnCodec.fit(
        assistant_turns,
        raw_dimensions=32,
        content_dimensions=4,
        active_content_features=2,
    )
    sequences = recurrent_assay.encode_training_sequences(conversations, codec)
    intact = recurrent_assay.fit_recurrent_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=0.8,
        ridge=0.1,
    )
    shuffled, audit = assay.fit_successor_shuffled_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=0.8,
        ridge=0.1,
        seed=17,
    )
    return conversations, codec, sequences, intact, shuffled, audit


def _pairs() -> tuple[codec_assay.DevelopmentPair, ...]:
    return (
        codec_assay.DevelopmentPair(
            "pair-0", "fixture", "selection", _side(b"alpha"), _side(b"beta")
        ),
        codec_assay.DevelopmentPair(
            "pair-1", "fixture", "evaluation", _side(b"gamma"), _side(b"delta")
        ),
    )


def test_transition_design_is_the_exact_teacher_forced_predictor_state() -> None:
    _, _, sequences, intact, _, audit = _fit_components()

    design = assay.build_transition_design(intact, sequences)
    manual: list[np.ndarray] = []
    for sequence in sequences:
        state = np.zeros(intact.state_dim, dtype=np.float64)
        previous: np.ndarray | None = None
        for turn in sequence:
            state = intact._advance_observed(state, turn)
            if previous is not None:
                manual.append(previous)
            previous = state

    assert np.array_equal(design.predictor_states, np.stack(manual))
    assert design.predictor_states.flags.writeable is False
    assert design.target_tokens.flags.writeable is False
    assert audit["predictor_rows_modified"] == 0
    assert audit["predictor_state_sha256"] == assay._array_sha256(
        design.predictor_states
    )


def test_successor_targets_are_deranged_only_within_role_source_strata() -> None:
    _, _, sequences, intact, _, _ = _fit_components()
    design = assay.build_transition_design(intact, sequences)

    permutation, strata = assay.stratified_successor_permutation(
        design.target_turns, seed=17
    )
    shuffled = assay.shuffled_target_tokens(intact, design, permutation)

    assert not np.any(permutation == np.arange(len(permutation)))
    assert sum(strata.values()) == len(permutation)
    for recipient_index, donor_index in enumerate(permutation):
        recipient = design.target_turns[recipient_index]
        donor = design.target_turns[int(donor_index)]
        assert recipient.source_id == donor.source_id
        assert recipient.role_id == donor.role_id
        assert not np.array_equal(recipient.vector, donor.vector)

    source = intact._source_slice
    role = intact._role_slice
    identity_columns = np.r_[
        np.arange(source.start, source.stop),
        np.arange(role.start, role.stop),
        np.asarray([intact.token_dim - 1]),
    ]
    assert np.array_equal(
        shuffled[:, identity_columns], design.target_tokens[:, identity_columns]
    )
    assert not np.array_equal(shuffled, design.target_tokens)


def test_successor_derangement_handles_duplicate_content_or_fails_visibly() -> None:
    feasible = tuple(
        EncodedTurn(np.asarray([value]), source_id="text", role_id="USER")
        for value in (0.0, 0.0, 1.0, 1.0)
    )

    permutation, _ = assay.stratified_successor_permutation(feasible, seed=0)

    assert all(
        not np.array_equal(feasible[index].vector, feasible[int(donor)].vector)
        for index, donor in enumerate(permutation)
    )
    infeasible = feasible[:3] + (
        EncodedTurn(np.asarray([0.0]), source_id="text", role_id="USER"),
    )
    try:
        assay.stratified_successor_permutation(infeasible, seed=0)
    except ValueError as error:
        assert "no exact content derangement" in str(error)
    else:
        raise AssertionError("an impossible content derangement was accepted")


def test_control_observes_identical_prefix_states_but_refits_successors() -> None:
    _, codec, sequences, intact, shuffled, _ = _fit_components()

    forecast_changed = False
    for sequence in sequences:
        intact.reset()
        shuffled.reset()
        for turn in sequence[:-1]:
            intact.observe(turn)
            shuffled.observe(turn)
            assert np.array_equal(intact.snapshot().state, shuffled.snapshot().state)
        forecast_changed |= not np.array_equal(
            intact.predicted_tape().slots[0].vector,
            shuffled.predicted_tape().slots[0].vector,
        )
    intact.reset()
    shuffled.reset()
    assert forecast_changed

    matched = assay.matched_forecast_context(
        intact,
        shuffled,
        codec,
        _side(b"alpha").interrupted_context,
        source_label="fixture",
    )
    assert matched.predictor_state_sha256
    assert intact.observed_count == shuffled.observed_count == 0


def test_latent_diagnostic_never_loads_native_and_keeps_states_matched() -> None:
    _, codec, _, intact, shuffled, _ = _fit_components()

    report = assay.latent_control_diagnostic(
        _pairs(), codec, intact, shuffled
    )

    assert report["native_cortex_loaded"] is False
    for mode in report["context_modes"].values():
        assert mode["available"]
        assert mode["predictor_states_exactly_matched"]
        assert 0.0 <= mode["quantized_code_distinct_fraction"] <= 1.0
    assert intact.observed_count == shuffled.observed_count == 0


def test_native_evaluator_contains_only_minimum_sampled_arms() -> None:
    conversations, codec, _, intact, shuffled, _ = _fit_components()
    bank = codec_assay.make_decoder_bank(
        persistent_feature_count=codec.persistent_feature_count,
        table_bits=4,
        learning_rate=0.2,
        local_orders=(1, 2),
    )
    cortex = FakeCortex()
    codec_assay.fit_decoder_bank(
        cortex,
        conversations,
        codec,
        bank,
        response_cap=6,
        training_seed=7,
    )

    report = assay.evaluate_native_control(
        cortex,
        _pairs(),
        codec,
        intact,
        shuffled,
        bank,
        gain=1.0,
        residual_clip=2.0,
        horizon=3,
        sample_seed=11,
        bootstrap_seed=12,
        bootstrap_resamples=100,
    )

    assert tuple(assay.CONDITION_NAMES) == (
        "native",
        "recurrent_code_gain0",
        "recurrent_code_coupled",
        "successor_shuffled_code_coupled",
    )
    for mode in report["context_modes"].values():
        assert mode["available"]
        assert tuple(mode["pair_rows"]) == assay.CONDITION_NAMES
        assert mode["summaries"]["all"]["pairs"] == 2
        assert mode["invariants"]["gain0_recurrent_exactly_native"]
        assert mode["invariants"][
            "observed_prefix_predictor_states_identical_across_transitions"
        ]
    assert report["writer_fingerprints_unchanged_by_evaluation"]
    assert report["transition_fingerprints_unchanged_by_evaluation"]
    assert cortex.learn_flags and not any(cortex.learn_flags)


def test_source_has_no_protected_payload_or_split_switch() -> None:
    source = inspect.getsource(assay)
    forbidden = ("test_" + "pairs.bin", "test_" + "pairs.jsonl")

    assert all(value not in source for value in forbidden)
    assert "--allow-test" not in source
    assert 'add_argument("--split"' not in source
    assert '"dev"' in source
    assert "load_development_pairs" in source
