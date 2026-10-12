from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_recurrent_codec_ablation as assay,
)


@dataclass
class FakeCortex:
    observed: list[int] = field(default_factory=list)
    learn_flags: list[bool] = field(default_factory=list)

    def reset_episode(self) -> None:
        self.observed.clear()

    def observe_chunk_probabilities(self, byte_ids: np.ndarray, *, learn: bool) -> None:
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
    parsed = assay.observed_turns_from_transcript(payload, generation_context=False)
    matches = list(assay._TOP_LEVEL_TURN_PATTERN.finditer(payload))
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
    sequences = assay.encode_training_sequences(conversations, codec)
    recurrent = assay.fit_recurrent_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=0.8,
        ridge=0.1,
    )
    one_shot, _, _ = assay.fit_one_shot_predictor(
        assistant_turns,
        codec,
        hash_dimensions=16,
        alpha=0.5,
    )
    return conversations, codec, sequences, recurrent, one_shot


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


def test_strict_parser_preserves_exact_content_and_drops_generation_slot() -> None:
    context = (
        b"USER: request\nUSER: quoted marker"
        b"\n\nASSISTANT: answer\nASSISTANT: quoted marker"
        b"\n\nUSER: follow up"
        b"\n\nASSISTANT: "
    )

    turns = assay.observed_turns_from_transcript(context, generation_context=True)

    assert [turn.role for turn in turns] == ["USER", "ASSISTANT", "USER"]
    assert turns[0].content == b"request\nUSER: quoted marker"
    assert turns[1].content == b"answer\nASSISTANT: quoted marker"
    assert turns[-1].content == b"follow up"

    invalid = (
        b"preamble\n\nUSER: request\n\nASSISTANT: ",
        b"USER: request",
        b"USER: request\n\nASSISTANT: nonempty",
        b"USER: one\n\nUSER: two\n\nASSISTANT: ",
    )
    for payload in invalid:
        try:
            assay.observed_turns_from_transcript(payload, generation_context=True)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid generation transcript was accepted")


def test_training_sequences_use_every_prior_role_tagged_vector() -> None:
    conversations, codec, sequences, recurrent, _ = _fit_components()

    assert len(sequences) == len(conversations) == 4
    assert {len(sequence) for sequence in sequences} == {4}
    assert [turn.role_id for turn in sequences[0]] == [
        "USER",
        "ASSISTANT",
        "USER",
        "ASSISTANT",
    ]
    assert recurrent.state_dim == 24

    context = (
        b"USER: choose alpha"
        b"\n\nASSISTANT: alpha response one"
        b"\n\nUSER: continue alpha"
        b"\n\nASSISTANT: "
    )
    full = assay.forecast_context(recurrent, codec, context, source_label="fixture")
    lesion = assay.forecast_context(
        recurrent,
        codec,
        context,
        source_label="fixture",
        current_user_only=True,
    )

    assert full.observed_events == 3
    assert lesion.observed_events == 1
    assert full.predicted_role == lesion.predicted_role == "ASSISTANT"
    assert not np.array_equal(full.vector, lesion.vector)
    assert recurrent.observed_count == 0


def test_recurrent_free_generation_evaluator_covers_controls_and_stays_frozen() -> None:
    conversations, codec, _, recurrent, one_shot = _fit_components()
    pairs = (
        codec_assay.DevelopmentPair(
            "pair-0", "fixture", "selection", _side(b"alpha"), _side(b"beta")
        ),
        codec_assay.DevelopmentPair(
            "pair-1", "fixture", "evaluation", _side(b"gamma"), _side(b"delta")
        ),
    )
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
    writer_before = {name: writer.fingerprint() for name, writer in bank.items()}
    recurrent_before = recurrent.fingerprint(include_state=False)

    report = assay.evaluate_recurrent_codec(
        cortex,
        pairs,
        codec,
        one_shot,
        recurrent,
        bank,
        causal_hash_dimensions=16,
        gain=1.0,
        residual_clip=2.0,
        horizon=3,
        sample_seed=11,
        control_seed=12,
        transition_shuffle_seed=13,
        bootstrap_seed=14,
        bootstrap_resamples=100,
    )

    assert report["writer_fingerprints_unchanged_by_evaluation"]
    assert report["recurrent_transition_unchanged_by_evaluation"]
    assert report["transient_recurrent_state_reset_after_evaluation"]
    assert recurrent.fingerprint(include_state=False) == recurrent_before
    assert {
        name: writer.fingerprint() for name, writer in bank.items()
    } == writer_before
    assert recurrent.observed_count == 0
    assert cortex.learn_flags and not any(cortex.learn_flags)
    assert report["development_partition_counts"] == {
        "selection": 1,
        "evaluation": 1,
    }
    for mode in (
        "original_unanswered_prefix",
        "long_interruption_crossover",
    ):
        mode_report = report["context_modes"][mode]
        assert mode_report["available"]
        assert tuple(mode_report["pair_rows"]) == assay.CONDITION_NAMES
        assert mode_report["summaries"]["all"]["pairs"] == 2
        assert mode_report["invariants"]["gain0_oracle_exactly_native"]
        assert mode_report["invariants"][
            "full_history_is_sequence_of_prior_role_tagged_vectors"
        ]
        assert mode_report["invariants"][
            "native_oracle_draw_identifier_matches_codec_ceiling_assay"
        ]
        assert set(mode_report["forecast_diagnostics"]) == {
            "one_shot",
            "recurrent_full_history",
            "recurrent_current_user_only",
            "recurrent_transition_shuffled",
        }


def test_source_has_no_protected_payload_or_split_switch() -> None:
    source = inspect.getsource(assay)
    forbidden = ("test_" + "pairs.bin", "test_" + "pairs.jsonl")

    assert all(value not in source for value in forbidden)
    assert "--allow-test" not in source
    assert 'add_argument("--split"' not in source
    assert "load_development_pairs" in source


def test_native_runtime_provenance_records_loaded_binary(
    tmp_path: Path, monkeypatch
) -> None:
    binary = tmp_path / "libalm_native.so"
    binary.write_bytes(b"native-test-binary")
    library = SimpleNamespace(_name=str(binary), alm_version=lambda: b"alm-native test")
    monkeypatch.setenv("ALM_NATIVE_DLL", str(binary))

    provenance = assay.native_runtime_provenance(SimpleNamespace(_library=library))

    assert provenance == {
        "resolved_path": str(binary.resolve()),
        "bytes": len(b"native-test-binary"),
        "sha256": hashlib.sha256(b"native-test-binary").hexdigest(),
        "version": "alm-native test",
        "alm_native_dll_override_set": True,
        "loaded_via_override": True,
    }
