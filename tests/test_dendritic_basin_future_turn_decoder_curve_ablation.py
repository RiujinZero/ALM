from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_decoder_curve_ablation as assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_recurrent_codec_ablation as recurrent_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_successor_shuffle_ablation as successor_assay,
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
        result = np.ones(256, dtype=np.float64)
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
        + b" answer one"
        + b"\n\nUSER: continue "
        + label
        + b"\n\nASSISTANT: "
        + label
        + b" answer two"
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
        episode_id, "fixture", payload, tuple(assistants)
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
    response = label + b" answer one"
    return codec_assay.DevelopmentSide(
        interrupted_context=interrupted,
        original_context=original,
        request=b"choose " + label,
        response_window=response,
        full_response=response,
    )


def _pairs() -> tuple[codec_assay.DevelopmentPair, ...]:
    return (
        codec_assay.DevelopmentPair(
            "pair-0", "fixture", "selection", _side(b"alpha"), _side(b"beta")
        ),
        codec_assay.DevelopmentPair(
            "pair-1", "fixture", "evaluation", _side(b"gamma"), _side(b"delta")
        ),
    )


def _components():
    labels = (b"alpha", b"beta", b"gamma", b"delta", b"epsilon", b"zeta")
    conversations = tuple(
        _conversation(f"episode-{index}", label)
        for index, label in enumerate(labels)
    )
    assistants = [
        turn for conversation in conversations for turn in conversation.assistant_turns
    ]
    codec = codec_assay.FutureTurnCodec.fit(
        assistants,
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
    shuffled, _ = successor_assay.fit_successor_shuffled_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=0.8,
        ridge=0.1,
        seed=17,
    )
    return conversations, codec, sequences, intact, shuffled


def test_exposure_and_fold_schedules_are_strict_and_balanced() -> None:
    conversations, _, _, _, _ = _components()

    assert assay._strict_exposures((1, 2, 4, 8)) == (1, 2, 4, 8)
    for invalid in ((), (0, 1), (1, 1), (1, 4, 2), (1, 64)):
        try:
            assay._strict_exposures(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid exposure schedule accepted: {invalid}")

    assignment = assay.conversation_fold_assignment(
        conversations, folds=3, seed=9
    )
    counts = [int(np.sum(assignment == fold)) for fold in range(3)]
    assert max(counts) - min(counts) <= 1
    assert assignment.flags.writeable is False
    assert np.array_equal(
        assignment,
        assay.conversation_fold_assignment(conversations, folds=3, seed=9),
    )


def test_cross_fitted_codes_cover_each_held_out_conversation_once() -> None:
    conversations, codec, sequences, _, _ = _components()

    result = assay.cross_fitted_training_codes(
        conversations,
        sequences,
        codec,
        folds=3,
        fold_seed=9,
        successor_shuffle_seed=30,
        memory_decay=0.8,
        ridge=0.1,
    )

    assert len(result.intact) == len(conversations)
    assert len(result.successor_shuffled) == len(conversations)
    assert all(
        len(codes) == len(conversation.assistant_turns)
        for conversation, codes in zip(conversations, result.intact, strict=True)
    )
    assert result.audit["assistant_codes"] == sum(
        len(conversation.assistant_turns) for conversation in conversations
    )
    assert sum(result.audit["fold_counts"].values()) == len(conversations)
    assert len(result.audit["folds_detail"]) == 3
    assert 0.0 <= result.audit[
        "intact_vs_successor_shuffled_distinct_fraction"
    ] <= 1.0
    assert (
        result.audit["feature_support"][
            "mean_intact_oracle_shared_content_features"
        ]
        >= 0.0
    )


def test_receiver_pass_matches_all_bytes_and_never_trains_native() -> None:
    conversations, codec, _, _, _ = _components()
    oracle = assay.oracle_training_codes(conversations, codec)
    writers = {
        name: assay._new_coupled_writer(
            persistent_feature_count=codec.persistent_feature_count,
            table_bits=5,
            learning_rate=0.2,
            local_orders=(1, 2),
        )
        for name in assay.TRACK_NAMES
    }
    codes = {name: oracle for name in assay.TRACK_NAMES}
    cortex = FakeCortex()

    report = assay.fit_receiver_pass(
        cortex,
        conversations,
        writers,
        codes,
        response_cap=8,
        training_seed=7,
    )

    totals = list(report["writers"].values())
    assert report["matched_episode_symbol_and_base_stream"]
    assert len({row["episodes"] for row in totals}) == 1
    assert len({row["symbols"] for row in totals}) == 1
    assert len({row["base_nll_nats"] for row in totals}) == 1
    assert all(row["symbol_updates"] == row["symbols"] for row in totals)
    assert all(row["raw_feature_activations"] > 0 for row in totals)
    assert all(row["unique_feature_row_updates"] > 0 for row in totals)
    assert all(
        row["scalar_weight_slots_touched"]
        == 256 * row["unique_feature_row_updates"]
        for row in totals
    )
    assert cortex.learn_flags and not any(cortex.learn_flags)
    assert len({writer.fingerprint() for writer in writers.values()}) == 1

    with pytest.raises(ValueError, match="response cap must be positive"):
        assay.fit_receiver_pass(
            FakeCortex(),
            conversations,
            writers,
            codes,
            response_cap=0,
            training_seed=7,
        )


def test_cumulative_curve_snapshots_are_independent_and_accounted() -> None:
    conversations, codec, sequences, _, _ = _components()
    cross_fit = assay.cross_fitted_training_codes(
        conversations,
        sequences,
        codec,
        folds=3,
        fold_seed=9,
        successor_shuffle_seed=30,
        memory_decay=0.8,
        ridge=0.1,
    )
    cortex = FakeCortex()
    curve = assay.fit_receiver_curve(
        cortex,
        conversations,
        codec,
        cross_fit,
        exposures=(1, 2),
        table_bits=5,
        learning_rate=0.2,
        local_orders=(1, 2),
        response_cap=8,
        training_seed=7,
    )

    assert tuple(curve.checkpoints) == (1, 2)
    assert curve.training["maximum_completed_passes"] == 2
    assert len(curve.training["per_pass"]) == 2
    assert all(
        curve.checkpoints[1][name].symbols_seen * 2
        == curve.checkpoints[2][name].symbols_seen
        for name in assay.TRACK_NAMES
    )
    assert all(
        curve.checkpoints[1][name].fingerprint()
        != curve.checkpoints[2][name].fingerprint()
        for name in assay.TRACK_NAMES
    )
    before = curve.checkpoints[1]["legacy_oracle"].fingerprint()
    state = assay.oracle_training_codes(conversations, codec)[0][0]
    curve.checkpoints[2]["legacy_oracle"].fit_episode(b"x", state)
    assert curve.checkpoints[1]["legacy_oracle"].fingerprint() == before


def test_curve_evaluation_generates_dynamic_arms_and_keeps_models_frozen() -> None:
    conversations, codec, sequences, intact, shuffled = _components()
    cross_fit = assay.cross_fitted_training_codes(
        conversations,
        sequences,
        codec,
        folds=3,
        fold_seed=9,
        successor_shuffle_seed=30,
        memory_decay=0.8,
        ridge=0.1,
    )
    cortex = FakeCortex()
    curve = assay.fit_receiver_curve(
        cortex,
        conversations,
        codec,
        cross_fit,
        exposures=(1, 2),
        table_bits=5,
        learning_rate=0.2,
        local_orders=(1, 2),
        response_cap=8,
        training_seed=7,
    )
    before = {
        exposure: {
            name: writer.fingerprint() for name, writer in writers.items()
        }
        for exposure, writers in curve.checkpoints.items()
    }

    report = assay.evaluate_receiver_curve(
        cortex,
        _pairs(),
        codec,
        intact,
        shuffled,
        curve,
        exposures=(1, 2),
        gain=1.0,
        residual_clip=2.0,
        horizon=3,
        sample_seed=11,
        bootstrap_seed=12,
        bootstrap_resamples=100,
    )

    expected = assay._evaluation_arm_names((1, 2))
    for mode in report["context_modes"].values():
        assert mode["available"]
        assert tuple(mode["pair_rows"]) == expected
        assert mode["invariants"]["gain0_exactly_native"]
        assert (
            0.0
            <= mode["code_distinctness_by_partition"]["evaluation"][
                "intact_paired_code_distinct_fraction"
            ]
            <= 1.0
        )
        for track in ("oof_intact", "oof_successor_shuffled"):
            support = mode["development_code_support_against_receiver_training"][
                track
            ]["evaluation"]
            assert support["unseen_feature_occurrences"] >= 0
        assert mode["summaries"]["all"]["pairs"] == 2
        contrasts = mode["summaries"]["all"]["controlled_contrasts"]
        assert "oof_p1_intact_minus_paired_swap" in contrasts
        assert "oof_p2_intact_minus_paired_swap" in contrasts
        assert "oof_p2_minus_p1_paired_swap_contrast_change" in contrasts
    after = {
        exposure: {
            name: writer.fingerprint() for name, writer in writers.items()
        }
        for exposure, writers in curve.checkpoints.items()
    }
    assert before == after
    assert report["writer_fingerprints_unchanged_by_evaluation"]
    assert report["transition_fingerprints_unchanged_by_evaluation"]
    assert cortex.learn_flags and not any(cortex.learn_flags)


def test_reference_replay_mapping_detects_exact_and_modified_rows() -> None:
    row = {
        "pair_id": "p",
        "partition": "evaluation",
        "sampled_margin": 0.0,
        "mean_teacher_forced_bpb": 1.0,
        "sides": {"a": {}, "b": {}},
    }
    current_rows = {
        name: [row.copy()]
        for name in (
            "native",
            "gain0",
            "legacy_p1_intact",
            "legacy_p1_successor_shuffled",
        )
    }
    prior_rows = {
        name: [row.copy()]
        for name in (
            "native",
            "recurrent_code_gain0",
            "recurrent_code_coupled",
            "successor_shuffled_code_coupled",
        )
    }
    current = {
        "context_modes": {
            mode: {"available": True, "pair_rows": current_rows}
            for mode in (
                "original_unanswered_prefix",
                "long_interruption_crossover",
            )
        }
    }
    reference = {
        "evaluation": {
            "context_modes": {
                mode: {"available": True, "pair_rows": prior_rows}
                for mode in (
                    "original_unanswered_prefix",
                    "long_interruption_crossover",
                )
            }
        }
    }

    assert assay.compare_pass1_reference_replay(current, reference)["passed"]
    current_rows["legacy_p1_intact"][0]["sampled_margin"] = 1.0
    assert not assay.compare_pass1_reference_replay(current, reference)["passed"]


def test_undertraining_gate_uses_same_writer_specificity_curve() -> None:
    positive = {"mean": 0.2, "ci95_low": 0.1, "ci95_high": 0.3}
    health = {
        "mean_printable_fraction": 1.0,
        "valid_utf8_fraction": 1.0,
        "outputs_with_run_at_least_16": 0,
    }
    contrasts = {
        "oof_matched_p8_intact_minus_successor_shuffled": {
            "sampled_free_generation_margin_difference": positive
        },
        "oof_p8_minus_p1_paired_swap_contrast_change": {
            "sampled_free_generation_difference_in_differences": positive
        },
        "oof_p8_intact_minus_paired_swap": {
            "sampled_free_generation_margin_difference": positive
        },
    }
    evaluation = {
        "context_modes": {
            "original_unanswered_prefix": {
                "code_distinctness_by_partition": {
                    "evaluation": {
                        "intact_vs_successor_shuffled_code_distinct_fraction": 1.0,
                        "intact_paired_code_distinct_fraction": 1.0,
                    }
                },
                "summaries": {
                    "evaluation": {
                        "controlled_contrasts": contrasts,
                        "arms": {
                            "native": {"sampled_degeneration": health},
                            "oof_p8_intact": {"sampled_degeneration": health},
                        },
                    }
                },
            }
        }
    }

    gate = assay.compute_development_gate(
        evaluation, {"passed": True}, final_exposure=8
    )
    assert gate["predicted_code_receiver_transmission_demonstrated"]
    assert gate["receiver_undertraining_demonstrated"]

    contrasts["oof_p8_minus_p1_paired_swap_contrast_change"] = {
        "sampled_free_generation_difference_in_differences": {
            "mean": 0.2,
            "ci95_low": 0.0,
            "ci95_high": 0.3,
        }
    }
    gate = assay.compute_development_gate(
        evaluation, {"passed": True}, final_exposure=8
    )
    assert gate["predicted_code_receiver_transmission_demonstrated"]
    assert not gate["receiver_undertraining_demonstrated"]


def test_source_has_no_protected_payload_switch_or_backpropagation() -> None:
    source = inspect.getsource(assay)
    forbidden_payloads = ("test_" + "pairs.bin", "test_" + "pairs.jsonl")

    assert all(value not in source for value in forbidden_payloads)
    assert "--allow-test" not in source
    assert 'add_argument("--split"' not in source
    assert "torch" not in source
    assert "autograd" in source
    assert "backpropagation" in source
    assert "load_development_pairs" in source
