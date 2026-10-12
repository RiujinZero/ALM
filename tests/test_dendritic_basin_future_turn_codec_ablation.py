from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as assay,
)


class FakeCortex:
    embodied_bytes = 500_000_000

    def __init__(self) -> None:
        self.history = bytearray()
        self.learn_flags: list[bool] = []
        self.reset_count = 0

    def reset_episode(self) -> None:
        self.history.clear()
        self.reset_count += 1

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> np.ndarray:
        self.learn_flags.append(learn)
        values = np.asarray(byte_ids, dtype=np.uint8).reshape(-1)
        self.history.extend(int(value) for value in values)
        return np.full(len(values), 1.0 / 256.0)

    def predict_next_distribution(self) -> np.ndarray:
        tail = bytes(self.history[-8:])
        favored = (
            int.from_bytes(
                hashlib.blake2b(tail or b"empty", digest_size=2).digest(), "little"
            )
            % 256
        )
        result = np.full(256, 0.4 / 255.0, dtype=np.float64)
        result[favored] = 0.6
        return result

    def advance_observation(self, byte_id: int, *, learn: bool) -> None:
        self.learn_flags.append(learn)
        self.history.append(int(byte_id))


def _append_range(blob: bytearray, payload: bytes) -> dict[str, object]:
    start = len(blob)
    blob.extend(payload)
    return {
        "start": start,
        "end": len(blob),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _subrange(start: int, payload: bytes) -> dict[str, object]:
    return {
        "start": start,
        "end": start + len(payload),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _conversation(
    turns: list[tuple[str, bytes]], *, absolute_start: int
) -> tuple[bytes, list[dict[str, object]]]:
    payload = bytearray()
    metadata: list[dict[str, object]] = []
    for index, (role, content) in enumerate(turns):
        if index:
            payload.extend(b"\n\n")
        marker_start = absolute_start + len(payload)
        payload.extend(role.encode() + b": ")
        content_start = absolute_start + len(payload)
        payload.extend(content)
        metadata.append(
            {
                "index": index,
                "role": role,
                "marker_start": marker_start,
                "content_start": content_start,
                "end": absolute_start + len(payload),
            }
        )
    return bytes(payload), metadata


def _write_fixture(root: Path) -> None:
    root.mkdir()
    training_specs = (
        [
            ("USER", b"choose alpha"),
            ("ASSISTANT", b"alpha response one"),
            ("USER", b"continue alpha"),
            ("ASSISTANT", b"alpha response two"),
        ],
        [
            ("USER", b"choose beta"),
            ("ASSISTANT", b"beta response one"),
            ("USER", b"continue beta"),
            ("ASSISTANT", b"beta response two"),
        ],
        [
            ("USER", b"choose gamma"),
            ("ASSISTANT", b"gamma response one"),
            ("USER", b"continue gamma"),
            ("ASSISTANT", b"gamma response two"),
        ],
    )
    train_blob = bytearray()
    train_rows = []
    for index, turns in enumerate(training_specs):
        start = len(train_blob)
        episode, turn_rows = _conversation(turns, absolute_start=start)
        train_blob.extend(episode)
        train_rows.append(
            {
                "assigned_split": "train",
                "start": start,
                "end": len(train_blob),
                "episode_id": f"episode-{index}",
                "source_label": "fixture",
                "sha256": hashlib.sha256(episode).hexdigest(),
                "turns": turn_rows,
            }
        )
    (root / "train.bin").write_bytes(train_blob)
    (root / "train.index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8"
    )

    dev_blob = bytearray()
    dev_rows = []
    for pair_index, (left_word, right_word) in enumerate(
        ((b"alpha", b"beta"), (b"gamma", b"delta"))
    ):
        original_a = b"USER: choose " + left_word
        original_b = b"USER: choose " + right_word
        common = b"\n\nASSISTANT: bridge\n\nUSER: resume\n\nASSISTANT: "
        context_a = original_a + common
        context_b = original_b + common
        response_a = left_word + b" reply"
        response_b = right_word + b" reply"
        full_a = response_a + b" with complete content"
        full_b = response_b + b" with complete content"
        ranges = {
            "context_a": _append_range(dev_blob, context_a),
            "context_b": _append_range(dev_blob, context_b),
            "response_a": _append_range(dev_blob, response_a),
            "response_b": _append_range(dev_blob, response_b),
            "full_response_a": _append_range(dev_blob, full_a),
            "full_response_b": _append_range(dev_blob, full_b),
        }
        context_a_start = int(ranges["context_a"]["start"])
        context_b_start = int(ranges["context_b"]["start"])
        dev_rows.append(
            {
                "schema_version": 4,
                "split": "dev",
                "pair_id": f"fixture-{pair_index}",
                "source_label": "fixture",
                "invariants": {"valid": True},
                "blob_ranges": ranges,
                "audit_subranges": {
                    "original_prefix_a": _subrange(context_a_start, original_a),
                    "original_prefix_b": _subrange(context_b_start, original_b),
                },
                "common_suffix": {"bytes": len(common)},
                "episodes": {
                    "a": {
                        "target_user_content_start": len(b"USER: "),
                        "target_user_content_end": len(original_a),
                        "unanswered_prefix_bytes": len(original_a),
                    },
                    "b": {
                        "target_user_content_start": len(b"USER: "),
                        "target_user_content_end": len(original_b),
                        "unanswered_prefix_bytes": len(original_b),
                    },
                },
            }
        )
    (root / "dev_pairs.bin").write_bytes(dev_blob)
    (root / "dev_pairs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dev_rows), encoding="utf-8"
    )

    outputs = []
    for name in (
        "train.bin",
        "train.index.jsonl",
        "dev_pairs.bin",
        "dev_pairs.jsonl",
    ):
        payload = (root / name).read_bytes()
        outputs.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 4, "outputs": outputs}), encoding="utf-8"
    )


def _turns() -> list[assay.AssistantTurn]:
    return [
        assay.AssistantTurn(
            episode_id=f"e-{index}",
            source_label="fixture",
            turn_index=1,
            prompt=b"USER: " + label + assay.ASSISTANT_PROMPT,
            current_user_turn=label,
            content=label + b" answer content",
            content_start=0,
            content_end=len(label) + len(b" answer content"),
        )
        for index, label in enumerate((b"alpha", b"beta", b"gamma", b"delta"))
    ]


def _codec() -> assay.FutureTurnCodec:
    return assay.FutureTurnCodec.fit(
        _turns(),
        raw_dimensions=32,
        content_dimensions=4,
        active_content_features=2,
    )


def test_loaders_extract_every_completed_turn_and_both_dev_contexts(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_fixture(dataset)

    conversations = assay.load_training_conversations(dataset)
    pairs = assay.load_development_pairs(dataset, excluded_pair_ids=())

    assert len(conversations) == 3
    assert sum(len(row.assistant_turns) for row in conversations) == 6
    first = conversations[0].assistant_turns[0]
    assert first.current_user_turn == b"choose alpha"
    assert first.content == b"alpha response one"
    assert first.prompt.endswith(b"ASSISTANT: ")
    assert pairs[0].side_a.full_response.endswith(b"complete content")
    assert pairs[0].side_a.original_context == (
        b"USER: choose alpha" + assay.ASSISTANT_PROMPT
    )
    assert pairs[0].side_a.interrupted_context.endswith(b"ASSISTANT: ")


def test_last_user_turn_uses_only_strict_top_level_markers() -> None:
    context = (
        b"USER: original request\nUSER: quoted user marker"
        b"\n\nASSISTANT: bridge\nASSISTANT: quoted assistant marker"
        b"\n\nUSER: immediate request\nUSER: still quoted content"
        b"\n\nASSISTANT: "
    )

    assert assay.last_user_turn_from_context(context) == (
        b"immediate request\nUSER: still quoted content"
    )

    invalid = (
        b"preamble\n\nUSER: request\n\nASSISTANT: ",
        b"USER: request",
        b"USER: request\n\nASSISTANT: answer",
        b"USER: request\n\nUSER: repeated\n\nASSISTANT: ",
        b"USER: \n\nASSISTANT: ",
    )
    for payload in invalid:
        with pytest.raises(ValueError):
            assay.last_user_turn_from_context(payload)


def test_codec_has_dense_binding_and_fixed_sparse_binding_features() -> None:
    codec = _codec()
    positive = codec.code_from_content(
        np.asarray([3.0, 2.0, 0.0, 0.0]), role="ASSISTANT", source="fixture"
    )
    negative = codec.code_from_content(
        np.asarray([-3.0, 2.0, 0.0, 0.0]), role="ASSISTANT", source="fixture"
    )
    user = codec.code_from_content(positive.content, role="USER", source="fixture")

    assert len(positive.vector) == codec.dense_dimensions
    assert codec.persistent_feature_count == 4
    assert codec.quantize(positive) != codec.quantize(negative)
    assert codec.quantize(positive)[-1] == codec.quantize(user)[-1]
    assert codec.quantize(positive)[-2] != codec.quantize(user)[-2]
    assert not positive.vector.flags.writeable


def test_byte_sketch_uses_both_signed_hash_polarities() -> None:
    vector = assay._hashed_byte_features(
        b"alpha beta gamma delta epsilon zeta eta theta", 64
    )

    assert np.any(vector > 0.0)
    assert np.any(vector < 0.0)


def test_nonrecurrent_ridge_learns_a_direct_causal_mapping() -> None:
    x = np.asarray([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    y = np.column_stack((2.0 * x[:, 0], -3.0 * x[:, 1]))

    predictor = assay.RidgeCodePredictor.fit(x, y, alpha=1e-6)
    predicted = np.asarray([predictor.predict(row) for row in x])

    np.testing.assert_allclose(predicted, y, atol=1e-5)
    assert not hasattr(predictor, "recurrent_state")


def test_decoder_controls_are_allocation_and_active_count_matched() -> None:
    bank = assay.make_decoder_bank(
        persistent_feature_count=4,
        table_bits=4,
        learning_rate=0.2,
        local_orders=(1, 2),
    )

    audit = assay.decoder_allocation_audit(bank, 4)

    assert audit["all_weight_tables_allocation_matched"]
    assert audit["additive_and_coupled_active_count_matched"]
    assert (
        audit["raw_active_features"]["additive"]
        == audit["raw_active_features"]["coupled"]
    )


def test_gain_zero_is_exact_native_under_greedy_and_common_draws() -> None:
    bank = assay.make_decoder_bank(
        persistent_feature_count=4,
        table_bits=4,
        learning_rate=0.2,
        local_orders=(1, 2),
    )
    native = assay.RouteAssignment(None, (), 0.0)
    gain0 = assay.RouteAssignment(bank.coupled, (1, 2, 3, 4), 0.0)
    draws = np.asarray([0.1, 0.9, 0.4, 0.7])

    greedy_native = assay.generate_branch(
        FakeCortex(),
        b"USER: prompt\n\nASSISTANT: ",
        native,
        horizon=4,
        residual_clip=2.0,
        uniform_draws=None,
    )
    greedy_gain0 = assay.generate_branch(
        FakeCortex(),
        b"USER: prompt\n\nASSISTANT: ",
        gain0,
        horizon=4,
        residual_clip=2.0,
        uniform_draws=None,
    )
    sampled_native = assay.generate_branch(
        FakeCortex(),
        b"USER: prompt\n\nASSISTANT: ",
        native,
        horizon=4,
        residual_clip=2.0,
        uniform_draws=draws,
    )
    sampled_gain0 = assay.generate_branch(
        FakeCortex(),
        b"USER: prompt\n\nASSISTANT: ",
        gain0,
        horizon=4,
        residual_clip=2.0,
        uniform_draws=draws,
    )

    assert greedy_native == greedy_gain0
    assert sampled_native == sampled_gain0


def test_code_control_collision_audit_reports_ineffective_wrong_codes() -> None:
    codes = ((1,), (2,), (3,), (3,))
    shuffle = np.asarray([2, 3, 0, 1])

    audit = assay.code_control_collision_audit(codes, shuffle)

    assert audit["paired_swap_identical_code_pairs"] == 1
    assert audit["paired_swap_distinct_code_fraction"] == 0.5
    assert audit["cross_pair_shuffle_identical_code_sides"] == 0


def test_decoder_training_uses_frozen_native_and_matched_updates(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_fixture(dataset)
    conversations = assay.load_training_conversations(dataset)
    turns = [turn for row in conversations for turn in row.assistant_turns]
    codec = assay.FutureTurnCodec.fit(
        turns,
        raw_dimensions=32,
        content_dimensions=4,
        active_content_features=2,
    )
    bank = assay.make_decoder_bank(
        persistent_feature_count=codec.persistent_feature_count,
        table_bits=4,
        learning_rate=0.2,
        local_orders=(1, 2),
    )
    cortex = FakeCortex()

    report = assay.fit_decoder_bank(
        cortex,
        conversations,
        codec,
        bank,
        response_cap=5,
        training_seed=4,
    )

    assert report["passes"] == 1
    assert report["matched_episode_and_symbol_updates"]
    assert {row["episodes"] for row in report["writers"].values()} == {6}
    assert {row["symbols"] for row in report["writers"].values()} == {30}
    assert cortex.learn_flags and not any(cortex.learn_flags)


def test_source_exposes_no_protected_payload_or_split_switch() -> None:
    source = inspect.getsource(assay)

    assert "test_pairs.bin" not in source
    assert "test_pairs.jsonl" not in source
    assert "--allow-test" not in source
    assert 'add_argument("--split"' not in source
    assert "dev_pairs.bin" in source


def test_tiny_end_to_end_smoke_reports_oracle_and_causal_context_modes(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    _write_fixture(dataset)
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"fake checkpoint")
    cortex = FakeCortex()
    monkeypatch.setattr(assay, "_load_native", lambda _: cortex)
    args = SimpleNamespace(
        dataset_dir=dataset,
        checkpoint=checkpoint,
        report=tmp_path / "report.json",
        raw_dimensions=32,
        content_dimensions=4,
        active_content_features=2,
        causal_hash_dimensions=16,
        ridge_alpha=0.5,
        table_bits=4,
        learning_rate=0.2,
        local_orders=(1, 2),
        response_cap=5,
        generation_horizon=3,
        gain=1.0,
        residual_clip=2.0,
        training_seed=1,
        sample_seed=2,
        control_seed=3,
        bootstrap_seed=4,
        bootstrap_resamples=100,
        wall_limit_seconds=60.0,
        smoke=True,
        max_train_conversations=3,
        max_dev_pairs=2,
        exclude_dev_pair_ids=(),
    )

    report = assay.run_ablation(args)

    json.dumps(report)
    assert report["status"] == "complete_smoke"
    assert report["claim_scope"][
        "oracle_contains_held_out_answer_content_and_is_leakage_ceiling"
    ]
    assert report["causal_predictor"]["non_recurrent"]
    assert report["claim_scope"][
        "causal_predictor_uses_immediate_user_turn_parsed_from_context"
    ]
    assert report["claim_scope"][
        "no_evaluator_selected_old_request_is_reinjected_into_predictor"
    ]
    assert report["counts"]["completed_training_assistant_turns"] == 6
    assert report["evaluation"]["development_partition_counts"] == {
        "selection": 1,
        "evaluation": 1,
    }
    for mode in (
        "original_unanswered_prefix",
        "long_interruption_crossover",
    ):
        mode_report = report["evaluation"]["context_modes"][mode]
        assert mode_report["available"]
        assert mode_report["invariants"]["gain0_oracle_exactly_native"]
        assert mode_report["invariants"][
            "predictor_uses_immediate_user_turn_from_generation_context"
        ]
        assert mode_report["predictor_code_diagnostic"][
            "evaluator_target_request_reinjected_as_current_turn"
        ] is False
        assert set(mode_report["pair_rows"]) == set(assay.CONDITION_NAMES)
        assert mode_report["summaries"]["all"]["pairs"] == 2
    original = report["evaluation"]["context_modes"]["original_unanswered_prefix"]
    interrupted = report["evaluation"]["context_modes"][
        "long_interruption_crossover"
    ]
    original_turns = (b"choose alpha", b"choose beta", b"choose gamma", b"choose delta")
    expected_original_hash = hashlib.sha256(
        b"".join(len(turn).to_bytes(8, "little") + turn for turn in original_turns)
    ).hexdigest()
    expected_interrupted_hash = hashlib.sha256(
        b"".join(len(b"resume").to_bytes(8, "little") + b"resume" for _ in range(4))
    ).hexdigest()
    assert (
        original["predictor_code_diagnostic"]["immediate_user_turn_sha256"]
        == expected_original_hash
    )
    assert (
        interrupted["predictor_code_diagnostic"]["immediate_user_turn_sha256"]
        == expected_interrupted_hash
    )
    assert expected_original_hash != expected_interrupted_hash
    implementation = report["implementation_files"]
    source_path = Path(assay.__file__).resolve()
    source_record = implementation[
        "src/manifoldflow/experiments/dendritic_basin_future_turn_codec_ablation.py"
    ]
    assert source_record["sha256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    assert cortex.learn_flags and not any(cortex.learn_flags)


def test_unsliced_development_is_required_outside_smoke() -> None:
    args = SimpleNamespace(
        bootstrap_resamples=100,
        wall_limit_seconds=60.0,
        max_train_conversations=2,
        max_dev_pairs=1,
        smoke=False,
    )

    try:
        assay.run_ablation(args)
    except ValueError as error:
        assert "only with --smoke" in str(error)
    else:
        raise AssertionError("sliced development run was accepted")
