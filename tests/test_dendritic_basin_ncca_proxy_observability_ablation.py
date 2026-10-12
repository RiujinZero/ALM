from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_ncca_proxy_observability_ablation as assay,
)

TURN_PATTERN = re.compile(rb"(?:^|\n\n)(USER|ASSISTANT):[ \t]*")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _turns(raw: bytes, start: int) -> list[dict[str, int | str]]:
    matches = list(TURN_PATTERN.finditer(raw))
    result: list[dict[str, int | str]] = []
    for index, match in enumerate(matches):
        separator = match.start()
        marker = separator + (2 if raw[separator : separator + 2] == b"\n\n" else 0)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        result.append(
            {
                "index": index,
                "role": match.group(1).decode("ascii"),
                "separator_start": start + separator,
                "marker_start": start + marker,
                "content_start": start + match.end(),
                "end": start + end,
            }
        )
    return result


def _write_training_dataset(
    directory: Path, conversations: list[tuple[str, bytes]], *, schema: int = 4
) -> None:
    directory.mkdir(parents=True)
    blob = bytearray()
    rows: list[dict[str, object]] = []
    for source_line, (episode_id, raw) in enumerate(conversations, 1):
        start = len(blob)
        blob.extend(raw)
        end = len(blob)
        rows.append(
            {
                "episode_id": episode_id,
                "episode_bytes": len(raw),
                "assigned_split": "train",
                "source_label": "synthetic",
                "declared_source": "synthetic/source",
                "source_path": f"synthetic/{episode_id}.jsonl",
                "source_line": source_line,
                "source_byte_offset": start,
                "split_key": f"synthetic:{episode_id}",
                "split_key_kind": "synthetic_fixture",
                "start": start,
                "end": end,
                "sha256": _sha256(raw),
                "turns": _turns(raw, start),
            }
        )
    binary_path = directory / assay.TRAIN_BINARY_NAME
    index_path = directory / assay.TRAIN_INDEX_NAME
    binary_path.write_bytes(bytes(blob))
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": schema,
        "outputs": [
            {
                "path": binary_path.name,
                "bytes": binary_path.stat().st_size,
                "sha256": _sha256(binary_path.read_bytes()),
            },
            {
                "path": index_path.name,
                "bytes": index_path.stat().st_size,
                "sha256": _sha256(index_path.read_bytes()),
            },
            {
                "path": "sealed_payload.bin",
                "bytes": 999,
                "sha256": "0" * 64,
            },
        ],
    }
    (directory / assay.MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def _refresh_manifest_output(directory: Path, name: str) -> None:
    manifest_path = directory / assay.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = directory / name
    record = next(row for row in manifest["outputs"] if row["path"] == name)
    record["bytes"] = path.stat().st_size
    record["sha256"] = _sha256(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _small_config(**changes: object) -> assay.AssayConfig:
    values: dict[str, object] = {
        "feature_dimensions": 16,
        "state_decays": (0.5, 0.85, 0.97),
        "state_feature_count": 8,
        "response_bytes": 32,
        "next_user_bytes": 32,
        "pca_dimensions": 2,
        "initial_actions": 2,
        "min_action_support": 1,
        "kmeans_iterations": 10,
        "bootstrap_resamples": 100,
        "table_bits": 7,
    }
    values.update(changes)
    return assay.AssayConfig(**values)


def _correlated_conversations(count: int = 90) -> list[tuple[str, bytes]]:
    rows: list[tuple[str, bytes]] = []
    for index in range(count):
        if index % 2:
            response = b"omega omega omega omega"
            outcome = b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        else:
            response = b"alpha alpha alpha alpha"
            outcome = b"rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"
        raw = b"USER: choose the route\n\nASSISTANT: " + response
        raw += b"\n\nUSER: " + outcome
        rows.append((f"episode-{index:04d}", raw))
    return rows


def test_extracts_every_transition_with_absolute_ranges(tmp_path: Path) -> None:
    first = (
        b"USER: first caf\xc3\xa9\n\nASSISTANT: one\n\nUSER: second"
        b"\n\nASSISTANT: two\n\nUSER: outcome"
    )
    second = b"USER: another\n\nASSISTANT: answer\n\nUSER: consequence"
    dataset = tmp_path / "dataset"
    _write_training_dataset(dataset, [("one", first), ("two", second)])

    loaded = assay.load_training_transitions(dataset, _small_config())

    assert len(loaded.transitions) == 3
    first_rows = [row for row in loaded.transitions if row.conversation_id == "one"]
    assert [row.assistant_turn_index for row in first_rows] == [1, 3]
    assert first_rows[0].next_assistant_turn_index == 3
    assert first_rows[0].next_response == b"two"
    assert first_rows[0].next_response_start is not None
    assert first_rows[0].next_response_end is not None
    assert (
        first[first_rows[0].next_response_start : first_rows[0].next_response_end]
        == b"two"
    )
    assert not first_rows[0].terminal_after_next_user
    assert first_rows[1].terminal_after_next_user
    assert first_rows[1].next_assistant_turn_index is None
    assert first_rows[0].response == b"one"
    assert first_rows[0].next_user == b"second"
    assert first_rows[0].history_end <= first_rows[0].response_start
    assert first_rows[0].response_end <= first_rows[0].next_user_start
    assert loaded.transitions[-1].history_start == len(first)
    assert loaded.extraction["transitions_terminal_after_next_user"] == 2
    assert (
        "next_assistant_response_range"
        in loaded.extraction["transition_index_hashed_fields"]
    )
    assert "source_line" in loaded.extraction["transition_index_hashed_fields"]
    assert loaded.provenance["opened_input_basenames"] == list(assay.TRAIN_INPUT_NAMES)


def test_loader_rejects_wrong_schema_and_hash_mismatch(tmp_path: Path) -> None:
    raw = b"USER: q\n\nASSISTANT: a\n\nUSER: u"
    wrong_schema = tmp_path / "wrong-schema"
    _write_training_dataset(wrong_schema, [("one", raw)], schema=3)
    with pytest.raises(ValueError, match="schema_version"):
        assay.load_training_transitions(wrong_schema, _small_config())

    wrong_hash = tmp_path / "wrong-hash"
    _write_training_dataset(wrong_hash, [("one", raw)])
    (wrong_hash / assay.TRAIN_BINARY_NAME).write_bytes(raw + b"x")
    with pytest.raises(ValueError, match="byte count mismatch"):
        assay.load_training_transitions(wrong_hash, _small_config())

    same_size_hash = tmp_path / "same-size-hash"
    _write_training_dataset(same_size_hash, [("one", raw)])
    corrupted = bytearray((same_size_hash / assay.TRAIN_BINARY_NAME).read_bytes())
    corrupted[-1] ^= 1
    (same_size_hash / assay.TRAIN_BINARY_NAME).write_bytes(corrupted)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        assay.load_training_transitions(same_size_hash, _small_config())


def test_loader_validates_raw_role_markers_not_only_index_metadata(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    raw = b"USER: q\n\nASSISTANT: a\n\nUSER: u"
    _write_training_dataset(dataset, [("one", raw)])
    index_path = dataset / assay.TRAIN_INDEX_NAME
    row = json.loads(index_path.read_text(encoding="utf-8"))
    row["turns"][1]["role"] = "USER"
    index_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_output(dataset, assay.TRAIN_INDEX_NAME)

    with pytest.raises(ValueError, match="disagrees with raw role marker"):
        assay.load_training_transitions(dataset, _small_config())


def test_training_firewall_has_only_explicit_training_inputs() -> None:
    source = inspect.getsource(assay)
    assert assay.TRAIN_INPUT_NAMES == (
        "manifest.json",
        "train.bin",
        "train.index.jsonl",
    )
    assert "load_crossover_pairs" not in source
    assert "allow-test" not in source
    assert "dev_pairs.bin" not in source
    assert "test_pairs.bin" not in source
    parser = assay._parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_loader_never_opens_present_pair_payload_decoys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    raw = b"USER: q\n\nASSISTANT: a\n\nUSER: u"
    _write_training_dataset(dataset, [("one", raw)])
    decoys = {
        "dev_pairs.bin",
        "dev_pairs.jsonl",
        "test_pairs.bin",
        "test_pairs.jsonl",
    }
    for name in decoys:
        (dataset / name).write_bytes(b"must-not-open")

    opened: list[str] = []
    original_open = Path.open

    def tracked_open(path: Path, *args: object, **kwargs: object) -> object:
        opened.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    loaded = assay.load_training_transitions(dataset, _small_config())

    assert decoys.isdisjoint(opened)
    assert set(opened) <= set(assay.TRAIN_INPUT_NAMES)
    assert loaded.provenance["opened_input_basenames"] == list(assay.TRAIN_INPUT_NAMES)


def test_conversation_folds_are_deterministic_stratified_and_disjoint(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_training_dataset(dataset, _correlated_conversations(18))
    rows = assay.load_training_transitions(dataset, _small_config()).transitions

    first = assay.assign_conversation_folds(rows, seed=7)
    second = assay.assign_conversation_folds(tuple(reversed(rows)), seed=7)

    assert first == second
    assert set(first.values()) == set(assay.FOLD_NAMES)
    counts = Counter(first.values())
    assert max(counts.values()) - min(counts.values()) <= 1
    for conversation in first:
        observed = {
            first[row.conversation_id]
            for row in rows
            if row.conversation_id == conversation
        }
        assert len(observed) == 1


def test_tokenizer_is_outcome_blind_and_meets_fold_support(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_training_dataset(dataset, _correlated_conversations(90))
    config = _small_config()
    rows = assay.load_training_transitions(dataset, config).transitions
    folds = assay.assign_conversation_folds(rows, seed=config.fold_seed)

    responses = tuple(row.response for row in rows)
    fold_names = tuple(folds[row.conversation_id] for row in rows)
    first, labels = assay.fit_action_tokenizer(responses, fold_names, config)
    changed_outcomes = tuple(
        replace(row, next_user=b"unrelated", next_user_sha256="f" * 64) for row in rows
    )
    second, changed_labels = assay.fit_action_tokenizer(
        tuple(row.response for row in changed_outcomes),
        fold_names,
        config,
    )

    assert tuple(inspect.signature(assay.fit_action_tokenizer).parameters) == (
        "responses",
        "fold_names",
        "config",
    )
    assert first.action_count == 2
    assert np.all(first.fold_support >= 1)
    assert np.array_equal(labels, changed_labels)
    assert np.array_equal(first.components, second.components)
    assert np.array_equal(first.initial_centroids, second.initial_centroids)


def test_spherical_kmeans_returns_labels_for_final_centers() -> None:
    values = assay._normalize_rows(
        np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [-1.0, 0.0],
                [-0.6, -0.4],
                [0.0, 1.0],
            ]
        )
    )
    centers, labels = assay._spherical_kmeans(
        values,
        clusters=2,
        iterations=1,
        seed=11,
    )
    assert np.array_equal(labels, np.argmax(values @ centers.T, axis=1))


def test_support_merging_and_singleton_shuffle_failure_are_explicit() -> None:
    centroids = np.eye(4, dtype=np.float64)
    labels = np.asarray([0, 0, 0, 1, 2, 3, 3, 3, 3], dtype=np.int64)
    folds = ["A", "B", "C", "A", "B", "A", "B", "C", "C"]
    mapping, support, history, mapped = assay._merge_for_fold_support(
        centroids, labels, folds, 1
    )
    assert len(history) >= 1
    assert support.shape[1] >= 2
    assert np.all(support >= 1)
    assert np.array_equal(mapped, mapping[labels])

    row = assay.Transition(
        decision_id="d",
        conversation_id="c",
        assistant_turn_index=1,
        source_label="source",
        script="latin",
        horizon_bin=0,
        response_length_bin=0,
        history_start=0,
        history_end=1,
        response_start=2,
        response_end=3,
        response_window_end=3,
        next_user_start=4,
        next_user_end=5,
        next_user_window_end=5,
        next_history_end=5,
        next_assistant_turn_index=None,
        terminal_after_next_user=True,
        history_sha256="0" * 64,
        response_sha256="1" * 64,
        next_user_sha256="2" * 64,
        response_window_sha256="1" * 64,
        next_user_window_sha256="2" * 64,
        next_history_sha256="3" * 64,
        response=b"r",
        next_user=b"u",
        evidence=np.ones(32),
    )
    with pytest.raises(ValueError, match="singleton"):
        assay.stratified_shuffled_actions([row], [0], seed=3)


def test_stratified_shuffle_has_no_row_fixed_points() -> None:
    base = assay.Transition(
        decision_id="",
        conversation_id="",
        assistant_turn_index=1,
        source_label="source",
        script="latin",
        horizon_bin=0,
        response_length_bin=0,
        history_start=0,
        history_end=1,
        response_start=2,
        response_end=3,
        response_window_end=3,
        next_user_start=4,
        next_user_end=5,
        next_user_window_end=5,
        next_history_end=5,
        next_assistant_turn_index=None,
        terminal_after_next_user=True,
        history_sha256="0" * 64,
        response_sha256="1" * 64,
        next_user_sha256="2" * 64,
        response_window_sha256="1" * 64,
        next_user_window_sha256="2" * 64,
        next_history_sha256="3" * 64,
        response=b"r",
        next_user=b"u",
        evidence=np.ones(32),
    )
    rows = [
        replace(base, decision_id=f"d{index}", conversation_id=f"c{index}")
        for index in range(8)
    ]
    actions = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    first, audit = assay.stratified_shuffled_actions(rows, actions, seed=9)
    second, _ = assay.stratified_shuffled_actions(rows, actions, seed=9)
    assert np.array_equal(first, second)
    assert audit["row_index_fixed_points"] == 0
    assert audit["minimum_stratum_rows"] == 8


def test_persistent_slots_and_writer_allocations_are_exactly_matched() -> None:
    state = tuple(range(8))
    null = assay.persistent_slots(state, None)
    action = assay.persistent_slots(state, 3)
    assert len(null) == len(action) == 17
    assert null[:8] == action[:8]
    assert null[8:] != action[8:]

    config = _small_config()
    writers = [assay._writer(config) for _ in range(3)]
    assert {writer.parameter_count for writer in writers} == {256 * (1 << 7)}
    assert {writer.raw_active_feature_count(17) for writer in writers} == {107}


def test_conversation_bootstrap_is_deterministic_and_clustered() -> None:
    rows = [
        assay.ScoredTransition(
            conversation_id=f"c{conversation}",
            decision_id=f"d{conversation}-{decision}",
            action=decision % 2,
            target_bytes=10,
            bits_h=80.0,
            bits_ha=50.0,
            bits_shuffle=70.0,
        )
        for conversation in range(6)
        for decision in range(2)
    ]
    first = assay.conversation_cluster_bootstrap(rows, seed=4, resamples=200)
    second = assay.conversation_cluster_bootstrap(rows, seed=4, resamples=200)
    assert first == second
    assert first["conversation_count"] == 6
    assert first["transition_count"] == 12
    assert first["q_HA_improvement_over_q_H"]["point_estimate_bits_per_byte"] == 3.0
    assert (
        first["q_HA_improvement_over_stratified_shuffle"][
            "point_estimate_bits_per_byte"
        ]
        == 2.0
    )
    assert first["q_HA_improvement_over_q_H"]["bootstrap_ci95_above_zero"]
    assert first["q_HA_improvement_over_stratified_shuffle"][
        "bootstrap_ci95_above_zero"
    ]
    assert not first["enough_conversations_for_gate"]


def test_bootstrap_uses_ratio_of_summed_bits_for_heterogeneous_sizes() -> None:
    rows = [
        assay.ScoredTransition("small", "d0", 0, 1, 18.0, 8.0, 13.0),
        assay.ScoredTransition("large", "d1", 1, 9, 72.0, 72.0, 72.0),
    ]
    result = assay.conversation_cluster_bootstrap(rows, seed=7, resamples=200)
    assert result["target_bytes"] == 10
    assert result["bpb"] == {"q_H": 9.0, "q_HA": 8.0, "q_stratified_shuffle": 8.5}
    assert result["q_HA_improvement_over_q_H"]["point_estimate_bits_per_byte"] == 1.0

    with pytest.raises(ValueError, match="at least two conversations"):
        assay.conversation_cluster_bootstrap(rows[:1], seed=7, resamples=200)


def test_small_table_full_synthetic_run_is_always_selection_ineligible(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_training_dataset(dataset, _correlated_conversations())
    config = _small_config()
    checkpoints: list[tuple[str, int, int, tuple[str, ...]]] = []

    def hook(
        rotation: str,
        rows_seen: int,
        bytes_seen: int,
        snapshots: dict[str, object],
    ) -> None:
        checkpoints.append((rotation, rows_seen, bytes_seen, tuple(sorted(snapshots))))

    report = assay.run_assay(dataset, config, checkpoint_hook=hook)

    assert report["schema_version"] == 1
    assert len(report["rotations"]) == 3
    assert report["protocol_eligibility"]["diagnostic_small_table"]
    assert not report["protocol_eligibility"]["selection_eligible"]
    assert report["gate_zero"]["statistical_pattern_passed"]
    assert not report["gate_zero"]["passed"]
    assert len(checkpoints) == 12
    assert all(row["audit"]["checkpoint_hook_invoked"] for row in report["rotations"])
    assert all(report["invariants"].values())
    for rotation in report["rotations"]:
        bootstrap = rotation["bootstrap"]
        assert bootstrap["enough_conversations_for_gate"]
        assert (
            bootstrap["q_HA_improvement_over_q_H"]["point_estimate_bits_per_byte"] > 0.0
        )
        assert bootstrap["q_HA_improvement_over_q_H"]["bootstrap_ci95_above_zero"]
        assert (
            bootstrap["q_HA_improvement_over_stratified_shuffle"][
                "point_estimate_bits_per_byte"
            ]
            > 0.0
        )
        assert bootstrap["q_HA_improvement_over_stratified_shuffle"][
            "bootstrap_ci95_above_zero"
        ]
    assert report["scientific_contract"][
        "downstream_primary_endpoint_is_free_generation_not_this_bpb"
    ]


def test_default_protocol_is_eligible_but_diagnostic_override_never_is() -> None:
    locked = assay.protocol_eligibility(assay.AssayConfig())
    diagnostic = assay.protocol_eligibility(
        assay.AssayConfig(table_bits=assay.LOCKED_TABLE_BITS - 1)
    )
    assert locked["selection_eligible"]
    assert not diagnostic["selection_eligible"]
    assert (
        "diagnostic_small_table_never_selection_eligible"
        in diagnostic["ineligible_reasons"]
    )
