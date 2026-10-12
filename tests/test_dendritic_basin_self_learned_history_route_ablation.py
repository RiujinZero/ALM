from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_self_learned_history_route_ablation as assay,
)


class FakeCortex:
    embodied_bytes = 500_000_000

    def __init__(self) -> None:
        self.history = bytearray()
        self.learn_flags: list[bool] = []

    def reset_episode(self) -> None:
        self.history.clear()

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> np.ndarray:
        self.learn_flags.append(learn)
        values = np.asarray(byte_ids, dtype=np.uint8).reshape(-1)
        self.history.extend(int(value) for value in values)
        return np.full(len(values), 1.0 / 256.0)

    def predict_next_distribution(self) -> np.ndarray:
        tail = bytes(self.history[-12:])
        favored = (
            int.from_bytes(
                hashlib.blake2b(tail or b"empty", digest_size=2).digest(), "little"
            )
            % 256
        )
        result = np.full(256, 0.25 / 255.0, dtype=np.float64)
        result[favored] = 0.75
        return result

    def advance_observation(self, byte_id: int, *, learn: bool) -> None:
        self.learn_flags.append(learn)
        self.history.append(int(byte_id))


def _range(blob: bytearray, payload: bytes) -> dict[str, object]:
    start = len(blob)
    blob.extend(payload)
    return {
        "start": start,
        "end": len(blob),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _training_row(episode: bytes, *, start: int, identifier: str) -> dict[str, object]:
    request_marker = episode.index(b"USER: ")
    request_start_local = request_marker + len(b"USER: ")
    request_end_local = episode.index(b"\n\nASSISTANT:")
    assistant_marker_local = request_end_local + 2
    response_start_local = assistant_marker_local + len(b"ASSISTANT: ")
    response_end_local = len(episode)
    return {
        "assigned_split": "train",
        "start": start,
        "end": start + len(episode),
        "episode_id": identifier,
        "sha256": hashlib.sha256(episode).hexdigest(),
        # Schema-v4 row-level request offsets are episode-local.
        "target_user_content_start": request_start_local,
        "target_user_content_end": request_end_local,
        "target_offsets": {
            "role_marker_start": start + assistant_marker_local,
            "score_window_start": start + response_start_local,
            "score_window_end": start + response_end_local,
        },
        "turns": [
            {
                "content_start": start + request_start_local,
                "end": start + request_end_local,
            },
            {
                "content_start": start + response_start_local,
                "end": start + response_end_local,
            },
        ],
    }


def _write_fixture(root: Path) -> None:
    root.mkdir()
    train_rows = []
    train_blob = bytearray()
    for index, episode in enumerate(
        (
            b"USER: alpha request\n\nASSISTANT: alpha answer\n",
            b"USER: beta request\n\nASSISTANT: beta answer\n",
        )
    ):
        start = len(train_blob)
        train_blob.extend(episode)
        train_rows.append(
            _training_row(episode, start=start, identifier=f"episode-{index}")
        )
    (root / "train.bin").write_bytes(train_blob)
    (root / "train.index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8"
    )

    common = b"\n\nASSISTANT: held\n\nUSER: common resume\n\nASSISTANT: "
    context_a = b"USER: alpha request" + common
    context_b = b"USER: beta request" + common
    response_a = b"alpha answer\n"
    response_b = b"beta answer\n"
    dev_blob = bytearray()
    ranges = {
        "context_a": _range(dev_blob, context_a),
        "context_b": _range(dev_blob, context_b),
        "response_a": _range(dev_blob, response_a),
        "response_b": _range(dev_blob, response_b),
    }
    row = {
        "schema_version": 4,
        "split": "dev",
        "pair_id": "pair-0",
        "source_label": "fixture",
        "invariants": {"valid": True},
        "blob_ranges": ranges,
        "common_suffix": {"bytes": len(common)},
        "episodes": {
            "a": {
                "target_user_content_start": len(b"USER: "),
                "target_user_content_end": len(b"USER: alpha request"),
                "unanswered_prefix_bytes": len(b"USER: alpha request"),
            },
            "b": {
                "target_user_content_start": len(b"USER: "),
                "target_user_content_end": len(b"USER: beta request"),
                "unanswered_prefix_bytes": len(b"USER: beta request"),
            },
        },
    }
    (root / "dev_pairs.bin").write_bytes(dev_blob)
    (root / "dev_pairs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

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


def _pair() -> assay.DevelopmentPair:
    common = b"\n\nASSISTANT: held\n\nUSER: common resume\n\nASSISTANT: "
    return assay.DevelopmentPair(
        pair_id="pair",
        source_label="fixture",
        context_a=b"USER: alpha request" + common,
        context_b=b"USER: beta request" + common,
        request_a=b"alpha request",
        request_b=b"beta request",
        response_a=b"alpha answer\n",
        response_b=b"beta answer\n",
    )


def _states() -> assay.PairStates:
    return assay.PairStates(
        request_motif_a=(1, 2),
        request_motif_b=(3, 4),
        native_full_a=(5, 6),
        native_full_b=(7, 8),
        native_pending_a=(9, 10),
        native_pending_b=(9, 10),
    )


def _bank() -> assay.WriterBank:
    return assay.make_writer_bank(
        state_feature_count=2,
        motif_feature_count=2,
        table_bits=4,
        learning_rate=0.2,
        local_orders=(1, 2),
    )


def test_train_loader_converts_episode_local_request_offsets(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_fixture(dataset)

    episodes = assay.load_training_episodes(dataset, response_bytes=128)

    assert len(episodes) == 2
    assert episodes[0].target_request == b"alpha request"
    assert episodes[1].target_request == b"beta request"
    assert episodes[1].prompt == b"USER: beta request\n\nASSISTANT: "
    assert episodes[1].response == b"beta answer\n"
    assert episodes[1].completed_event_ends == (len(b"USER: beta request"),)


def test_dev_loader_reads_only_causal_request_and_declared_responses(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_fixture(dataset)

    pairs = assay.load_development_pairs(dataset)

    assert len(pairs) == 1
    assert pairs[0].request_a == b"alpha request"
    assert pairs[0].request_b == b"beta request"
    assert pairs[0].response_a == b"alpha answer\n"

    try:
        assay.load_development_pairs(
            dataset, excluded_pair_ids=(pairs[0].pair_id, pairs[0].pair_id)
        )
    except ValueError as error:
        assert "must be unique" in str(error)
    else:
        raise AssertionError("duplicate excluded development pair IDs were accepted")


def test_full_context_fifo128_alias_is_recorded_and_never_fitted() -> None:
    common = bytes(range(256)) * 3
    pair = assay.DevelopmentPair(
        pair_id="alias",
        source_label="fixture",
        context_a=b"different-a" + common,
        context_b=b"different-b" + common,
        request_a=b"different-a",
        request_b=b"different-b",
        response_a=b"a",
        response_b=b"b",
    )

    report = assay.full_context_fifo_alias_report([pair])

    assert report["capacity"] == 128
    assert report["exactly_equal_pairs"] == 1
    assert report["mean_jaccard"] == 1.0
    assert report["writer_fitted"] is False
    assert assay.fifo_fourgram_feature_ids(pair.request_a, 8) != (
        assay.fifo_fourgram_feature_ids(pair.request_b, 8)
    )


def test_native_predictive_trajectory_is_causal_and_retains_slow_difference() -> None:
    cortex = FakeCortex()
    pair = _pair()

    left = assay.native_predictive_trajectory(cortex, pair.context_a, decays=(0.5, 0.9))
    right = assay.native_predictive_trajectory(
        cortex, pair.context_b, decays=(0.5, 0.9)
    )

    assert left.completed_events == right.completed_events == 3
    np.testing.assert_array_equal(left.pending_vector, right.pending_vector)
    assert not np.array_equal(left.full_vector, right.full_vector)
    assert cortex.learn_flags and not any(cortex.learn_flags)


def test_exposure_interval_updates_every_matched_writer_identically() -> None:
    episodes = [
        assay.TrainingEpisode(f"e{i}", b"", bytes([65 + i]) * 4, bytes([97 + i, 10]))
        for i in range(4)
    ]
    request_states = [(i + 1, i + 10) for i in range(4)]
    native_states = [(i + 20, i + 30) for i in range(4)]
    order = np.asarray([2, 0, 3, 1])
    shuffled = assay.receiver.sattolo_indices(4, 71)
    bank = _bank()

    report = assay.fit_exposure_interval(
        bank,
        episodes,
        request_states,
        native_states,
        shuffled,
        order,
        previous_passes=0.0,
        target_passes=0.5,
    )

    assert report["episode_exposures"] == 2
    assert {writer.episodes_seen for _, writer in bank.items()} == {2}
    assert {writer.symbols_seen for _, writer in bank.items()} == {4}
    assert set(report["writers"]) == {name for name, _ in bank.items()}
    assert all(
        row["mean_conditional_nll_nats"] < row["mean_base_nll_nats"]
        for row in report["writers"].values()
    )


def test_checkpoint_zero_is_exact_and_controlled_contrasts_are_explicit() -> None:
    bank = _bank()

    report = assay.evaluate_checkpoint(
        FakeCortex(),
        [_pair()],
        [_states()],
        bank,
        pooled_state=(11, 12),
        betas=(0.5, 1.0),
        horizon=5,
        log_ratio_clip=2.0,
        sample_seed=100,
        bootstrap_seed=101,
        bootstrap_resamples=100,
        expect_zero_route=True,
    )

    assert report["invariants"] == {
        "native_greedy_sampled_and_bpb_identical_across_betas": True,
        "checkpoint_zero_every_route_exactly_native": True,
        "writer_fingerprints_unchanged_by_evaluation": True,
    }
    assert set(report["beta_runs"]["0.5"]["controlled_contrasts"]) == set(
        assay.CONTROLLED_CONTRASTS
    )
    first = report["beta_runs"]["0.5"]["pair_rows"]["native"][0]
    second = report["beta_runs"]["1.0"]["pair_rows"]["native"][0]
    assert first["sides"] == second["sides"]
    for condition in assay.CONDITION_NAMES:
        row = report["beta_runs"]["1.0"]["pair_rows"][condition][0]
        assert row["sides"] == second["sides"]


def test_source_has_no_protected_pair_path_or_test_enable_switch() -> None:
    source = inspect.getsource(assay)

    assert "test_pairs.bin" not in source
    assert "test_pairs.jsonl" not in source
    assert "allow-test" not in source
    assert "load_crossover_pairs" not in source
    assert "dev_pairs.bin" in source


def test_tiny_end_to_end_smoke_is_json_serializable_and_marks_alias_skip(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    _write_fixture(dataset)
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"fixture checkpoint")
    monkeypatch.setattr(assay, "_load_native", lambda _: FakeCortex())
    args = SimpleNamespace(
        dataset_dir=dataset,
        checkpoint=checkpoint,
        report=tmp_path / "report.json",
        snapshot_dir=None,
        passes=(0.0,),
        state_decays=(0.5, 0.9),
        state_feature_count=2,
        motif_feature_count=2,
        table_bits=4,
        learning_rate=0.2,
        local_orders=(1, 2),
        response_bytes=16,
        beta_values=(0.5, 1.0),
        log_ratio_clip=2.0,
        generation_horizon=3,
        training_seed=1,
        sample_seed=2,
        bootstrap_seed=3,
        bootstrap_resamples=100,
        wall_limit_seconds=60.0,
        smoke=True,
        max_train_episodes=2,
        max_dev_pairs=1,
        exclude_dev_pair_ids=(),
    )

    report = assay.run_ablation(args)

    json.dumps(report)
    assert report["status"] == "complete_smoke"
    assert report["checkpoint"] == {
        "path": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "embodied_bytes": 500_000_000,
        "embodied_bytes_at_load": 500_000_000,
        "transient_embodied_bytes_at_report": 500_000_000,
    }
    assert report["representation_preflight"]["full_context_fifo128_training_skipped"]
    assert report["checkpoints"][0]["evaluation"]["invariants"][
        "checkpoint_zero_every_route_exactly_native"
    ]
    assert set(report["verified_inputs"]) == {
        "train.bin",
        "train.index.jsonl",
        "dev_pairs.bin",
        "dev_pairs.jsonl",
    }


def test_data_slices_are_rejected_without_explicit_smoke() -> None:
    args = SimpleNamespace(
        passes=(0.0,),
        bootstrap_resamples=100,
        wall_limit_seconds=60.0,
        max_train_episodes=2,
        max_dev_pairs=1,
        smoke=False,
    )

    try:
        assay.run_ablation(args)
    except ValueError as error:
        assert "only with --smoke" in str(error)
    else:
        raise AssertionError("a sliced non-smoke run was accepted")
