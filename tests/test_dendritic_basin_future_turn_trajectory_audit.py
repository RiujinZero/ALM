from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from manifoldflow.experiments.dendritic_basin_future_turn_trajectory_audit import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    ENGINEERING_EXPOSED_PAIR_IDS,
    OPENED_DATA_FILES,
    _bootstrap_mean_interval,
    _decode_raw_turn,
    _turns_from_marked_payload,
    lexical_event_vector,
    load_development_data,
    run_audit,
)

_TURN = re.compile(rb"(?m)^(USER|ASSISTANT):[ \t]*")


def test_empty_observed_turn_has_a_defined_event_code() -> None:
    turn = _decode_raw_turn("ASSISTANT", b"   \n")

    assert turn.text == ""
    vector = lexical_event_vector(turn.text)
    assert vector.shape == (96,)
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) > 0.0


def test_top_level_parser_does_not_split_embedded_role_lines() -> None:
    payload = (
        b"\n\nUSER: ASSISTANT: quoted setup\nUSER: quoted request"
        b"\n\nASSISTANT: actual answer"
    )

    turns = _turns_from_marked_payload(
        payload, label="embedded role fixture", expected_count=2
    )

    assert [turn.role for turn in turns] == ["USER", "ASSISTANT"]
    assert "USER: quoted request" in turns[0].text


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _range(start: int, payload: bytes) -> dict[str, object]:
    return {
        "start": start,
        "end": start + len(payload),
        "bytes": len(payload),
        "sha256": _sha(payload),
    }


def _conversation_bytes(turns: list[tuple[str, str]]) -> bytes:
    return "\n\n".join(f"{role}: {text}" for role, text in turns).encode()


def _turn_rows(payload: bytes, offset: int) -> list[dict[str, object]]:
    matches = list(_TURN.finditer(payload))
    result: list[dict[str, object]] = []
    for index, match in enumerate(matches):
        end = len(payload) if index + 1 == len(matches) else matches[index + 1].start()
        while end > match.end() and payload[end - 1 : end] in {b"\r", b"\n"}:
            end -= 1
        result.append(
            {
                "index": index,
                "role": match.group(1).decode(),
                "content_start": offset + match.end(),
                "end": offset + end,
            }
        )
    return result


def _append(blob: bytearray, payload: bytes) -> dict[str, object]:
    start = len(blob)
    blob.extend(payload)
    return _range(start, payload)


def _build_fake_dataset(
    root: Path, *, schema_version: int = 4
) -> dict[str, str]:
    root.mkdir()
    train_blob = bytearray()
    train_rows: list[dict[str, object]] = []
    for index in range(12):
        branch = "amber" if index % 2 == 0 else "violet"
        turns = [
            ("USER", f"establish {branch} objective case {index}"),
            ("ASSISTANT", f"acknowledge {branch} objective case {index}"),
            ("USER", f"inspect {branch} evidence case {index}"),
            ("ASSISTANT", f"explain {branch} evidence case {index}"),
            ("USER", f"verify {branch} outcome case {index}"),
            ("ASSISTANT", f"complete {branch} outcome case {index}"),
        ]
        payload = _conversation_bytes(turns)
        record = _append(train_blob, payload)
        train_rows.append(
            {
                **record,
                "assigned_split": "train",
                "episode_bytes": len(payload),
                "episode_id": _sha(payload + bytes([index])),
                "goal_target_kind": "successor",
                "goal_target_turn_end": len(turns) - 1,
                "turns": _turn_rows(payload, int(record["start"])),
            }
        )
    train_index = (
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train_rows)
    ).encode()

    dev_blob = bytearray()
    pair_ids = (*ENGINEERING_EXPOSED_PAIR_IDS, "unexposed-development-pair")
    dev_rows: list[dict[str, object]] = []
    for pair_index, pair_id in enumerate(pair_ids):
        records: dict[str, dict[str, object]] = {}
        metadata: dict[str, dict[str, object]] = {}
        for side, branch in (("a", "amber"), ("b", "violet")):
            prefix = _conversation_bytes(
                [
                    ("USER", f"establish {branch} objective dev {pair_index}"),
                    ("ASSISTANT", f"acknowledge {branch} objective dev {pair_index}"),
                    ("USER", "what should happen next"),
                ]
            )
            response = f"explain {branch} evidence dev {pair_index}".encode()
            successor = _conversation_bytes(
                [
                    ("USER", f"inspect {branch} consequence dev {pair_index}"),
                    ("ASSISTANT", f"verify {branch} consequence dev {pair_index}"),
                    ("USER", f"close {branch} objective dev {pair_index}"),
                ]
            )
            records[f"prefix_{side}"] = _append(dev_blob, prefix)
            records[f"response_{side}"] = _append(dev_blob, response)
            records[f"goal_{side}"] = _append(dev_blob, successor)
            metadata[side] = {
                "assigned_split": "dev",
                "episode_id": f"episode-{pair_index}-{side}",
                "goal_target_kind": "successor",
                "target_user_turn_index": 2,
                "goal_target_turn_start": 4,
                "goal_target_turn_end": 6,
                "unanswered_prefix_bytes": len(prefix),
                "unanswered_prefix_sha256": _sha(prefix),
                "target_full_response_sha256": _sha(response),
                "goal_target_sha256": _sha(successor),
            }
        dev_rows.append(
            {
                "schema_version": schema_version,
                "split": "dev",
                "goal_target_kind": "successor",
                "pair_id": pair_id,
                "invariants": {"same_local_suffix": True, "distinct_history": True},
                "audit_subranges": {
                    "original_prefix_a": records["prefix_a"],
                    "original_prefix_b": records["prefix_b"],
                },
                "blob_ranges": {
                    "full_response_a": records["response_a"],
                    "full_response_b": records["response_b"],
                    "goal_target_a": records["goal_a"],
                    "goal_target_b": records["goal_b"],
                },
                "episodes": metadata,
            }
        )
    dev_index = (
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dev_rows)
    ).encode()

    payloads = {
        "train.bin": bytes(train_blob),
        "train.index.jsonl": train_index,
        "dev_pairs.bin": bytes(dev_blob),
        "dev_pairs.jsonl": dev_index,
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return {name: _sha(payload) for name, payload in payloads.items()}


def test_tiny_dev_only_audit_fits_all_adjacent_turns_and_recurses(
    tmp_path: Path,
) -> None:
    hashes = _build_fake_dataset(tmp_path / "fixture")

    report = run_audit(
        tmp_path / "fixture",
        locked_hashes=hashes,
        event_dimensions=8,
        horizon_widths=(8, 4, 2, 1),
        ridge=0.1,
    )

    dataset = report["dataset"]
    assert dataset["opened_data_files"] == list(OPENED_DATA_FILES)
    assert dataset["locked_sha256_verified"] == hashes
    assert dataset["train_conversations"] == 12
    assert dataset["train_turns"] == 72
    assert dataset["adjacent_training_transitions"] == 60
    assert dataset["every_adjacent_transition_in_full_training_conversations_used"]
    assert dataset["development_pairs_available"] == 3
    assert set(dataset["engineering_exposed_pair_ids_found_and_excluded"]) == set(
        ENGINEERING_EXPOSED_PAIR_IDS
    )
    assert dataset["development_pairs_evaluated"] == 1
    assert dataset["development_sides_by_observed_future_horizon"] == {
        "1": 2,
        "2": 2,
        "3": 2,
        "4": 2,
    }
    assert dataset["protected_payload_accessed"] is False
    assert dataset["split_selector_present"] is False

    model = report["model"]
    assert model["training_objective"] == "observed_one_step_ridge"
    assert model["trains_through_recursive_rollout"] is False
    assert model["learned_transition_operators"] == 1
    assert model["horizon_widths"] == [8, 4, 2, 1]

    conditions = report["conditions"]
    assert set(conditions) == {
        "recurrent_full_history",
        "mean_event_prior",
        "current_turn_only_reset_history",
        "shuffled_transition",
        "role_swapped_prefix",
        "reversed_horizon_order",
        "oracle_projection_ceiling",
    }
    recurrent = conditions["recurrent_full_history"]["by_horizon"]
    applications = [
        recurrent[str(h)]["recursive_transition_applications"]
        for h in range(1, 5)
    ]
    assert applications == [
        1,
        2,
        3,
        4,
    ]
    assert all(recurrent[str(h)]["count"] == 2 for h in range(1, 5))
    assert recurrent["1"]["retrieval_rank"]["candidate_count"] == 2
    assert recurrent["1"]["pair_discrimination"]["pair_count"] == 1
    assert recurrent["1"]["pair_discrimination"][
        "per_pair_matched_minus_crossed_cosine"
    ][0]["pair_id"] == "unexposed-development-pair"
    assert (
        conditions["oracle_projection_ceiling"]["classification"]
        == "noncausal_projection_ceiling"
    )
    assert report["validity"]["predicted_tapes_receive_observed_future"] is False

    residual_scale = report[
        "accumulated_residual_variance_heuristic_by_horizon"
    ]
    totals = [
        residual_scale[str(h)]["mean_accumulated_residual_variance_sum"]
        for h in range(1, 5)
    ]
    assert all(right >= left for left, right in zip(totals, totals[1:], strict=False))
    assert set(report["transition_loss_by_horizon"]) == {"1", "2", "3", "4"}

    inference = report["inferential_contrasts"]
    assert inference["resampling_unit"] == "development pair"
    assert inference["base_seed"] == DEFAULT_BOOTSTRAP_SEED
    assert inference["resamples"] == DEFAULT_BOOTSTRAP_RESAMPLES
    assert inference["oracle_projection_ceiling_excluded_from_causal_contrasts"]
    control_contrasts = inference["recurrent_full_history_minus_controls"]
    assert set(control_contrasts) == {
        "mean_event_prior",
        "current_turn_only_reset_history",
        "shuffled_transition",
        "role_swapped_prefix",
        "reversed_horizon_order",
    }
    for condition in control_contrasts.values():
        for horizon in range(1, 5):
            contrast = condition["by_horizon"][str(horizon)]
            assert contrast["pair_count"] == 1
            assert contrast["per_pair_margin_difference"][0]["pair_id"] == (
                "unexposed-development-pair"
            )
            assert contrast["bootstrap_resamples"] == 4_000
            assert contrast["ci95_pair_bootstrap"] == [
                contrast["mean_pair_margin_difference"],
                contrast["mean_pair_margin_difference"],
            ]
    degradation = inference["recurrent_horizon_k_minus_horizon_1"]
    assert set(degradation) == {"2", "3", "4"}
    assert all(row["pair_count"] == 1 for row in degradation.values())


def test_pair_bootstrap_is_fixed_reproducible_and_rejects_small_budget() -> None:
    values = np.asarray([-0.4, 0.1, 0.2, 0.9])
    first = _bootstrap_mean_interval(values, seed=73, resamples=4_000)
    second = _bootstrap_mean_interval(values, seed=73, resamples=4_000)

    assert first == second
    assert first[0] <= float(np.mean(values)) <= first[1]
    with pytest.raises(ValueError, match="at least 4000"):
        _bootstrap_mean_interval(values, seed=73, resamples=3_999)


def test_loader_rejects_hash_drift_before_parsing(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    hashes = _build_fake_dataset(root)
    hashes["train.bin"] = "0" * 64

    with pytest.raises(ValueError, match="locked checksum mismatch for train.bin"):
        load_development_data(root, locked_hashes=hashes)


def test_loader_requires_schema_four_and_both_smoke_exclusions(
    tmp_path: Path,
) -> None:
    wrong_schema = tmp_path / "wrong-schema"
    hashes = _build_fake_dataset(wrong_schema, schema_version=3)
    with pytest.raises(ValueError, match="schema version 4"):
        load_development_data(wrong_schema, locked_hashes=hashes)

    missing_smoke = tmp_path / "missing-smoke"
    hashes = _build_fake_dataset(missing_smoke)
    rows = [
        json.loads(line)
        for line in (missing_smoke / "dev_pairs.jsonl").read_text().splitlines()
    ]
    rows = [row for row in rows if row["pair_id"] != ENGINEERING_EXPOSED_PAIR_IDS[1]]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (missing_smoke / "dev_pairs.jsonl").write_bytes(payload)
    hashes["dev_pairs.jsonl"] = _sha(payload)
    with pytest.raises(ValueError, match="both engineering-exposed"):
        load_development_data(missing_smoke, locked_hashes=hashes)


def test_lexical_adapter_is_deterministic_unicode_and_role_free() -> None:
    composed = lexical_event_vector("Caf\u00e9 future trajectory", 96)
    decomposed = lexical_event_vector("Cafe\u0301 future trajectory", 96)
    different = lexical_event_vector("completely different event", 96)

    np.testing.assert_array_equal(composed, decomposed)
    assert composed.shape == (96,)
    assert np.isclose(np.linalg.norm(composed), 1.0)
    assert not np.array_equal(composed, different)
    with pytest.raises(ValueError, match="at least eight"):
        lexical_event_vector("too small", 7)


def test_runner_has_no_protected_filename_or_split_switch() -> None:
    module = sys.modules[run_audit.__module__]
    source = inspect.getsource(module)
    assert "test_pairs" not in source
    parser = module._parser()
    destinations = {action.dest for action in parser._actions}
    assert "split" not in destinations
    assert tuple(OPENED_DATA_FILES) == (
        "train.bin",
        "train.index.jsonl",
        "dev_pairs.bin",
        "dev_pairs.jsonl",
    )
