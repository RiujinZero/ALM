from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_goal_trajectory_leakage import (
    audit_dataset,
    load_dataset_metadata,
    main,
)


def _fingerprint(normalized: str, shingles: list[int]) -> dict[str, object]:
    return {
        "normalized_sha256": normalized,
        "word_5_shingle_bottom8_u64": shingles,
    }


def _episode(
    tag: str,
    *,
    source_line: int,
    request: str | None = None,
    response: str | None = None,
    response_payload: str | None = None,
    request_sketch: list[int] | None = None,
    response_sketch: list[int] | None = None,
    secret: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "episode_id": tag,
        "declared_source": "synthetic/source",
        "source_line": source_line,
    }
    if response_payload is not None:
        row["target_full_response_sha256"] = response_payload
    if request is not None:
        row["target_user_fingerprint"] = _fingerprint(
            request, request_sketch or []
        )
    if response is not None:
        row["target_full_response_fingerprint"] = _fingerprint(
            response, response_sketch or []
        )
    if secret is not None:
        # Deliberately unsupported raw content: the audit must ignore it.
        row["raw_text"] = secret
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_dataset(
    path: Path,
    *,
    schema_version: int,
    train: list[dict[str, object]],
    dev: list[dict[str, object]],
    test: list[dict[str, object]],
) -> None:
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"schema_version": schema_version}), encoding="utf-8"
    )
    _write_jsonl(path / "train.index.jsonl", train)
    for split, episodes in (("dev", dev), ("test", test)):
        assert len(episodes) % 3 == 0
        rows = []
        for index in range(0, len(episodes), 3):
            a, b, bridge = episodes[index : index + 3]
            rows.append({"episodes": {"a": a, "b": b, "bridge": bridge}})
        _write_jsonl(path / f"{split}_pairs.jsonl", rows)


def _unrelated_episode(tag: str, source_line: int, offset: int) -> dict[str, object]:
    return _episode(
        tag,
        source_line=source_line,
        request=f"request-{tag}",
        response=f"response-{tag}",
        response_payload=f"payload-{tag}",
        request_sketch=[offset, offset + 1],
        response_sketch=[offset + 100, offset + 101],
    )


def test_schema_v4_split_audit_finds_exact_and_sketch_overlap(
    tmp_path: Path,
) -> None:
    secret = "RAW-CONTENT-MUST-NOT-APPEAR"
    dataset = tmp_path / "v4"
    train = _episode(
        "same-episode",
        source_line=7,
        request="same-request",
        response="train-response",
        response_payload="same-payload",
        request_sketch=[1, 2, 3, 4, 5],
        response_sketch=[101, 102],
        secret=secret,
    )
    dev = [
        _episode(
            "same-episode",
            source_line=7,
            request="same-request",
            response="dev-response",
            response_payload="same-payload",
            request_sketch=[1, 2, 3, 4, 9],
            response_sketch=[201, 202],
            secret=secret,
        ),
        _unrelated_episode("dev-b", 8, 300),
        _unrelated_episode("dev-bridge", 9, 400),
    ]
    test = [
        _episode(
            "test-a",
            source_line=10,
            request="test-request",
            response="train-response",
            response_payload="test-payload",
            request_sketch=[501, 502],
            response_sketch=[101, 102],
        ),
        _unrelated_episode("test-b", 11, 600),
        _unrelated_episode("test-bridge", 12, 700),
    ]
    _write_dataset(
        dataset,
        schema_version=4,
        train=[train],
        dev=dev,
        test=test,
    )

    report = audit_dataset(load_dataset_metadata(dataset))

    assert report["primary"]["records_by_split"] == {
        "train": 1,
        "dev": 3,
        "test": 3,
    }
    train_dev = report["within_primary_split_comparisons"][0]
    assert (train_dev["left"], train_dev["right"]) == ("train", "dev")
    for field in ("episode", "source_line", "request_fingerprint"):
        assert train_dev["exact"][field]["shared_values"] == 1
    assert train_dev["exact"]["response_payload"]["shared_values"] == 1
    request_approximate = train_dev["approximate_bottom8_word5"]["request"]
    assert request_approximate["record_pairs_sharing_at_least_four"] == 1
    assert request_approximate["maximum_shared_shingles"] == 4
    train_test = report["within_primary_split_comparisons"][1]
    assert train_test["exact"]["response_fingerprint"]["shared_values"] == 1
    assert secret not in json.dumps(report)
    assert not report["summary"]["exact_overlap_free_where_comparable"]
    assert not report["summary"][
        "content_fingerprint_overlap_free_where_comparable"
    ]
    assert not report["summary"]["provenance_overlap_free_where_comparable"]


def test_prior_schema_comparison_marks_missing_fingerprints_unavailable(
    tmp_path: Path,
) -> None:
    current_dir = tmp_path / "current"
    prior_dir = tmp_path / "prior"
    current_train = _unrelated_episode("shared", 21, 10)
    current_train["target_full_response_sha256"] = "shared-payload"
    _write_dataset(
        current_dir,
        schema_version=4,
        train=[current_train],
        dev=[
            _unrelated_episode("current-dev-a", 22, 30),
            _unrelated_episode("current-dev-b", 23, 40),
            _unrelated_episode("current-dev-bridge", 24, 50),
        ],
        test=[
            _unrelated_episode("current-test-a", 25, 60),
            _unrelated_episode("current-test-b", 26, 70),
            _unrelated_episode("current-test-bridge", 27, 80),
        ],
    )
    prior_train = {
        "episode_id": "shared",
        "declared_source": "synthetic/source",
        "source_line": 21,
        "target_full_response_sha256": "shared-payload",
    }
    def plain_prior(tag: str, line: int) -> dict[str, object]:
        return {
            "episode_id": tag,
            "declared_source": "synthetic/source",
            "source_line": line,
            "target_full_response_sha256": f"payload-{tag}",
        }
    _write_dataset(
        prior_dir,
        schema_version=3,
        train=[prior_train],
        dev=[
            plain_prior("prior-dev-a", 101),
            plain_prior("prior-dev-b", 102),
            plain_prior("prior-dev-bridge", 103),
        ],
        test=[
            plain_prior("prior-test-a", 104),
            plain_prior("prior-test-b", 105),
            plain_prior("prior-test-bridge", 106),
        ],
    )

    report = audit_dataset(
        load_dataset_metadata(current_dir), [load_dataset_metadata(prior_dir)]
    )
    comparison = report["prior_dataset_comparisons"][0]

    assert comparison["right_schema_version"] == 3
    assert comparison["exact"]["episode"]["shared_values"] == 1
    assert comparison["exact"]["source_line"]["shared_values"] == 1
    assert comparison["exact"]["response_payload"]["shared_values"] == 1
    assert comparison["exact"]["request_fingerprint"]["status"] == (
        "metadata_unavailable"
    )
    assert comparison["exact"]["response_fingerprint"]["status"] == (
        "metadata_unavailable"
    )
    assert comparison["approximate_bottom8_word5"]["request"]["status"] == (
        "metadata_unavailable"
    )


def test_cli_is_sidecar_only_writes_sanitized_json_and_can_fail_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "cli-dataset"
    identifier_sentinel = "EPISODE-IDENTIFIER-SENTINEL-934817"
    shared = _unrelated_episode(identifier_sentinel, 1, 1)
    _write_dataset(
        dataset,
        schema_version=4,
        train=[shared],
        dev=[
            shared,
            _unrelated_episode("dev-b", 2, 20),
            _unrelated_episode("dev-bridge", 3, 30),
        ],
        test=[
            _unrelated_episode("test-a", 4, 40),
            _unrelated_episode("test-b", 5, 50),
            _unrelated_episode("test-bridge", 6, 60),
        ],
    )
    # No .bin files exist.  Success proves the command needs only sidecars.
    output = tmp_path / "reports" / "audit.json"
    exit_code = main(
        [str(dataset), "--output", str(output), "--fail-on-exact"]
    )

    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == json.loads(output.read_text(encoding="utf-8"))
    assert identifier_sentinel not in stdout
    assert str(dataset) not in stdout


def test_invalid_shingle_metadata_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid"
    invalid = _unrelated_episode("invalid", 1, 1)
    invalid["target_user_fingerprint"] = _fingerprint(
        "request", list(range(9))
    )
    _write_dataset(
        dataset,
        schema_version=4,
        train=[invalid],
        dev=[
            _unrelated_episode("dev-a", 2, 20),
            _unrelated_episode("dev-b", 3, 30),
            _unrelated_episode("dev-bridge", 4, 40),
        ],
        test=[
            _unrelated_episode("test-a", 5, 50),
            _unrelated_episode("test-b", 6, 60),
            _unrelated_episode("test-bridge", 7, 70),
        ],
    )

    with pytest.raises(ValueError, match="more than eight"):
        load_dataset_metadata(dataset)
