from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_goal_trajectory_ablation import (
    BuildConfig,
    SourceSpec,
    _dominant_script,
    _parser,
    build_dataset,
    load_conversations,
    parse_turns,
)


def _conversation(index: int, source: str) -> dict[str, str]:
    topic = f"topic-{source}-{index}"
    padding = (f" background-{index}" * 18).strip()
    salt = f"unique{source}{index:x}"

    def salted(value: str) -> str:
        # Keep synthetic rows distinct under the production word-shingle audit.
        return " ".join(f"{word}-{salt}" for word in value.split())

    turns = (
        ("USER", f"I want to work carefully on {topic}. {padding}"),
        ("ASSISTANT", f"We can establish the context for {topic}. {padding}"),
        ("USER", f"Please compare the alternatives for {topic}. {padding}"),
        ("ASSISTANT", f"The alternatives depend on the constraints. {padding}"),
        ("USER", f"What should the concrete next step be for {topic}? {padding}"),
        (
            "ASSISTANT",
            f"The next step for {topic} is to test the relevant constraint with a "
            f"small controlled trial. {padding}",
        ),
    )
    text = "\n\n".join(f"{role}: {salted(content)}" for role, content in turns)
    return {"text": text, "source": source, "conversation_id": f"{source}-{index}"}


def _write_source(path: Path, source: str, count: int = 240) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            handle.write(json.dumps(_conversation(index, source)))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _slice(blob: bytes, record: dict) -> bytes:
    return blob[record["start"] : record["end"]]


def test_parse_turns_tracks_byte_offsets() -> None:
    raw = "USER: café\n\nASSISTANT: réponse\n\nUSER: next".encode()
    turns = parse_turns(raw)
    assert [turn.role for turn in turns] == ["USER", "ASSISTANT", "USER"]
    assert raw[turns[1].marker_start : turns[1].content_start] == b"ASSISTANT: "
    assert raw[turns[1].content_start : turns[1].end] == "réponse".encode()


def test_dominant_script_is_deterministic_and_ignores_punctuation() -> None:
    assert _dominant_script(b"Please answer 123?!") == "latin"
    assert _dominant_script("Пожалуйста, ответьте 123?!".encode()) == "cyrillic"
    assert _dominant_script("请回答这个问题。".encode()) == "han"


def test_loader_preserves_exact_text_field_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    row = _conversation(0, "source")
    row["text"] = row["text"].replace("carefully", "cafe\u0301") + "  "
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = BuildConfig(
        sources=(SourceSpec("source", source),),
        output_dir=tmp_path / "unused",
        train_bytes=1_000,
        dev_pairs=0,
        test_pairs=0,
    )
    conversations, _ = load_conversations(config)
    assert len(conversations) == 1
    assert conversations[0].raw == row["text"].encode("utf-8")


def test_source_start_line_is_inclusive_bounded_and_recorded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source, "source", count=260)
    output = tmp_path / "windowed"
    start_line = 13
    scan_limit = 220
    manifest = build_dataset(
        BuildConfig(
            sources=(
                SourceSpec(
                    "source",
                    source,
                    scan_limit=scan_limit,
                    start_line=start_line,
                ),
            ),
            output_dir=output,
            train_bytes=8_000,
            dev_pairs=2,
            test_pairs=2,
            min_bridge_bytes=512,
            seed=31,
        )
    )

    assert manifest["schema_version"] == 4
    assert manifest["source_scan_policy"] == {
        "line_numbering": "zero_based",
        "start_line": "inclusive",
        "scan_limit": "maximum rows read after start_line",
    }
    source_record = manifest["sources"][0]
    assert source_record["start_line_inclusive"] == start_line
    assert source_record["end_line_exclusive"] == start_line + scan_limit
    assert source_record["scan_limit"] == scan_limit
    assert manifest["source_diagnostics"]["source"]["scanned"] == scan_limit
    assert manifest["invariants"]["selected_source_lines_within_declared_windows"]

    selected_lines = [
        row["source_line"] for row in _read_jsonl(output / "train.index.jsonl")
    ]
    for split in ("dev", "test"):
        for row in _read_jsonl(output / f"{split}_pairs.jsonl"):
            selected_lines.extend(
                row["episodes"][name]["source_line"] for name in ("a", "b", "bridge")
            )
    assert selected_lines
    assert min(selected_lines) >= start_line
    assert max(selected_lines) < start_line + scan_limit


def test_source_start_line_rejects_negative_values(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source, "source", count=4)
    with pytest.raises(ValueError, match="start lines"):
        build_dataset(
            BuildConfig(
                sources=(SourceSpec("source", source, start_line=-1),),
                output_dir=tmp_path / "out",
            )
        )


def test_cli_allows_lmsys_only_with_an_explicit_start_line(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "--lmsys",
            str(tmp_path / "lmsys.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--start-lmsys-line",
            "50000",
            "--scan-lmsys",
            "50000",
        ]
    )

    assert args.everyday is None
    assert args.start_lmsys_line == 50_000
    assert args.scan_lmsys == 50_000


def test_builder_writes_symmetric_disjoint_sidecars(tmp_path: Path) -> None:
    everyday = tmp_path / "everyday.jsonl"
    lmsys = tmp_path / "lmsys.jsonl"
    _write_source(everyday, "everyday")
    _write_source(lmsys, "lmsys")
    output = tmp_path / "out"
    config = BuildConfig(
        sources=(SourceSpec("everyday", everyday), SourceSpec("lmsys", lmsys)),
        output_dir=output,
        train_bytes=12_000,
        dev_pairs=4,
        test_pairs=4,
        min_prefix_bytes=512,
        min_bridge_bytes=512,
        score_bytes=96,
        seed=19,
    )
    manifest = build_dataset(config)
    assert manifest["selected"]["dev_pairs"] == 4
    assert manifest["selected"]["test_pairs"] == 4
    assert all(manifest["invariants"].values())

    train_blob = (output / "train.bin").read_bytes()
    train_rows = _read_jsonl(output / "train.index.jsonl")
    assert train_rows
    assert train_rows[0]["start"] == 0
    assert train_rows[-1]["end"] == len(train_blob)
    for left, right in zip(train_rows, train_rows[1:], strict=False):
        assert left["end"] == right["start"]
    for row in train_rows:
        target = row["target_offsets"]
        assert target["coordinate_space"] == "train.bin"
        assert row["start"] <= target["turn_separator_start"]
        assert target["turn_separator_start"] <= target["role_marker_start"]
        assert target["role_marker_start"] < target["content_start"]
        assert target["content_start"] < target["content_end"] <= row["end"]
        assert target["score_window_start"] == target["content_start"]
        assert target["score_window_end"] <= target["content_end"]
        response = train_blob[target["content_start"] : target["content_end"]]
        assert len(response) == row["target_full_response_bytes"]
        prefix_range = row["unanswered_prefix_range"]
        prefix = _slice(train_blob, prefix_range)
        assert len(prefix) == row["unanswered_prefix_bytes"]
        assert prefix_range["end"] == row["start"] + row["target_user_content_end"]

    split_episode_ids: dict[str, set[str]] = {
        "train": {row["episode_id"] for row in train_rows}
    }
    for split in ("dev", "test"):
        blob = (output / f"{split}_pairs.bin").read_bytes()
        rows = _read_jsonl(output / f"{split}_pairs.jsonl")
        seen: set[str] = set()
        for row in rows:
            assert all(row["invariants"].values())
            episodes = row["episodes"]
            ids = {episodes[name]["episode_id"] for name in ("a", "b", "bridge")}
            assert len(ids) == 3
            assert not (ids & seen)
            seen.update(ids)

            ranges = row["blob_ranges"]
            context_a = _slice(blob, ranges["context_a"])
            context_b = _slice(blob, ranges["context_b"])
            response_a = _slice(blob, ranges["response_a"])
            response_b = _slice(blob, ranges["response_b"])
            full_response_a = _slice(blob, ranges["full_response_a"])
            full_response_b = _slice(blob, ranges["full_response_b"])
            audit_ranges = row["audit_subranges"]
            original_prefix_a = _slice(blob, audit_ranges["original_prefix_a"])
            original_prefix_b = _slice(blob, audit_ranges["original_prefix_b"])
            target_user_a = _slice(blob, audit_ranges["target_user_a"])
            target_user_b = _slice(blob, audit_ranges["target_user_b"])
            common_bytes = row["common_suffix"]["bytes"]
            assert context_a[-common_bytes:] == context_b[-common_bytes:]
            prefix_a_bytes = (
                row["common_suffix"]["context_a_start"] - ranges["context_a"]["start"]
            )
            prefix_b_bytes = (
                row["common_suffix"]["context_b_start"] - ranges["context_b"]["start"]
            )
            assert prefix_a_bytes == episodes["a"]["unanswered_prefix_bytes"]
            assert prefix_b_bytes == episodes["b"]["unanswered_prefix_bytes"]
            assert (
                episodes["a"]["unanswered_prefix_bytes"]
                == episodes["a"]["target_user_content_end"]
            )
            assert (
                episodes["b"]["unanswered_prefix_bytes"]
                == episodes["b"]["target_user_content_end"]
            )
            assert len(response_a) == len(response_b) == row["response_score_bytes"]
            assert full_response_a.startswith(response_a)
            assert full_response_b.startswith(response_b)
            assert len(full_response_a) == episodes["a"]["target_full_response_bytes"]
            assert len(full_response_b) == episodes["b"]["target_full_response_bytes"]
            assert original_prefix_a == context_a[: len(original_prefix_a)]
            assert original_prefix_b == context_b[: len(original_prefix_b)]
            assert target_user_a in original_prefix_a
            assert target_user_b in original_prefix_b
            assert (
                episodes["a"]["target_user_script"]
                == episodes["b"]["target_user_script"]
                == row["matching"]["target_user_script"]
            )
            assert episodes["bridge"]["bridge_payload_bytes"] >= 512
            local_bytes = row["local_suffix"]["bytes"]
            assert context_a[-local_bytes:] == context_b[-local_bytes:]
            assert context_a[-local_bytes:].hex() == row["local_suffix"]["hex"]
        split_episode_ids[split] = seen

    assert not (split_episode_ids["train"] & split_episode_ids["dev"])
    assert not (split_episode_ids["train"] & split_episode_ids["test"])
    assert not (split_episode_ids["dev"] & split_episode_ids["test"])
    assert manifest["split_policy"]["fuzzy_template_or_user_cluster_split"] is False
    assert manifest["known_leakage_limitations"]


def test_content_duplicates_stay_single_and_build_is_deterministic(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_source(first, "first", count=180)
    _write_source(second, "second", count=180)
    duplicate = _conversation(0, "first")
    with second.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate) + "\n")

    base = dict(
        sources=(SourceSpec("first", first), SourceSpec("second", second)),
        train_bytes=8_000,
        dev_pairs=2,
        test_pairs=2,
        min_bridge_bytes=512,
        score_bytes=80,
        seed=7,
    )
    probe_config = BuildConfig(output_dir=tmp_path / "probe", **base)
    conversations, diagnostics = load_conversations(probe_config)
    assert len({row.episode_id for row in conversations}) == len(conversations)
    assert sum(item.get("duplicate_episode", 0) for item in diagnostics.values()) == 1

    first_output = tmp_path / "out-a"
    second_output = tmp_path / "out-b"
    build_dataset(BuildConfig(output_dir=first_output, **base))
    build_dataset(BuildConfig(output_dir=second_output, **base))
    for name in (
        "train.bin",
        "train.index.jsonl",
        "dev_pairs.bin",
        "dev_pairs.jsonl",
        "test_pairs.bin",
        "test_pairs.jsonl",
    ):
        assert (first_output / name).read_bytes() == (second_output / name).read_bytes()


def test_bridge_floor_is_not_configurable_below_512(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source, "source", count=10)
    with pytest.raises(ValueError, match="at least 512"):
        build_dataset(
            BuildConfig(
                sources=(SourceSpec("source", source),),
                output_dir=tmp_path / "out",
                train_bytes=1_000,
                dev_pairs=0,
                test_pairs=0,
                min_bridge_bytes=511,
            )
        )


def test_successor_target_is_later_trajectory_not_scored_response(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source, "source", count=240)
    output = tmp_path / "successor"
    manifest = build_dataset(
        BuildConfig(
            sources=(SourceSpec("source", source),),
            output_dir=output,
            train_bytes=12_000,
            dev_pairs=2,
            test_pairs=2,
            min_prior_exchanges=1,
            min_bridge_bytes=512,
            goal_target_kind="successor",
            seed=23,
        )
    )
    assert manifest["config"]["goal_target_kind"] == "successor"

    train_blob = (output / "train.bin").read_bytes()
    train_row = _read_jsonl(output / "train.index.jsonl")[0]
    goal_record = train_row["goal_target"]
    goal_payload = _slice(train_blob, goal_record)
    assert goal_record["kind"] == "successor"
    assert goal_record["start"] >= train_row["target_offsets"]["content_end"]
    assert [turn.role for turn in parse_turns(goal_payload)] == ["USER", "ASSISTANT"]

    pair_blob = (output / "test_pairs.bin").read_bytes()
    pair_row = _read_jsonl(output / "test_pairs.jsonl")[0]
    ranges = pair_row["blob_ranges"]
    response = _slice(pair_blob, ranges["response_a"])
    future = _slice(pair_blob, ranges["goal_target_a"])
    assert pair_row["goal_target_kind"] == "successor"
    assert response != future
    assert [turn.role for turn in parse_turns(future)] == ["USER", "ASSISTANT"]
