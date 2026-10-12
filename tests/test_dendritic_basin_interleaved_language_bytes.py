from __future__ import annotations

import json
from pathlib import Path

from manifoldflow.dendritic_basin.tasks.interleaved_language_bytes import (
    EPISODE_SEPARATOR,
    build_interleaved_stream,
    load_smoltalk_episodes,
)


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    tiny = tmp_path / "tiny.txt"
    tiny.write_bytes(b"tiny one\n<|endoftext|>\ntiny two\n<|endoftext|>\n")
    smol = tmp_path / "smol.jsonl"
    with smol.open("w", encoding="utf-8") as handle:
        for index in range(20):
            handle.write(
                json.dumps({"text": f"USER: q{index}\nASSISTANT: a{index}"}) + "\n"
            )
    return tiny, smol


def test_smoltalk_split_is_stable_and_extracts_only_text(tmp_path: Path) -> None:
    _, smol = _fixtures(tmp_path)
    first = load_smoltalk_episodes(smol, validation_fraction=0.25)
    second = load_smoltalk_episodes(smol, validation_fraction=0.25)

    assert first == second
    assert all(
        episode.endswith(EPISODE_SEPARATOR) for split in first for episode in split
    )
    assert b'"text"' not in b"".join(first[0] + first[1])


def test_interleave_has_exact_budget_and_reports_recycling(tmp_path: Path) -> None:
    tiny, smol = _fixtures(tmp_path)
    output = tmp_path / "mixed.bin"
    valid = tmp_path / "smol-valid.txt"

    report = build_interleaved_stream(
        tiny,
        smol,
        output,
        total_bytes=2_000,
        seed=5,
        validation_fraction=0.25,
        smoltalk_valid_output=valid,
    )

    assert output.stat().st_size == 2_000
    assert report.tinystories_bytes + report.smoltalk_bytes == 2_000
    assert report.tinystories_episodes > 0 and report.smoltalk_episodes > 0
    assert report.smoltalk_epochs_started > 1
    assert valid.read_bytes().endswith(EPISODE_SEPARATOR)
