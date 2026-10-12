#!/usr/bin/env python
"""Build the co-adapt corpus v1: de-duplicated, de-boilerplated, finely
interleaved wiki + chat + SmolTalk-CoT byte stream.

Differences from build_mixed_conversational_byte_stream.py, driven by the
attractor-collapse diagnosis (deep-band tier-2 collapse tracked corpus
position/recency, and lmsys boilerplate density was the indicated fix):

1. Unique-only quotas. The old builder cycled sources back to their start
   to fill byte quotas, which silently manufactured duplicates. Here a
   source contributes only records not already seen (normalized-case
   64-bit hash); if it runs out of unique records before its quota, the
   shortfall is logged and the build continues with the other sources.
2. Boilerplate cap. lmsys-style templated turns ("Sure! Here's...",
   essay prompts) were the dominant-completion attractor. A single global
   counter keyed on the first 96 normalized characters of each USER and
   ASSISTANT turn caps how many accepted records may share any one turn
   prefix; later records that would exceed the cap are dropped. The first
   occurrences survive, so legitimate repeated phrasing stays.
3. Fine interleave. Sources are written round-robin in small blocks
   (default 256 KB, down from 4 MB) so no corpus region is dominated by
   one source for long.
4. SmolTalk CoT added as a first-class source: the reasoning subset's
   ASSISTANT turns carry explicit <think>...</think> deliberation, which
   the byte model can learn as its own register.

Records are the same clean {"text": ...} JSONL convention the existing
sources and model harness already use, joined with blank lines. Reads
stream straight from the Google Drive mount; only the outputs are written
locally. The build is deterministic: fixed source order, sequential reads,
no RNG anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

_RECORD_SEPARATOR = b"\n\n"
_VALID_RESERVE_FRACTION = 0.02
_TURN_PATTERN = re.compile(rb"(?:\A|\n\n)(USER|ASSISTANT):[ \t]*")
_MIN_TEXT_BYTES = 80
_MAX_TEXT_BYTES = 65_536
_MIN_ASSISTANT_BYTES = 60
_MAX_IO_RETRIES = 6
_PROGRESS_INTERVAL_BYTES = 500_000_000
_DEFAULT_REMOUNT_CMD = "sudo -n mount -t drvfs G: /mnt/g"


def _normalize(text: bytes) -> bytes:
    return b" ".join(text.lower().split())


def _digest(payload: bytes) -> int:
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8, person=b"coadapt-v1").digest(),
        "little",
    )


def _extract_text(raw_line: bytes) -> bytes | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    text = record.get("text")
    if not text:
        return None
    return text.encode("utf-8", errors="ignore")


class SourceStats:
    def __init__(self, label: str):
        self.label = label
        self.lines_read = 0
        self.records_extracted = 0
        self.accepted = 0
        self.duplicates_dropped = 0
        self.hygiene_dropped = 0
        self.boilerplate_dropped = 0
        self.bytes_written = 0
        self.exhausted = False


class SourceReader:
    """Sequential unique-record reader over one JSONL region.

    Train reads [0, size - reserve); valid reads the reserved tail. No
    cycling: exhaustion is reported, never papered over with repeats.

    Google Drive's streaming mount drops long reads with OSError(EIO), so
    every readline is guarded: the reader remembers the last good offset,
    and on an I/O error it remounts the drive, reopens, seeks back, and
    continues -- transparently, without losing its place or duplicating a
    partially read line.
    """

    def __init__(
        self,
        label: str,
        path: Path,
        *,
        is_valid: bool,
        has_turns: bool,
        remount_cmd: str,
    ):
        self.stats = SourceStats(label)
        self.path = path
        self.has_turns = has_turns
        self.remount_cmd = remount_cmd
        size = path.stat().st_size
        reserve = int(size * _VALID_RESERVE_FRACTION)
        if is_valid:
            self.region_start = max(size - reserve, 0)
            self.region_end = size
        else:
            self.region_start = 0
            self.region_end = max(size - reserve, 0)
        self.handle = path.open("rb")
        self.handle.seek(self.region_start)
        if self.region_start > 0:
            self.handle.readline()  # discard the partial straddling line
        self._last_good_pos = self.handle.tell()

    def _safe_readline(self) -> bytes | None:
        for attempt in range(_MAX_IO_RETRIES):
            try:
                line = self.handle.readline()
                self._last_good_pos = self.handle.tell()
                return line
            except OSError as error:
                wait = min(2**attempt, 30)
                print(
                    f"  [{self.stats.label}] I/O error ({error}); retry "
                    f"{attempt + 1}/{_MAX_IO_RETRIES} after remount and {wait}s",
                    flush=True,
                )
                time.sleep(wait)
                if self.remount_cmd:
                    subprocess.run(
                        self.remount_cmd,
                        shell=True,
                        check=False,
                        capture_output=True,
                    )
                self.handle.close()
                self.handle = self.path.open("rb")
                self.handle.seek(self._last_good_pos)
        raise OSError(
            f"{self.stats.label}: read failed after {_MAX_IO_RETRIES} retries"
        )

    def next_record(
        self,
        seen: set[int],
        prefix_counts: dict[int, int],
        boilerplate_cap: int,
    ) -> bytes | None:
        """Return the next acceptable unique record, or None if exhausted."""

        while True:
            if self.stats.exhausted:
                return None
            if self._last_good_pos >= self.region_end:
                self.stats.exhausted = True
                return None
            raw_line = self._safe_readline()
            if not raw_line:
                self.stats.exhausted = True
                return None
            self.stats.lines_read += 1
            if self.stats.lines_read % 2000 == 0:
                print(
                    f"  [{self.stats.label}] {self.stats.lines_read} lines, "
                    f"{self.stats.bytes_written / 1e9:.2f} GB written",
                    flush=True,
                )
            text = _extract_text(raw_line)
            if text is None:
                continue
            self.stats.records_extracted += 1
            if not (_MIN_TEXT_BYTES <= len(text) <= _MAX_TEXT_BYTES):
                self.stats.hygiene_dropped += 1
                continue
            key = _digest(_normalize(text))
            if key in seen:
                self.stats.duplicates_dropped += 1
                continue
            if self.has_turns and boilerplate_cap > 0:
                turn_prefixes = self._turn_prefixes(text)
                if turn_prefixes is None:
                    self.stats.hygiene_dropped += 1
                    continue
                if any(
                    prefix_counts.get(p, 0) >= boilerplate_cap
                    for p in turn_prefixes
                ):
                    self.stats.boilerplate_dropped += 1
                    continue
                for prefix in turn_prefixes:
                    prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            seen.add(key)
            self.stats.accepted += 1
            return text

    def _turn_prefixes(self, text: bytes) -> list[int] | None:
        """Normalized 96-byte prefixes of every turn, or None if the record
        has no usable assistant content."""

        pieces = _TURN_PATTERN.split(text)
        # split() alternates [lead, label, body, label, body, ...]
        prefixes: list[int] = []
        assistant_bytes = 0
        has_user = False
        for index in range(1, len(pieces) - 1, 2):
            label = pieces[index]
            body = pieces[index + 1]
            if label == b"ASSISTANT":
                assistant_bytes += len(body.strip())
            else:
                has_user = True
            normalized = _normalize(body)[:96]
            if normalized:
                prefixes.append(_digest(normalized))
        if not has_user or assistant_bytes < _MIN_ASSISTANT_BYTES:
            return None
        normalized_lead = _normalize(pieces[0])[:96]
        if normalized_lead:
            prefixes.append(_digest(normalized_lead))
        return prefixes

    def close(self) -> None:
        self.handle.close()


def _write_stream(
    sources: list[SourceReader],
    seen: set[int],
    prefix_counts: dict[int, int],
    remaining: dict[str, int],
    output: Path,
    block_bytes: int,
    boilerplate_cap: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as out:
        while any(remaining[s.stats.label] > 0 for s in sources):
            progressed = False
            for source in sources:
                label = source.stats.label
                if remaining[label] <= 0 or source.stats.exhausted:
                    continue
                block_written = 0
                while block_written < block_bytes and remaining[label] > 0:
                    text = source.next_record(
                        seen, prefix_counts, boilerplate_cap=boilerplate_cap
                    )
                    if text is None:
                        break
                    out.write(text)
                    out.write(_RECORD_SEPARATOR)
                    chunk = len(text) + len(_RECORD_SEPARATOR)
                    block_written += chunk
                    remaining[label] -= chunk
                    source.stats.bytes_written += chunk
                    progressed = True
            if not progressed:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki", type=Path, required=True)
    parser.add_argument("--wiki-bytes", type=int, required=True)
    parser.add_argument("--smoltalk-cot", type=Path, required=True)
    parser.add_argument("--smoltalk-cot-bytes", type=int, required=True)
    parser.add_argument("--lmsys", type=Path, required=True)
    parser.add_argument("--lmsys-bytes", type=int, required=True)
    parser.add_argument("--smoltalk-preference", type=Path, required=True)
    parser.add_argument("--smoltalk-preference-bytes", type=int, required=True)
    parser.add_argument("--smoltalk-everyday", type=Path, required=True)
    parser.add_argument("--smoltalk-everyday-bytes", type=int, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--valid-output", type=Path, required=True)
    parser.add_argument("--valid-bytes", type=int, default=20_000_000)
    parser.add_argument("--block-bytes", type=int, default=262_144)
    parser.add_argument("--boilerplate-cap", type=int, default=16)
    parser.add_argument("--remount-cmd", type=str, default=_DEFAULT_REMOUNT_CMD)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    specs = [
        ("wiki", args.wiki, args.wiki_bytes, False),
        ("smoltalk_cot", args.smoltalk_cot, args.smoltalk_cot_bytes, True),
        ("lmsys", args.lmsys, args.lmsys_bytes, True),
        (
            "smoltalk_preference",
            args.smoltalk_preference,
            args.smoltalk_preference_bytes,
            True,
        ),
        (
            "smoltalk_everyday",
            args.smoltalk_everyday,
            args.smoltalk_everyday_bytes,
            True,
        ),
    ]

    seen: set[int] = set()
    prefix_counts: dict[int, int] = {}

    print("Building TRAIN stream...", flush=True)
    train_sources = [
        SourceReader(
            label,
            path,
            is_valid=False,
            has_turns=has_turns,
            remount_cmd=args.remount_cmd,
        )
        for label, path, _quota, has_turns in specs
    ]
    train_remaining = {label: quota for label, _path, quota, _t in specs}
    _write_stream(
        train_sources,
        seen,
        prefix_counts,
        train_remaining,
        args.train_output,
        args.block_bytes,
        args.boilerplate_cap,
    )

    print("Building VALID stream...", flush=True)
    valid_quota_each = args.valid_bytes // len(specs)
    valid_sources = [
        SourceReader(
            label,
            path,
            is_valid=True,
            has_turns=has_turns,
            remount_cmd=args.remount_cmd,
        )
        for label, path, _quota, has_turns in specs
    ]
    valid_remaining = {label: valid_quota_each for label, _path, _quota, _t in specs}
    _write_stream(
        valid_sources,
        seen,
        prefix_counts,
        valid_remaining,
        args.valid_output,
        args.block_bytes,
        args.boilerplate_cap,
    )

    for source in train_sources + valid_sources:
        source.close()

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def stats_block(sources: list[SourceReader]) -> list[dict]:
        return [vars(source.stats) | {"path": str(source.path)} for source in sources]

    report = {
        "schema_version": 1,
        "experiment": "coadapt_corpus_v1",
        "config": {
            "block_bytes": args.block_bytes,
            "boilerplate_cap": args.boilerplate_cap,
            "valid_bytes": args.valid_bytes,
            "quotas": {label: quota for label, _path, quota, _t in specs},
            "min_text_bytes": _MIN_TEXT_BYTES,
            "max_text_bytes": _MAX_TEXT_BYTES,
            "min_assistant_bytes": _MIN_ASSISTANT_BYTES,
            "turn_prefix_normalization": (
                "casefold, whitespace-collapsed, first 96 bytes, blake2b-64"
            ),
        },
        "train": {
            "output": str(args.train_output),
            "sha256": file_sha256(args.train_output),
            "sources": stats_block(train_sources),
        },
        "valid": {
            "output": str(args.valid_output),
            "sha256": file_sha256(args.valid_output),
            "sources": stats_block(valid_sources),
        },
        "unique_records_seen": len(seen),
        "wall_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "train"} | {
        "train_sources": report["train"]["sources"],
        "train_sha256": report["train"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
