"""Interleave wiki + chat corpora into one mixed byte stream for the
conversational-ability prototype run.

Each source is a JSON-lines file, one record per line, ``{"text": ...}``
(the same clean format ``wikipedia_en.jsonl`` and the chat exports already
use -- no markup to strip, just JSON extraction). Sources are interleaved
in blocks throughout the whole output, not concatenated end-to-end: this
architecture prunes/forgets under memory pressure, so continuous mixed
exposure matters more here than it would for a transformer trained with
random-shuffled batches.

Shorter sources (lmsys, smoltalk) are cycled (looped back to their own
start) to keep contributing their share for the full requested length;
wiki is effectively unlimited at the sizes used here so it never needs to
loop. Train reads from the front of each source file; valid reads from a
reserved region near each source's end, so the two splits stay disjoint at
the record level without a slow full-file line count first.

Usage::

    python scripts/build_mixed_conversational_byte_stream.py \\
        --wiki "C:/path/to/gdrive/training_data/knowledge/wikipedia_en.jsonl" --wiki-bytes 9000000000 \\
        --lmsys "C:/path/to/gdrive/training_data/chat/lmsys_chat_1m.jsonl" --lmsys-bytes 6000000000 \\
        --smoltalk-preference "C:/path/to/gdrive/training_data/chat/smoltalk_preference.jsonl" --smoltalk-preference-bytes 3000000000 \\
        --smoltalk-everyday "C:/path/to/gdrive/training_data/chat/smoltalk_everyday.jsonl" --smoltalk-everyday-bytes 100000000 \\
        --train-output runs/mixed_conversational_train.txt \\
        --valid-output runs/mixed_conversational_valid.txt --valid-bytes 2000000 \\
        --block-bytes 4000000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_RECORD_SEPARATOR = b"\n\n"
_VALID_RESERVE_FRACTION = 0.02  # last 2% of each source file is valid-only territory


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


class SourceReader:
    """Cycling reader over one JSONL source's train region (or a fixed
    valid region), yielding extracted ``text`` bytes, looping back to the
    region's start once exhausted."""

    def __init__(self, path: Path, *, is_valid: bool):
        self.path = path
        size = path.stat().st_size
        reserve = int(size * _VALID_RESERVE_FRACTION)
        if is_valid:
            self.region_start = max(size - reserve, 0)
            self.region_end = size
        else:
            self.region_start = 0
            self.region_end = max(size - reserve, 0)
        self.handle = path.open("rb")
        self._seek_to_region_start()

    def _seek_to_region_start(self) -> None:
        self.handle.seek(self.region_start)
        if self.region_start > 0:
            self.handle.readline()  # discard partial line straddling the seek point

    def next_record(self) -> bytes:
        while True:
            pos = self.handle.tell()
            if pos >= self.region_end:
                self._seek_to_region_start()
                continue
            raw_line = self.handle.readline()
            if not raw_line:
                self._seek_to_region_start()
                continue
            text = _extract_text(raw_line)
            if text is not None:
                return text

    def close(self) -> None:
        self.handle.close()


def _write_mixed_stream(
    sources: list[tuple[str, SourceReader, int]],  # (label, reader, byte quota)
    output: Path,
    block_bytes: int,
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    written_per_source = {label: 0 for label, _, _ in sources}
    with output.open("wb") as out:
        remaining = {label: quota for label, _, quota in sources}
        while any(remaining[label] > 0 for label, _, _ in sources):
            for label, reader, _ in sources:
                if remaining[label] <= 0:
                    continue
                block_target = min(block_bytes, remaining[label])
                block_written = 0
                while block_written < block_target:
                    text = reader.next_record()
                    out.write(text)
                    out.write(_RECORD_SEPARATOR)
                    n = len(text) + len(_RECORD_SEPARATOR)
                    block_written += n
                    written_per_source[label] += n
                remaining[label] -= block_written
    return written_per_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki", type=Path, required=True)
    parser.add_argument("--wiki-bytes", type=int, required=True)
    parser.add_argument("--lmsys", type=Path, required=True)
    parser.add_argument("--lmsys-bytes", type=int, required=True)
    parser.add_argument("--smoltalk-preference", type=Path, required=True)
    parser.add_argument("--smoltalk-preference-bytes", type=int, required=True)
    parser.add_argument("--smoltalk-everyday", type=Path, required=True)
    parser.add_argument("--smoltalk-everyday-bytes", type=int, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--valid-output", type=Path, required=True)
    parser.add_argument("--valid-bytes", type=int, default=2_000_000)
    parser.add_argument("--block-bytes", type=int, default=4_000_000)
    args = parser.parse_args()

    source_specs = [
        ("wiki", args.wiki, args.wiki_bytes),
        ("lmsys", args.lmsys, args.lmsys_bytes),
        ("smoltalk_preference", args.smoltalk_preference, args.smoltalk_preference_bytes),
        ("smoltalk_everyday", args.smoltalk_everyday, args.smoltalk_everyday_bytes),
    ]

    print("Building TRAIN stream...")
    train_sources = [
        (label, SourceReader(path, is_valid=False), quota) for label, path, quota in source_specs
    ]
    train_written = _write_mixed_stream(train_sources, args.train_output, args.block_bytes)
    for label, reader, _ in train_sources:
        reader.close()
    print(f"  wrote {args.train_output} ({sum(train_written.values()):,} bytes)")
    for label, n in train_written.items():
        print(f"    {label}: {n:,} bytes")

    print("Building VALID stream...")
    valid_quota_each = args.valid_bytes // len(source_specs)
    valid_sources = [
        (label, SourceReader(path, is_valid=True), valid_quota_each)
        for label, path, _ in source_specs
    ]
    valid_written = _write_mixed_stream(valid_sources, args.valid_output, args.block_bytes)
    for label, reader, _ in valid_sources:
        reader.close()
    print(f"  wrote {args.valid_output} ({sum(valid_written.values()):,} bytes)")
    for label, n in valid_written.items():
        print(f"    {label}: {n:,} bytes")


if __name__ == "__main__":
    main()
