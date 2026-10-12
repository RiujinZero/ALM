"""Flatten the ``knowledge/wikipedia_en.jsonl`` corpus into plain byte streams.

Supersedes ``build_wikipedia_byte_stream.py`` for Wikipedia specifically:
that script targets a WikiExtractor shard tree (``training_data/Wikipedia/``)
which turned out to be a mostly-empty partial local sync (12 of 388 shard
directories populated, ~1.08GB total). ``training_data/knowledge/`` instead
holds the real corpus as one 20GB JSON-lines file, one article per line::

    {"text": "Anarchism is a political philosophy ...", "source": "..."}

Each line's ``text`` field is already clean prose -- no ``<doc>`` wrapper,
no residual markup in the sampled records -- so there is no tag-stripping
step here, just JSON extraction and concatenation. Articles are joined with
a blank line, matching the paragraph spacing already inside each article's
own text.

Train and validation read from opposite ends of the file (train from the
front, valid from a reserved region at the tail) so the split is disjoint
at the article level without needing a slow full-file line count first --
the file is only opened for streaming, byte offsets place each read.

Usage::

    python scripts/build_wikipedia_jsonl_byte_stream.py \\
        --source "C:/path/to/gdrive/training_data/knowledge/wikipedia_en.jsonl" \\
        --train-output runs/wiki_train.txt --train-bytes 1900000000 \\
        --valid-output runs/wiki_valid.txt --valid-bytes 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ARTICLE_SEPARATOR = b"\n\n"


def _write_from_offset(
    source: Path, start_byte: int, stop_byte: int | None, output: Path, target_bytes: int
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open("rb") as handle, output.open("wb") as out:
        handle.seek(start_byte)
        if start_byte > 0:
            handle.readline()  # discard the partial line straddling the seek point
        for raw_line in handle:
            if written >= target_bytes:
                break
            if stop_byte is not None and handle.tell() > stop_byte:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            if not text:
                continue
            body = text.encode("utf-8") + _ARTICLE_SEPARATOR
            out.write(body)
            written += len(body)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--train-bytes", type=int, required=True)
    parser.add_argument("--valid-output", type=Path, required=True)
    parser.add_argument("--valid-bytes", type=int, required=True)
    parser.add_argument(
        "--valid-region-fraction",
        type=float,
        default=0.05,
        help="fraction of the source file (by raw byte size), taken from "
        "the tail, reserved for validation and off-limits to training "
        "(default: 0.05 = last 5%%)",
    )
    args = parser.parse_args()

    source_size = args.source.stat().st_size
    valid_region_start = int(source_size * (1 - args.valid_region_fraction))

    train_written = _write_from_offset(
        args.source, 0, valid_region_start, args.train_output, args.train_bytes
    )
    valid_written = _write_from_offset(
        args.source, valid_region_start, None, args.valid_output, args.valid_bytes
    )

    print(f"train: {train_written:,} bytes -> {args.train_output}")
    print(f"valid: {valid_written:,} bytes -> {args.valid_output}")
    if train_written < args.train_bytes:
        print(
            f"WARNING: requested {args.train_bytes:,} train bytes but only "
            f"{train_written:,} were available before the reserved "
            f"validation region (source is {source_size:,} bytes, "
            f"validation reserves the last {args.valid_region_fraction:.0%})",
            file=sys.stderr,
        )
    if valid_written < args.valid_bytes:
        print(
            f"WARNING: requested {args.valid_bytes:,} valid bytes but only "
            f"{valid_written:,} were available in the reserved region",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
