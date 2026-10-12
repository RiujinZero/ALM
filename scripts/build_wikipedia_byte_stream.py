"""Flatten WikiExtractor's sharded Wikipedia dump into plain byte streams.

The source tree (``C:/path/to/gdrive/training_data/Wikipedia/``) is WikiExtractor
output: hundreds of numbered directories (``00``, ``01``, ...), each holding
several ``wiki_NN`` files, each a concatenation of articles wrapped as::

    <doc title="Anarchism">
    Anarchism is a political philosophy ...
    </doc>
    <doc title="...">
    ...
    </doc>

Those ``<doc ...>``/``</doc>`` wrapper tags are stripped here, not fed to
ALM: they are identical structural boilerplate around every single article
(some containing thousands of instances), so leaving them in inflates
measured bits-per-byte with bytes that don't reflect real next-byte
difficulty, unlike TinyStories' plain prose. Nothing else in the article
body is touched -- residual WikiExtractor artifacts (``<ref .../>`` citation
markers, ``thumb|`` image captions, bare section-heading lines) are left as
found, since the ask here was the tag wrapper specifically.

Train and validation output pull from disjoint shard directories (train
from the low end, validation from the high end of the requested range) so
the split is genuinely held-out at the article level -- the same discipline
established for the CLEAR-10 and TinyStories evaluation splits elsewhere in
this project, not the "streaming"/train==test shortcut.

Usage::

    python scripts/build_wikipedia_byte_stream.py \\
        --source-root "C:/path/to/gdrive/training_data/Wikipedia" \\
        --train-output runs/wiki_train.txt --train-bytes 1900000000 \\
        --valid-output runs/wiki_valid.txt --valid-bytes 1000000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator

_DOC_OPEN_OR_CLOSE = re.compile(rb'<doc title="[^"]*">\r?\n?|</doc>\r?\n?')


def _shard_dirs(source_root: Path) -> list[Path]:
    """Non-empty shard directories only, in sorted order.

    This source tree is a partial/inconsistent local sync (checked
    2026-09-11): of 388 listed directories, only 12 actually contain files
    -- the rest are empty, and several of the populated ones carry Google
    Drive duplicate-conflict suffixes (``"36 (1)"``) rather than clean
    shard numbers. Silently including empty directories in a train/valid
    split would starve validation (or training) without any error, so
    they're filtered out here rather than trusted at face value.
    """
    all_dirs = sorted((p for p in source_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not all_dirs:
        raise ValueError(f"no shard directories found under {source_root}")
    dirs = [d for d in all_dirs if any(d.iterdir())]
    if not dirs:
        raise ValueError(f"every shard directory under {source_root} is empty")
    if len(dirs) < len(all_dirs):
        print(
            f"note: {len(all_dirs) - len(dirs)} of {len(all_dirs)} shard "
            f"directories under {source_root} are empty and were skipped "
            f"({len(dirs)} usable)",
            file=sys.stderr,
        )
    return dirs


def _shard_files(shard_dirs: list[Path]) -> Iterator[Path]:
    for shard_dir in shard_dirs:
        for wiki_file in sorted(shard_dir.iterdir(), key=lambda p: p.name):
            if wiki_file.is_file():
                yield wiki_file


def _strip_tags(raw: bytes) -> bytes:
    return _DOC_OPEN_OR_CLOSE.sub(b"", raw)


def _write_stream(files: Iterator[Path], output: Path, target_bytes: int) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("wb") as handle:
        for wiki_file in files:
            if written >= target_bytes:
                break
            body = _strip_tags(wiki_file.read_bytes())
            handle.write(body)
            written += len(body)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--train-bytes", type=int, required=True)
    parser.add_argument("--valid-output", type=Path, required=True)
    parser.add_argument("--valid-bytes", type=int, required=True)
    parser.add_argument(
        "--valid-shard-count",
        type=int,
        default=5,
        help="how many shard directories, taken from the high end of the "
        "sorted listing, are reserved for validation and excluded from "
        "training (default: 5)",
    )
    args = parser.parse_args()

    shard_dirs = _shard_dirs(args.source_root)
    if args.valid_shard_count >= len(shard_dirs):
        raise ValueError(
            f"valid_shard_count ({args.valid_shard_count}) must be smaller "
            f"than the total shard directory count ({len(shard_dirs)})"
        )
    train_dirs = shard_dirs[: -args.valid_shard_count]
    valid_dirs = shard_dirs[-args.valid_shard_count :]

    train_written = _write_stream(_shard_files(train_dirs), args.train_output, args.train_bytes)
    valid_written = _write_stream(_shard_files(valid_dirs), args.valid_output, args.valid_bytes)

    print(f"train: {train_written:,} bytes from {len(train_dirs)} shard dirs -> {args.train_output}")
    print(f"valid: {valid_written:,} bytes from {len(valid_dirs)} shard dirs -> {args.valid_output}")
    if train_written < args.train_bytes:
        print(
            f"WARNING: requested {args.train_bytes:,} train bytes but only "
            f"{train_written:,} were available in the non-validation shards",
            file=sys.stderr,
        )
    if valid_written < args.valid_bytes:
        print(
            f"WARNING: requested {args.valid_bytes:,} valid bytes but only "
            f"{valid_written:,} were available in the validation shards",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
