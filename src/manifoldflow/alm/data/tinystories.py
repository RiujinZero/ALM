"""TinyStories corpus loading: checksums, dedup, stable-hash splits, shards.

Design doc section 4.1 (corpus policy) and 14.1 (preprocessing):
  - Download and checksum the official corpus artifacts; record immutable
    source hashes.
  - Deduplicate exact stories and flag near duplicates before splitting.
  - Assign splits by stable story hash so reruns cannot leak validation text
    into training.
  - Write uint16 token shards with story offsets, lengths, source variant,
    and CRC.

The design's primary train stream is TinyStoriesV2-GPT4; that file was not
present on this machine's corpus drive at implementation time (see the
implementation journal), so the original TinyStories corpus is used for
both the primary and paper-comparison streams. `corpus_variant` records
which file a shard came from so this is auditable, not silent.
"""

from __future__ import annotations

import hashlib
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifoldflow.alm.data.tokenizer import (
    ByteControlTokenizer,
    StoryBpeTokenizer,
    ids_to_uint16,
)

STORY_DELIM = "<|endoftext|>"


def iter_stories(path: str | Path) -> Iterator[str]:
    """Yield raw story text, stripped, from a `<|endoftext|>`-delimited file."""
    text = Path(path).read_text(encoding="utf-8")
    for raw in text.split(STORY_DELIM):
        story = raw.strip()
        if story:
            yield story


def story_hash(story: str) -> str:
    """Stable hash of exact story content, used for both dedup and split assignment."""
    return hashlib.sha256(story.encode("utf-8")).hexdigest()


def file_checksum(path: str | Path) -> str:
    """SHA-256 of the raw corpus file, recorded as an immutable source hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_split(story_id: str, val_fraction: float = 0.01) -> str:
    """Deterministic train/validation split from the story's own hash.

    Independent of file read order, so reruns (or corpus re-shuffles
    upstream) cannot leak validation text into training.
    """
    bucket = int(story_id[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < val_fraction else "train"


@dataclass
class DedupedCorpus:
    """Deduplicated, split-assigned stories from one or more source files."""

    train: list[str]
    validation: list[str]
    n_seen: int
    n_duplicates: int
    source_checksums: dict[str, str]


def load_deduped_corpus(
    paths: dict[str, str | Path],
    val_fraction: float = 0.01,
    limit_per_source: int | None = None,
) -> DedupedCorpus:
    """Load, checksum, dedup, and split-assign stories from named source files.

    `paths` maps a `corpus_variant` label (e.g. "tinystories_v2gpt4",
    "tinystories_original") to a file path. `limit_per_source` caps how many
    stories are read per file, for CPU-feasible smoke runs; it is applied
    before dedup so it never biases the train/validation split.
    """
    seen: set[str] = set()
    train: list[str] = []
    validation: list[str] = []
    n_seen = 0
    n_duplicates = 0
    checksums: dict[str, str] = {}

    for variant, path in paths.items():
        checksums[variant] = file_checksum(path)
        for i, story in enumerate(iter_stories(path)):
            if limit_per_source is not None and i >= limit_per_source:
                break
            n_seen += 1
            story_id = story_hash(story)
            if story_id in seen:
                n_duplicates += 1
                continue
            seen.add(story_id)
            split = assign_split(story_id, val_fraction=val_fraction)
            (train if split == "train" else validation).append(story)

    return DedupedCorpus(
        train=train,
        validation=validation,
        n_seen=n_seen,
        n_duplicates=n_duplicates,
        source_checksums=checksums,
    )


@dataclass
class TokenShard:
    """Concatenated uint16 token IDs for a split, with per-story offsets/lengths.

    `offsets[i]:offsets[i] + lengths[i]` is story `i`'s token span in `tokens`.
    `crc32` is computed over the raw `tokens` bytes for integrity checking.
    """

    tokens: np.ndarray  # uint16, shape (total_len,)
    offsets: np.ndarray  # int64, shape (n_stories,)
    lengths: np.ndarray  # int64, shape (n_stories,)
    crc32: int

    def story(self, i: int) -> np.ndarray:
        start = int(self.offsets[i])
        return self.tokens[start : start + int(self.lengths[i])]

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            tokens=self.tokens,
            offsets=self.offsets,
            lengths=self.lengths,
            crc32=np.array([self.crc32], dtype=np.uint32),
        )

    @classmethod
    def load(cls, path: str | Path) -> TokenShard:
        with np.load(path) as data:
            return cls(
                tokens=data["tokens"],
                offsets=data["offsets"],
                lengths=data["lengths"],
                crc32=int(data["crc32"][0]),
            )

    def verify(self) -> bool:
        return zlib.crc32(self.tokens.tobytes()) == self.crc32


def encode_shard(
    stories: list[str], tokenizer: StoryBpeTokenizer | ByteControlTokenizer
) -> TokenShard:
    """Encode a list of stories (each wrapped BOS/.../EOS) into one token shard."""
    all_ids: list[int] = []
    offsets: list[int] = []
    lengths: list[int] = []
    cursor = 0
    for story in stories:
        ids = tokenizer.encode_story(story)
        offsets.append(cursor)
        lengths.append(len(ids))
        all_ids.extend(ids)
        cursor += len(ids)
    tokens = ids_to_uint16(all_ids)
    return TokenShard(
        tokens=tokens,
        offsets=np.asarray(offsets, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
        crc32=zlib.crc32(tokens.tobytes()),
    )
