"""Phase 8 (design doc section 54: "Larger raw language stream -- Move to
TinyStories and then unrelated text such as Wikipedia"; section 55:
"Start with raw UTF-8 TinyStories because language provides an easy
quantitative evaluation... Feed 1-10 MB continuously for the first run").

No tokenizer: the raw file bytes are the vocabulary (0-255), exactly
"raw UTF-8" as section 55 asks -- this project's own TinyStories BPE/byte
tokenizer from the earlier (now-paused) MF-ALM line
(`manifoldflow/alm/data/tokenizer.py`) is deliberately not reused here,
since the whole point of this line of work is that no tokenizer is
needed at all.
"""

from __future__ import annotations

from collections.abc import Iterator

BYTE_VOCAB_SIZE = 256


def iter_byte_stream(
    path: str,
    max_bytes: int,
    offset: int = 0,
    *,
    chunk_size: int = 1 << 20,
) -> Iterator[int]:
    """Yield raw bytes in bounded chunks without retaining the corpus."""
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    remaining = max_bytes
    with open(path, "rb") as handle:
        handle.seek(offset)
        while remaining:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            yield from chunk
            remaining -= len(chunk)


def load_byte_stream(path: str, max_bytes: int, offset: int = 0) -> list[int]:
    """Reads up to `max_bytes` raw bytes from `path`, starting at
    `offset` -- plain `list[int]` in 0-255, no decoding/encoding step.
    """
    return list(iter_byte_stream(path, max_bytes, offset))
