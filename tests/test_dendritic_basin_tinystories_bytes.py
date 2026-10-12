"""Tests for the raw-byte TinyStories loader (design doc section 55)."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.tinystories_bytes import (
    iter_byte_stream,
    load_byte_stream,
)


def test_load_byte_stream_reads_exact_requested_length(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"Once upon a time" * 10)
    stream = load_byte_stream(str(path), max_bytes=25)
    assert len(stream) == 25
    assert stream == list(b"Once upon a time" * 10)[:25]


def test_load_byte_stream_respects_offset(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    content = b"abcdefghij"
    path.write_bytes(content)
    stream = load_byte_stream(str(path), max_bytes=4, offset=3)
    assert stream == list(content[3:7])


def test_load_byte_stream_values_are_valid_byte_range(tmp_path) -> None:
    rng = np.random.default_rng(0)
    path = tmp_path / "sample.bin"
    path.write_bytes(bytes(rng.integers(0, 256, size=200, dtype=np.uint8)))
    stream = load_byte_stream(str(path), max_bytes=200)
    assert len(stream) == 200
    assert all(0 <= b <= 255 for b in stream)


def test_load_byte_stream_truncates_at_end_of_file(tmp_path) -> None:
    path = tmp_path / "short.txt"
    path.write_bytes(b"hi")
    stream = load_byte_stream(str(path), max_bytes=1000)
    assert stream == list(b"hi")


def test_iter_byte_stream_crosses_chunks_without_retaining_input(tmp_path) -> None:
    path = tmp_path / "chunked.bin"
    content = bytes(range(64))
    path.write_bytes(content)

    stream = iter_byte_stream(str(path), max_bytes=31, offset=7, chunk_size=5)

    assert iter(stream) is stream
    assert list(stream) == list(content[7:38])
