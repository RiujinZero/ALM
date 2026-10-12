"""Round trip, coverage, and deterministic-ID tests for the ALM tokenizers.

Design doc section 17 (software architecture): data/tokenizer.py must pass
"round trip, coverage, deterministic IDs."
"""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.alm.data.tokenizer import (
    BYTE_VOCAB_SIZE,
    SPECIAL_TOKENS,
    ByteControlTokenizer,
    StoryBpeTokenizer,
    ids_to_uint16,
)

SAMPLE_STORIES = [
    "Once upon a time, there was a little cat named Tom. Tom liked to play in the sun.",
    'Lina\'s kite is green. "Look at my kite!" she said, and ran across the field.',
    "The dog and the cat were friends. They played all day, and then they slept.",
    "Anna had a red ball. She threw the ball, and it rolled under the big tree.",
]


@pytest.fixture(scope="module")
def bpe_tokenizer() -> StoryBpeTokenizer:
    return StoryBpeTokenizer.train(SAMPLE_STORIES, vocab_size=512)


def test_bpe_round_trip(bpe_tokenizer: StoryBpeTokenizer) -> None:
    for story in SAMPLE_STORIES:
        ids = bpe_tokenizer.encode(story)
        assert bpe_tokenizer.decode(ids) == story


def test_bpe_deterministic_ids(bpe_tokenizer: StoryBpeTokenizer) -> None:
    for story in SAMPLE_STORIES:
        assert bpe_tokenizer.encode(story) == bpe_tokenizer.encode(story)


def test_bpe_reserves_special_tokens(bpe_tokenizer: StoryBpeTokenizer) -> None:
    for i, token in enumerate(SPECIAL_TOKENS):
        assert bpe_tokenizer.token_to_id(token) == i
    ids = bpe_tokenizer.ids
    assert (ids.pad, ids.bos, ids.eos, ids.unk, ids.story_sep) == (0, 1, 2, 3, 4)


def test_bpe_encode_story_wraps_bos_eos(bpe_tokenizer: StoryBpeTokenizer) -> None:
    wrapped = bpe_tokenizer.encode_story(SAMPLE_STORIES[0])
    assert wrapped[0] == bpe_tokenizer.ids.bos
    assert wrapped[-1] == bpe_tokenizer.ids.eos
    assert wrapped[1:-1] == bpe_tokenizer.encode(SAMPLE_STORIES[0])


def test_bpe_save_load_round_trip(bpe_tokenizer: StoryBpeTokenizer, tmp_path) -> None:
    path = tmp_path / "tok.json"
    bpe_tokenizer.save(path)
    reloaded = StoryBpeTokenizer.load(path)
    for story in SAMPLE_STORIES:
        assert reloaded.encode(story) == bpe_tokenizer.encode(story)


def test_byte_control_vocab_size_is_260() -> None:
    assert BYTE_VOCAB_SIZE == 260
    assert ByteControlTokenizer().vocab_size == 260


def test_byte_control_round_trip_covers_unicode() -> None:
    tok = ByteControlTokenizer()
    text = "Lina's kite is green. éè中文"
    ids = tok.encode(text)
    assert all(4 <= i < 260 for i in ids)  # 4 specials, then 256 byte values
    assert tok.decode(ids) == text


def test_byte_control_deterministic() -> None:
    tok = ByteControlTokenizer()
    for story in SAMPLE_STORIES:
        assert tok.encode(story) == tok.encode(story)


def test_byte_control_encode_story_wraps_bos_eos() -> None:
    tok = ByteControlTokenizer()
    wrapped = tok.encode_story("hi")
    assert wrapped[0] == tok.ids.bos
    assert wrapped[-1] == tok.ids.eos


def test_ids_to_uint16_round_trips() -> None:
    ids = [0, 1, 4095, 65535]
    array = ids_to_uint16(ids)
    assert array.dtype == np.uint16
    assert array.tolist() == ids


def test_ids_to_uint16_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        ids_to_uint16([0, 70000])
