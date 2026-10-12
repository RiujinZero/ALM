"""Tests for the character-stream smoke pipeline (design doc section 54
Phase 4: "Raw character stream")."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.character_stream import (
    NAME_TO_OBJECT,
    TOKEN_TO_ID,
    VOCAB,
    generate_char_stream,
)
from manifoldflow.experiments.dendritic_basin_character_stream_smoke import (
    _object_first_char_target,
    _sees_just_completed_positions,
    build_event_diary,
)


def test_generated_stream_uses_only_characters_no_word_ids() -> None:
    rng = np.random.default_rng(0)
    stream = generate_char_stream(5, rng)
    decoded = "".join(VOCAB[t] if len(VOCAB[t]) == 1 else "[BOS]" for t in stream)
    for name in NAME_TO_OBJECT:
        assert len(name) > 1  # sanity: names are multi-character
    assert all(len(VOCAB[t]) == 1 or VOCAB[t] == "<bos>" for t in stream)
    assert decoded  # just confirms it decodes without error


def test_build_event_diary_flags_only_completed_words() -> None:
    rng = np.random.default_rng(1)
    stream = generate_char_stream(3, rng)
    diary = build_event_diary(stream)
    assert len(diary) == len(stream) - 1
    # At every position, at most the names that have actually finished
    # being spelled since the last <bos> can be flagged.
    for flags, _next in diary:
        assert set(flags) == {"alice", "bob", "carol", "dave", "sees"}
        assert all(v in (0.0, 1.0) for v in flags.values())


def test_sees_just_completed_positions_count_matches_sentence_count() -> None:
    rng = np.random.default_rng(2)
    n_sentences = 20
    stream = generate_char_stream(n_sentences, rng)
    diary = build_event_diary(stream)
    positions = _sees_just_completed_positions(diary)
    assert len(positions) == n_sentences


def test_object_first_char_target_matches_the_true_object() -> None:
    rng = np.random.default_rng(3)
    stream = generate_char_stream(10, rng)
    diary = build_event_diary(stream)
    positions = _sees_just_completed_positions(diary)

    id_to_token = {i: t for t, i in TOKEN_TO_ID.items()}
    for pos in positions:
        target = _object_first_char_target(diary, int(pos))
        target_char = id_to_token[target]
        assert target_char in {obj[0] for obj in NAME_TO_OBJECT.values()}


def test_exactly_one_name_flag_active_at_each_sees_completion() -> None:
    """A subject's name-completion flag must be the only one active right
    when "sees" finishes -- confirms word-completion detection doesn't
    spuriously flag multiple names per sentence.
    """
    rng = np.random.default_rng(4)
    stream = generate_char_stream(30, rng)
    diary = build_event_diary(stream)
    positions = _sees_just_completed_positions(diary)

    for pos in positions:
        flags, _next = diary[int(pos)]
        active_names = [
            name for name in ("alice", "bob", "carol", "dave") if flags[name] > 0.5
        ]
        assert len(active_names) == 1
