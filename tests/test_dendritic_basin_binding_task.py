"""Tests for the first-entity binding task generator (design doc section
21 of `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`)."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.binding import (
    NAME_TO_OBJECT,
    TOKEN_TO_ID,
    generate_token_stream,
)


def test_stream_length_is_seven_tokens_per_sentence() -> None:
    rng = np.random.default_rng(0)
    stream = generate_token_stream(10, rng)
    assert len(stream) == 70  # <bos> subject meets distractor sees object .


def test_object_always_matches_the_first_name_not_the_distractor() -> None:
    rng = np.random.default_rng(1)
    stream = generate_token_stream(200, rng)
    id_to_token = {i: t for t, i in TOKEN_TO_ID.items()}
    for i in range(0, len(stream), 7):
        subject = id_to_token[stream[i + 1]]
        distractor = id_to_token[stream[i + 3]]
        obj = id_to_token[stream[i + 5]]
        assert obj == NAME_TO_OBJECT[subject]
        assert distractor != subject


def test_distractor_is_never_the_same_as_subject() -> None:
    rng = np.random.default_rng(2)
    stream = generate_token_stream(300, rng)
    for i in range(0, len(stream), 7):
        assert stream[i + 1] != stream[i + 3]
