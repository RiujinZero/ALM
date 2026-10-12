"""Tests for word-completion event detection (design doc section 54
Phase 4: "Raw character stream")."""

from __future__ import annotations

from manifoldflow.dendritic_basin.word_completion import (
    WordCompletionDetector,
    build_completion_event_stream,
)

# Toy character channel ids: a=0, b=1, c=2, s=3, e=4
A, B, C, S, E = 0, 1, 2, 3, 4


def test_detector_fires_exactly_at_the_completion_position() -> None:
    detector = WordCompletionDetector({"cab": [C, A, B]})
    results = [detector.update(ch)["cab"] for ch in [C, A, B, C]]
    assert results == [0.0, 0.0, 1.0, 0.0]


def test_detector_does_not_fire_on_a_partial_or_reordered_match() -> None:
    detector = WordCompletionDetector({"cab": [C, A, B]})
    results = [detector.update(ch)["cab"] for ch in [A, B, C]]  # "abc", not "cab"
    assert results == [0.0, 0.0, 0.0]


def test_detector_handles_overlapping_candidate_words_independently() -> None:
    detector = WordCompletionDetector({"cab": [C, A, B], "ab": [A, B]})
    outputs = [detector.update(ch) for ch in [C, A, B]]
    assert outputs[-1]["cab"] == 1.0
    assert outputs[-1]["ab"] == 1.0  # "ab" also completes at the same position


def test_reset_clears_the_sliding_window() -> None:
    detector = WordCompletionDetector({"cab": [C, A, B]})
    detector.update(C)
    detector.update(A)
    detector.reset()
    result = detector.update(B)
    assert result["cab"] == 0.0  # window was cleared, only "b" remains


def test_build_completion_event_stream_shape_and_values() -> None:
    stream = [C, A, B, S, E, A]
    events = build_completion_event_stream(stream, {"cab": [C, A, B], "se": [S, E]})
    assert events.shape == (6, 2)
    assert events[2, 0] == 1.0  # "cab" completes at index 2
    assert events[4, 1] == 1.0  # "se" completes at index 4
    assert events.sum() == 2.0  # no spurious completions anywhere else
