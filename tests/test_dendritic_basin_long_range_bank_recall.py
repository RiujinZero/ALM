"""The actual Idea 2 Stage A go/no-go test: can the long-range bank carry
identity-specific information across a gap the ordinary short-range banks
cannot see across at all, using nothing but the same context-matching
mechanism already used for short-range prediction?

Geometry, verified explicitly rather than assumed (two earlier probe drafts
got this wrong in different ways -- see docs/HTM_Inspired_Followup_Experiments.md):
  - max_context_bytes=6: the ordinary banks' entire reach.
  - A 24-byte bridge phrase, identical in both training sentences, sits
    between the identifying name and the target word -- so the last 6
    bytes before the target are byte-for-byte identical in both cases,
    and the ordinary short banks genuinely cannot disambiguate.
  - long_range_context_bytes=27 (name + bridge, the whole query length) --
    it must exceed the bridge's own 24 bytes, or the trailing window is
    *only* the bridge and drops the name entirely (confirmed directly: with
    long_range_context_bytes=24, both names mapped to the identical key).
"""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)

MAX_CONTEXT_BYTES = 6
LONG_RANGE_CONTEXT_BYTES = 27
BRIDGE = b" said that the ball was "  # 24 bytes, identical in both sentences


def _build_probe():
    sentence_a = b"Amy" + BRIDGE + b"red."
    sentence_b = b"Bob" + BRIDGE + b"blue."
    train = (sentence_a + b" ") * 40 + (sentence_b + b" ") * 40
    query = b"Bob" + BRIDGE  # correct continuation: "blue"
    return train, query


def _predict(learner, query: bytes) -> np.ndarray:
    learner.reset_episode()
    learner.observe_chunk_probabilities(np.frombuffer(query, dtype=np.uint8), learn=False)
    return learner.predict_next_distribution()


def test_geometry_hides_the_identity_from_ordinary_short_banks() -> None:
    train, query = _build_probe()
    assert len(BRIDGE) > MAX_CONTEXT_BYTES
    assert len(query) >= LONG_RANGE_CONTEXT_BYTES
    assert query[-MAX_CONTEXT_BYTES:] == b"l was "  # says nothing about which name


def test_baseline_is_near_chance_on_the_hidden_identity() -> None:
    train, query = _build_probe()
    learner = SequenceCortexOpponentLearner(max_context_bytes=MAX_CONTEXT_BYTES)
    learner.observe_chunk_probabilities(np.frombuffer(train, dtype=np.uint8), learn=True)
    dist = _predict(learner, query)
    p_correct, p_wrong = float(dist[ord("b")]), float(dist[ord("r")])
    # Near-chance, not confidently correct: this is what "ordinary banks
    # cannot see the identity" looks like in practice, not p ~= 0 for both.
    assert p_correct < 0.7


def test_long_range_bank_resolves_the_identity_across_the_gap() -> None:
    train, query = _build_probe()
    learner = SequenceCortexOpponentLearner(
        max_context_bytes=MAX_CONTEXT_BYTES,
        long_range_bank=True,
        long_range_context_bytes=LONG_RANGE_CONTEXT_BYTES,
        long_range_strength=4.0,
    )
    learner.observe_chunk_probabilities(np.frombuffer(train, dtype=np.uint8), learn=True)
    dist = _predict(learner, query)
    p_correct, p_wrong = float(dist[ord("b")]), float(dist[ord("r")])
    assert p_correct > 0.9
    assert p_correct > p_wrong


def test_gate_creation_only_keeps_recall_confidence_and_selectivity() -> None:
    """Refined gate: only the *creation* of a new long-range key is gated
    on burst_fraction; an already-existing key is reinforced on every
    match regardless of current burst_fraction. The original
    gate-everything version measurably cost recall confidence here (95%
    unconditional -> 63% gated, because the correct key was confirmed once
    out of 40 real repetitions instead of 36) -- this refinement should
    recover that confidence while keeping the sparsity win measured in
    test_dendritic_basin_long_range_bank.py's
    test_burst_gated_growth_is_sparser_than_unconditional_growth.
    """
    train, query = _build_probe()
    learner = SequenceCortexOpponentLearner(
        max_context_bytes=MAX_CONTEXT_BYTES,
        long_range_bank=True,
        long_range_context_bytes=LONG_RANGE_CONTEXT_BYTES,
        long_range_strength=4.0,
        burst_surprise_signal=True,
        burst_ema_rate=0.02,
        long_range_gate_on_burst=True,
        long_range_gate_threshold=0.1,
    )
    learner.observe_chunk_probabilities(np.frombuffer(train, dtype=np.uint8), learn=True)
    dist = _predict(learner, query)
    p_correct = float(dist[ord("b")])
    assert p_correct > 0.9
