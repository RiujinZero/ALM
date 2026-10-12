"""Tests for automatic temporal-relation discovery (design doc sections
14-18, 28 of `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`)."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.relation_discovery import (
    build_stream_diary,
    discover_and_rank_relations,
    discover_candidate_relations,
    relation_information_gain,
)

# A tiny closed vocabulary: 0=<bos>, 1=alice, 2=bob, 3=sees, 4=cat, 5=dog
VOCAB_SIZE = 6
ALICE, BOB, SEES, CAT, DOG = 1, 2, 3, 4, 5


def _generate_stream(n_sentences: int, rng: np.random.Generator) -> list[int]:
    tokens = []
    for _ in range(n_sentences):
        if rng.integers(0, 2) == 0:
            tokens += [0, ALICE, SEES, CAT]
        else:
            tokens += [0, BOB, SEES, DOG]
    return tokens


def test_build_stream_diary_length_and_next_token_alignment() -> None:
    stream = [0, 1, 3, 4]
    diary = build_stream_diary(stream, vocab_size=VOCAB_SIZE, timescales=(1, 4))
    assert len(diary) == len(stream) - 1
    for i, (token_id, _q, next_token) in enumerate(diary):
        assert token_id == stream[i]
        assert next_token == stream[i + 1]


def test_diary_trace_snapshot_shape_matches_timescales_and_vocab() -> None:
    stream = [0, 1, 3, 4, 0, 2, 3, 5]
    diary = build_stream_diary(stream, vocab_size=VOCAB_SIZE, timescales=(1, 2, 4))
    _token_id, q, _next = diary[0]
    assert q.shape == (3, VOCAB_SIZE)


def test_discover_candidate_relations_accepts_structurally_valid_pairs() -> None:
    """`discover_candidate_relations` alone only checks J (a genuine,
    stable interaction) -- it is expected to accept both truly useful
    pairs (ALICE/BOB, paired with anchor=SEES) *and* spurious-but-real
    interactions (CAT/DOG's traces are structurally related to each other
    too, just not causally relevant to predicting the *current*
    sentence's object). Separating "valid" from "useful" is
    discover_and_rank_relations's job (tested below), not this function's.
    """
    rng = np.random.default_rng(0)
    stream = _generate_stream(300, rng)
    diary = build_stream_diary(stream, vocab_size=VOCAB_SIZE, timescales=(1, 2, 4, 8))

    accepted = discover_candidate_relations(
        diary,
        anchor=SEES,
        vocab_size=VOCAB_SIZE,
        n_timescales=4,
        seed=1,
        rounds=40,
        perturbations_per_round=12,
    )

    assert len(accepted) > 0


def test_discover_and_rank_relations_surfaces_alice_and_bob_not_cat_or_dog() -> None:
    """The actual "automatic structural discovery" pipeline (section 28):
    rank by predictive-improvement (information gain against the real
    next token), not by J alone. The top-ranked channels must be ALICE
    and BOB (genuinely predictive of the object) -- CAT and DOG's traces,
    despite being structurally related to each other, must not rank
    highly, since they carry no information about the *current*
    sentence's object.
    """
    rng = np.random.default_rng(0)
    stream = _generate_stream(500, rng)
    diary = build_stream_diary(stream, vocab_size=VOCAB_SIZE, timescales=(1, 2, 4, 8))

    ranked = discover_and_rank_relations(
        diary,
        anchor=SEES,
        vocab_size=VOCAB_SIZE,
        n_timescales=4,
        seed=1,
        min_gain=0.1,
        n_candidates=3,
        rounds=60,
        perturbations_per_round=12,
    )

    assert len(ranked) > 0
    top_channels = {key[0] for key, _unit, _gain in ranked}
    assert top_channels <= {ALICE, BOB}  # never CAT or DOG


def test_discover_candidate_relations_excludes_a_structurally_fixed_channel() -> None:
    """Regression test: <bos> always occurs a fixed 2 steps before every
    "sees" in this task's sentence template, so its trace reaches nearly
    the same value every time "sees" fires -- it must never be discovered
    as a candidate relation, even though (before the coefficient-of-
    variation guard existed) a dendrite could latch onto its tiny
    floating-point-level differences and score a spuriously high
    information gain. See relation_discovery.py's
    discover_candidate_relations docstring for the full story.
    """
    BOS = 0
    rng = np.random.default_rng(3)
    stream = _generate_stream(500, rng)
    diary = build_stream_diary(stream, vocab_size=VOCAB_SIZE, timescales=(1, 2, 4, 8))

    accepted = discover_candidate_relations(
        diary,
        anchor=SEES,
        vocab_size=VOCAB_SIZE,
        n_timescales=4,
        seed=4,
        rounds=60,
        perturbations_per_round=12,
        episodes_per_candidate=3,
    )

    assert not any(trace_channel == BOS for trace_channel, _ti in accepted)


def test_relation_information_gain_is_high_for_a_predictive_relation() -> None:
    """A hand-built "perfect" relation (fires exactly when ALICE's trace
    is present) must show a large information gain, since it fully
    resolves the object ambiguity; a relation that never fires must show
    zero gain (it never contributes evidence).
    """
    rng = np.random.default_rng(2)
    stream = _generate_stream(200, rng)
    diary = build_stream_diary(stream, vocab_size=VOCAB_SIZE, timescales=(1, 4))

    class _AlwaysOffUnit:
        def respond(self, a: float, b: float) -> float:
            return 0.0

    class _FiresOnAliceTraceUnit:
        def respond(self, a: float, b: float) -> float:
            return 1.0 if b > 0.05 else 0.0

    zero_gain = relation_information_gain(
        _AlwaysOffUnit(), (ALICE, 0), diary, anchor=SEES, vocab_size=VOCAB_SIZE
    )
    alice_gain = relation_information_gain(
        _FiresOnAliceTraceUnit(), (ALICE, 0), diary, anchor=SEES, vocab_size=VOCAB_SIZE
    )

    assert zero_gain == 0.0
    assert alice_gain > 0.5  # bits: should meaningfully reduce next-token entropy
