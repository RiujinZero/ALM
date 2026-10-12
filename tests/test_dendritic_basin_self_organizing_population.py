"""Tests for the self-organizing relation population (design doc section
54 "Adaptive bell dendrites on generic vectors", section 50's bounded
candidate-search goal)."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.relation_discovery import build_stream_diary
from manifoldflow.dendritic_basin.self_organizing_population import (
    discover_relations_by_self_organizing_population,
)

# Same tiny closed vocabulary as test_dendritic_basin_relation_discovery.py:
# 0=<bos>, 1=alice, 2=bob, 3=sees, 4=cat, 5=dog
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


def test_population_discovers_alice_or_bob_not_cat_or_dog() -> None:
    """The bounded population must still surface genuinely predictive
    channels (ALICE/BOB), same as the exhaustive `discover_and_rank_relations`
    -- it is a different search strategy over the same candidate space, not
    a different notion of what counts as useful.
    """
    rng = np.random.default_rng(0)
    stream = _generate_stream(500, rng)
    diary = build_stream_diary(stream, vocab_size=VOCAB_SIZE, timescales=(1, 2, 4, 8))

    results, stats = discover_relations_by_self_organizing_population(
        diary,
        anchor=SEES,
        vocab_size=VOCAB_SIZE,
        n_timescales=4,
        seed=1,
        population_size=6,
        max_candidates_per_slot=6,
        min_gain=0.1,
        n_candidates=3,
        rounds=60,
        perturbations_per_round=12,
        episodes_per_candidate=3,
    )

    assert len(results) > 0
    top_channels = {relation.key[0] for relation in results}
    assert top_channels <= {ALICE, BOB}


def test_population_never_evaluates_more_than_its_bounded_budget() -> None:
    """The whole point of Phase 6: candidate evaluations must be bounded
    by `population_size * max_candidates_per_slot`, never by
    `vocab_size * n_timescales` -- this must hold even as vocab_size
    grows, unlike `discover_candidate_relations`'s exhaustive scan.
    """
    rng = np.random.default_rng(0)
    stream = _generate_stream(500, rng)
    diary = build_stream_diary(stream, vocab_size=VOCAB_SIZE, timescales=(1, 2, 4, 8))

    population_size = 4
    max_candidates_per_slot = 5
    _results, stats = discover_relations_by_self_organizing_population(
        diary,
        anchor=SEES,
        vocab_size=VOCAB_SIZE,
        n_timescales=4,
        seed=2,
        population_size=population_size,
        max_candidates_per_slot=max_candidates_per_slot,
        min_gain=0.1,
        rounds=40,
        perturbations_per_round=12,
    )

    budget = population_size * max_candidates_per_slot
    assert stats.total_candidate_evaluations <= budget
    assert stats.slots_converged <= population_size


def test_population_stops_reallocating_once_every_candidate_is_tried() -> None:
    """With a tiny vocabulary, `population_size * max_candidates_per_slot`
    can exceed the number of distinct (channel, timescale) pairs that
    exist at all -- the population must not error or hang, it must simply
    stop once every candidate has been tried once.
    """
    rng = np.random.default_rng(0)
    stream = _generate_stream(200, rng)
    diary = build_stream_diary(stream, vocab_size=VOCAB_SIZE, timescales=(1, 2))

    max_distinct_candidates = (VOCAB_SIZE - 1) * 2  # exclude anchor channel itself
    _results, stats = discover_relations_by_self_organizing_population(
        diary,
        anchor=SEES,
        vocab_size=VOCAB_SIZE,
        n_timescales=2,
        seed=3,
        population_size=20,
        max_candidates_per_slot=20,
        min_gain=0.1,
        rounds=20,
        perturbations_per_round=8,
    )

    assert stats.total_candidate_evaluations <= max_distinct_candidates


def test_population_excludes_a_structurally_fixed_channel() -> None:
    """Regression check carried over from the exhaustive search
    (test_dendritic_basin_relation_discovery.py): <bos> always occurs a
    fixed 2 steps before every "sees", so its trace can't discriminate
    anything there -- the coefficient-of-variation guard must reject it
    here too, not just in the exhaustive path.
    """
    BOS = 0
    rng = np.random.default_rng(3)
    stream = _generate_stream(500, rng)
    diary = build_stream_diary(stream, vocab_size=VOCAB_SIZE, timescales=(1, 2, 4, 8))

    results, _stats = discover_relations_by_self_organizing_population(
        diary,
        anchor=SEES,
        vocab_size=VOCAB_SIZE,
        n_timescales=4,
        seed=4,
        population_size=10,
        max_candidates_per_slot=10,
        min_gain=0.1,
        rounds=60,
        perturbations_per_round=12,
        episodes_per_candidate=3,
    )

    assert not any(relation.key[0] == BOS for relation in results)
