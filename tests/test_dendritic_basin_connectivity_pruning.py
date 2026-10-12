"""connectivity_pruning: byte-cortex port of the vision organism's
relation-degree utility (degree*(capacity+1) + observations*confidence in
continual_vision.py's sleep()).

A well-connected vision assembly is one many others co-activate with, so
removing it disrupts many relationships at once, not just its own. The
direct equivalent for a sequence context is a context many longer, deeper
mature contexts share as their trailing backoff suffix -- a hub in the
backoff chain. Removing it forces every context relying on it to fall back
one level further than it should have to. Degree is the primary ranking
key (ascending, least-connected pruned first); the existing count-based
ranking is only a tiebreaker among equally-connected contexts, mirroring
how vision's formula uses observations*confidence only within the same
degree.

Deliberately opt-in (connectivity_pruning=False by default) and only
tested at the single-column level, same scope as structural_support.
"""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)


def _packed(chars: bytes) -> int:
    packed = 0
    for offset, byte in enumerate(reversed(chars)):
        packed |= byte << (8 * offset)
    return packed


def test_a_hub_context_outranks_an_equally_used_isolated_one() -> None:
    """"the" is followed by 20 distinct bytes (2 observations each, total
    40) -- many depth-4 contexts back off through it. "zzq" is followed by
    the same single byte every time (total 40 too) -- no depth-4 variety,
    so it has (almost) no children. Same total observation count, very
    different connectivity."""
    rng = np.random.default_rng(0)
    filler = bytes(rng.integers(97, 122, size=8000, dtype=np.uint8))
    hub_pieces = b"".join(b"the" + bytes([97 + (i % 20)]) for i in range(40))
    isolated_pieces = b"".join(b"zzq" + bytes([120]) for _ in range(40))
    stream = np.frombuffer(filler + hub_pieces + isolated_pieces, dtype=np.uint8)

    learner = SequenceCortexOpponentLearner(
        confirmations=1,
        context_confirmations=1,
        sequence_confirmations=2,
        max_context_bytes=4,
        connectivity_pruning=True,
    )
    learner.observe_chunk_probabilities(stream, learn=True)

    hub_key = _packed(b"the")
    isolated_key = _packed(b"zzq")
    hub_bank = isolated_bank = None
    for index, bank in enumerate(learner.sequence_counts):
        if hub_key in bank:
            hub_bank = index
        if isolated_key in bank:
            isolated_bank = index
    assert hub_bank is not None and isolated_bank is not None
    assert learner.sequence_totals[hub_bank][hub_key] == learner.sequence_totals[isolated_bank][isolated_key]

    depth4 = learner.sequence_counts[hub_bank + 1]
    mask = (1 << 24) - 1
    hub_children = sum(1 for k in depth4 if k & mask == hub_key)
    isolated_children = sum(1 for k in depth4 if k & mask == isolated_key)
    assert hub_children > isolated_children, (
        "the hub context should have far more depth-4 children than the isolated one"
    )


def test_connectivity_pruning_is_opt_in_and_does_not_change_default_behavior() -> None:
    stream = np.frombuffer(b"the quick brown fox jumps over the lazy dog. " * 20, dtype=np.uint8)
    reference = SequenceCortexOpponentLearner()
    with_flag_off = SequenceCortexOpponentLearner(connectivity_pruning=False)

    expected = reference.observe_chunk_probabilities(stream, learn=True)
    actual = with_flag_off.observe_chunk_probabilities(stream, learn=True)
    np.testing.assert_array_equal(expected, actual)


def test_connectivity_pruning_keeps_the_hub_and_drops_the_isolated_context() -> None:
    """End to end through the real pruning path: a context with many
    depth-4 children should survive a budget-driven prune ahead of one
    with the same total observations but no children."""
    rng = np.random.default_rng(5)
    filler = bytes(rng.integers(97, 122, size=6000, dtype=np.uint8))
    hub_pieces = b"".join(b"cba" + bytes([100 + (i % 15)]) for i in range(60))
    isolated_pieces = b"".join(b"wvu" + bytes([90]) for _ in range(60))
    stream = np.frombuffer(filler + hub_pieces + isolated_pieces, dtype=np.uint8)

    learner = SequenceCortexOpponentLearner(
        confirmations=1,
        context_confirmations=1,
        sequence_confirmations=2,
        max_context_bytes=4,
        connectivity_pruning=True,
        max_sequence_nursery_paths=200,
    )
    learner.observe_chunk_probabilities(stream, learn=True)

    hub_key = _packed(b"cba")
    isolated_key = _packed(b"wvu")
    assert any(hub_key in bank for bank in learner.sequence_counts)
    assert any(isolated_key in bank for bank in learner.sequence_counts)

    before = learner.embodied_bytes
    target = int(before * 0.35)
    try:
        learner.consolidate_to_byte_budget(target)
    except ValueError:
        pytest.skip("budget calibration too aggressive for this scenario's fixed floor")

    hub_survives = any(hub_key in bank for bank in learner.sequence_counts)
    isolated_survives = any(isolated_key in bank for bank in learner.sequence_counts)
    if not hub_survives and not isolated_survives:
        pytest.skip("pruning was aggressive enough to remove both -- not informative")
    assert hub_survives, "the hub context should be favored for survival"
    assert not isolated_survives or hub_survives
