"""structural_support: local, per-edge confirmation credit driving pruning.

s_ij <- s_ij + eta*((1{j==y_obs} - b_j) - s_ij), applied only while the
connection's cue (context i) is active -- e_i=0 (context not active this
step) is never counted as a failed prediction, it just means the update
never runs. b_j is the receiver's own unigram baseline rate for target j,
so a connection only earns support by predicting its target better than
chance, not merely by being active when the target happens to be common.

Support is computed per edge (per (context, target) connection), but
pruning acts on whole context rows, ranked by each row's strongest edge's
support*log(1+count). Two things were tried and measured worse on real
text before landing here: pruning individual edges out of an otherwise-good
row (a row's surviving counts feed the Bayesian backoff formula's
denominator, which implicitly assumes the row is the *complete* locally
observed distribution -- partial rows distort backoff weighting for every
prediction from that context, not just the pruned edges), and ranking by
raw support with no volume term (a single lucky observation reaches a
support level a well-established, high-volume edge needs dozens of
confirmations to match, and held-out prediction quality is dominated by
the common, high-volume cases).

This is deliberately opt-in (structural_support=False by default) and only
tested at the single-column level -- see the consolidate_delta docstring
for the known multi-column merge gap not yet addressed here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)

_SETTINGS = dict(
    confirmations=1,
    context_confirmations=1,
    sequence_confirmations=2,
    max_context_bytes=4,
    structural_support=True,
    structural_support_rate=0.2,
)


def _packed(chars: bytes) -> int:
    packed = 0
    for offset, byte in enumerate(reversed(chars)):
        packed |= byte << (8 * offset)
    return packed


def test_a_reliable_rare_cue_accumulates_high_support() -> None:
    """"xyz" always precedes 'Q' (otherwise never seen): a near-zero
    baseline target that this cue predicts perfectly should converge to
    support close to 1, not just accumulate a large raw count."""
    rng = np.random.default_rng(0)
    filler = bytes(rng.integers(97, 122, size=4000, dtype=np.uint8))  # a-y, no 'Q'/'z'
    reliable = b"xyzQ" * 40
    stream = np.frombuffer(filler + reliable, dtype=np.uint8)

    learner = SequenceCortexOpponentLearner(**_SETTINGS)
    learner.observe_chunk_probabilities(stream, learn=True)

    key = _packed(b"xyz")
    support = None
    for bank in learner.sequence_support:
        if key in bank and ord("Q") in bank[key]:
            support = bank[key][ord("Q")]
            break
    assert support is not None, "the xyz->Q edge should exist and have support"
    assert support > 0.8, f"a perfectly reliable rare cue should have high support, got {support}"


def test_an_indiscriminate_cue_earns_little_credit_for_any_one_target() -> None:
    """"abc" precedes a uniformly random byte each time: no single target is
    predicted better than its own baseline, so every edge's support should
    stay near zero despite the context itself being frequently active."""
    rng = np.random.default_rng(1)
    vocabulary = np.array([ord(c) for c in "abcdefghijklmnopqrstuvwxyz "], dtype=np.uint8)
    # warm up the vocabulary so every byte is "live" before the real test
    warmup = rng.choice(vocabulary, size=4000)
    pieces = []
    for _ in range(3000):
        pieces.append(b"abc")
        pieces.append(bytes([int(rng.choice(vocabulary))]))
    stream = np.frombuffer(
        bytes(warmup) + b"".join(pieces), dtype=np.uint8
    )

    learner = SequenceCortexOpponentLearner(**_SETTINGS)
    learner.observe_chunk_probabilities(stream, learn=True)

    key = _packed(b"abc")
    supports = []
    for bank in learner.sequence_support:
        if key in bank:
            supports.extend(bank[key].values())
    assert supports, "the abc-> context should have accumulated some edges"
    # Per-edge sampling noise means an occasional edge can drift up over a
    # finite run even with a truly uniform target distribution -- what the
    # rule actually promises is no *systematic* favoritism, so check the
    # average across all of this context's edges rather than the single
    # worst one.
    mean_abs_support = sum(abs(s) for s in supports) / len(supports)
    assert mean_abs_support < 0.15, (
        f"an indiscriminate cue's edges should average near zero support, "
        f"got mean_abs={mean_abs_support} over {supports}"
    )


def test_no_opportunity_means_no_update() -> None:
    """A context that never occurs should never get a support entry at
    all -- e_i=0 is not a failed prediction, it is simply never evaluated."""
    rng = np.random.default_rng(2)
    stream = np.frombuffer(bytes(rng.integers(97, 123, size=5000, dtype=np.uint8)), dtype=np.uint8)

    learner = SequenceCortexOpponentLearner(**_SETTINGS)
    learner.observe_chunk_probabilities(stream, learn=True)

    never_occurred_key = _packed(b"ZZZ")  # uppercase never appears in lowercase-only stream
    for bank in learner.sequence_support:
        assert never_occurred_key not in bank


def test_row_level_pruning_keeps_the_reliable_context_and_drops_the_weak_one() -> None:
    """Two contexts: "qrs" reliably (90%) predicts one target, "xyz" is
    followed by a uniformly random byte every time (no reliable prediction
    at all). Pruning should drop the "xyz" row and keep "qrs" whole.

    Rows are pruned whole, not edge by edge: an edge-level version of this
    rule was tried first and measured worse than the plain count-based
    baseline on real text (~1.97 vs ~1.48 bits/byte held out) even at 3x
    the retained edges, because a row's surviving counts feed directly into
    the Bayesian backoff formula's denominator, which implicitly assumes a
    row represents the *complete* locally observed distribution -- pruning
    individual edges out of an otherwise-good row distorts backoff
    weighting for every prediction from that context, not just the pruned
    ones. A row-level ablation using the same support*log(count) signal
    (rows kept or dropped whole) closed most of that gap, which is what
    this now implements: a row's fate is decided by its single strongest
    edge's support*log(count) -- does this context have *some* reliable,
    well-confirmed predictive value.
    """
    rng = np.random.default_rng(3)
    filler = bytes(rng.integers(97, 122, size=6000, dtype=np.uint8))
    reliable_pieces = []
    for _ in range(90):
        if rng.random() < 0.9:
            reliable_pieces.append(b"qrsK")
        else:
            reliable_pieces.append(b"qrs" + bytes([int(rng.integers(65, 90))]))
    weak_pieces = [b"xyz" + bytes([int(rng.integers(65, 90))]) for _ in range(90)]
    stream = np.frombuffer(
        filler + b"".join(reliable_pieces) + b"".join(weak_pieces), dtype=np.uint8
    )
    learner = SequenceCortexOpponentLearner(**_SETTINGS)
    learner.observe_chunk_probabilities(stream, learn=True)

    reliable_key = _packed(b"qrs")
    weak_key = _packed(b"xyz")
    reliable_bank = weak_bank = None
    for index, bank in enumerate(learner.sequence_counts):
        if reliable_key in bank:
            reliable_bank = index
        if weak_key in bank:
            weak_bank = index
    assert reliable_bank is not None, "the reliable context should be mature"
    assert weak_bank is not None, "the weak context should also be mature"

    def row_utility(bank_index: int, context: int) -> float:
        row = learner.sequence_counts[bank_index][context]
        support_row = learner.sequence_support[bank_index].get(context, {})
        return max(support_row.get(t, 0.0) * math.log1p(c) for t, c in row.items())

    reliable_utility = row_utility(reliable_bank, reliable_key)
    weak_utility = row_utility(weak_bank, weak_key)
    assert reliable_utility > weak_utility, (
        f"the reliable context's row utility ({reliable_utility}) should rank above "
        f"the weak context's ({weak_utility}) -- consolidate_to_byte_budget prunes "
        "the lowest-utility rows first, so this ordering is what keeps the reliable "
        "context and drops the weak one"
    )

    # Confirm the same thing end to end through the real pruning path too, at
    # a scale where the byte-budget math has real headroom to work with
    # (rather than fighting pickle-size non-linearity at a few hundred
    # tracked rows): a much larger, denser stream forces genuine competition.
    #
    # max_sequence_nursery_paths is capped small here on purpose: nursery is
    # never pruned by consolidate_to_byte_budget (by design -- it is not
    # confirmed anatomy yet), and 60,000 bytes of high-entropy random
    # filler at max_context_bytes=4 mints huge numbers of near-unique
    # depth-4 contexts that would otherwise flood an uncapped nursery well
    # past the size of everything prunable, making no budget below that
    # floor achievable regardless of pruning strategy (confirmed directly
    # while debugging this test -- not a pruning bug, just the wrong choice
    # of parameters for what this test is actually checking).
    dense_settings = dict(_SETTINGS, max_sequence_nursery_paths=500)
    dense_rng = np.random.default_rng(30)
    dense_filler = bytes(dense_rng.integers(97, 122, size=60_000, dtype=np.uint8))
    dense_reliable = b"".join(
        b"qrsK" if dense_rng.random() < 0.9 else b"qrs" + bytes([int(dense_rng.integers(65, 90))])
        for _ in range(900)
    )
    dense_weak = b"".join(
        b"xyz" + bytes([int(dense_rng.integers(65, 90))]) for _ in range(900)
    )
    dense_stream = np.frombuffer(dense_filler + dense_reliable + dense_weak, dtype=np.uint8)
    dense_learner = SequenceCortexOpponentLearner(**dense_settings)
    dense_learner.observe_chunk_probabilities(dense_stream, learn=True)
    before = dense_learner.embodied_bytes
    dense_learner.consolidate_to_byte_budget(int(before * 0.42))

    reliable_survives = any(
        reliable_key in bank for bank in dense_learner.sequence_counts
    )
    weak_survives = any(weak_key in bank for bank in dense_learner.sequence_counts)
    assert reliable_survives, "the reliable context must survive a real budget-driven prune"
    assert not weak_survives, "the weak context should be pruned by a real budget-driven prune"


def test_nursery_fifo_eviction_cleans_up_its_support_entry() -> None:
    """Regression: a FIFO-evicted nursery context's support entry used to
    leak (never cleaned up), and with the nursery persisting across sleeps
    and churning constantly, that leak dominated embodied_bytes over a real
    run (~3x the count-based baseline's anatomy size for identical input,
    confirmed directly, before this was fixed)."""
    rng = np.random.default_rng(4)
    filler = bytes(rng.integers(97, 122, size=3000, dtype=np.uint8))
    learner = SequenceCortexOpponentLearner(
        confirmations=1,
        context_confirmations=1,
        sequence_confirmations=1000,  # never promote -- keep everything in nursery
        max_context_bytes=4,
        max_sequence_nursery_paths=50,  # small cap forces real FIFO churn
        structural_support=True,
        structural_support_rate=0.2,
    )
    learner.observe_chunk_probabilities(np.frombuffer(filler, dtype=np.uint8), learn=True)

    assert learner.sequence_nursery_evictions > 0, "this scenario should force real FIFO churn"
    for bank_index, bank in enumerate(learner.sequence_nursery):
        live_contexts = set(bank)
        stale = set(learner.sequence_support[bank_index]) - live_contexts
        assert not stale, f"bank {bank_index} leaked support for evicted contexts: {stale}"


def test_structural_support_is_opt_in_and_does_not_change_default_behavior() -> None:
    stream = np.frombuffer(b"the quick brown fox jumps over the lazy dog. " * 20, dtype=np.uint8)
    reference = SequenceCortexOpponentLearner()
    with_flag_off = SequenceCortexOpponentLearner(structural_support=False)

    expected = reference.observe_chunk_probabilities(stream, learn=True)
    actual = with_flag_off.observe_chunk_probabilities(stream, learn=True)
    np.testing.assert_array_equal(expected, actual)
    assert all(bank == {} for bank in with_flag_off.sequence_support)
