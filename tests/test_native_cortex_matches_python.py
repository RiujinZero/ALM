"""The native cortex must be a bit-exact stand-in for the Python learner.

Speed is only worth having if the organism is unchanged, so these compare
raw IEEE-754 bits rather than a tolerance: every probability the native
backend emits has to equal the Python reference exactly, and the developed
anatomy has to match count for count.
"""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)

native_cortex = pytest.importorskip("manifoldflow.dendritic_basin.native_cortex")

pytestmark = pytest.mark.skipif(
    not native_cortex.native_available(),
    reason="alm_native shared library is not built (see cpp/README.md)",
)


def _language_like_stream(byte_count: int, seed: int = 20260911) -> np.ndarray:
    """Text-shaped bytes: repeated vocabulary, varied order, real spacing.

    A uniform random stream would never grow deep sequence anatomy, so it
    would exercise none of the paths that matter here.
    """
    rng = np.random.default_rng(seed)
    words = [
        b"the", b"little", b"girl", b"named", b"lily", b"went", b"to", b"the",
        b"park", b"and", b"played", b"with", b"her", b"dog", b"tom", b"was",
        b"very", b"happy", b"they", b"found", b"a", b"red", b"ball", b"then",
    ]
    out = bytearray()
    while len(out) < byte_count:
        sentence_length = int(rng.integers(4, 12))
        for _ in range(sentence_length):
            out += words[int(rng.integers(0, len(words)))]
            out += b" "
        out += b".\n"
    return np.frombuffer(bytes(out[:byte_count]), dtype=np.uint8)


def _mismatches(left: np.ndarray, right: np.ndarray) -> int:
    both_nan = np.isnan(left) & np.isnan(right)
    return int(np.sum(left[~both_nan] != right[~both_nan]))


def test_growable_matches_python_bit_for_bit() -> None:
    stream = _language_like_stream(300_000)

    reference = SequenceCortexOpponentLearner()
    native = native_cortex.NativeGrowableCortex()

    expected = reference.observe_chunk_probabilities(stream, learn=True)
    actual = native.observe_chunk_probabilities(stream, learn=True)

    assert _mismatches(expected, actual) == 0
    assert np.isnan(expected).sum() == np.isnan(actual).sum()


def test_growable_matches_python_anatomy() -> None:
    stream = _language_like_stream(300_000)

    reference = SequenceCortexOpponentLearner()
    native = native_cortex.NativeGrowableCortex()
    reference.observe_chunk_probabilities(stream, learn=True)
    native.observe_chunk_probabilities(stream, learn=True)

    assert native.steps == reference.steps
    assert native.live_sequence_count == reference.live_sequence_count
    assert native.live_context_count == reference.live_context_count
    assert native.sequence_births == reference.sequence_births
    assert native.context_births == reference.context_births
    assert native.sequence_nursery_count == reference.sequence_nursery_count


def test_nursery_eviction_order_matches_python() -> None:
    """FIFO eviction decides which paths ever mature, so it must agree.

    A tight nursery forces constant eviction; if the native side evicted a
    different key than Python's `next(iter(dict))` the two anatomies would
    diverge within a few thousand bytes.
    """
    stream = _language_like_stream(200_000, seed=7)
    settings = {"max_sequence_nursery_paths": 256}

    reference = SequenceCortexOpponentLearner(**settings)
    native = native_cortex.NativeGrowableCortex(**settings)

    expected = reference.observe_chunk_probabilities(stream, learn=True)
    actual = native.observe_chunk_probabilities(stream, learn=True)

    assert reference.sequence_nursery_evictions > 0
    assert native.sequence_nursery_evictions == reference.sequence_nursery_evictions
    assert native.sequence_births == reference.sequence_births
    assert _mismatches(expected, actual) == 0


def test_dense_row_promotion_matches_python() -> None:
    """Rows past the dense threshold switch representation; counts must not."""
    rng = np.random.default_rng(99)
    # Every byte follows every byte, so depth-3 rows fill out well past the
    # sparse/dense crossover.
    stream = rng.integers(0, 256, size=120_000, dtype=np.uint8)

    reference = SequenceCortexOpponentLearner()
    native = native_cortex.NativeGrowableCortex()

    expected = reference.observe_chunk_probabilities(stream, learn=True)
    actual = native.observe_chunk_probabilities(stream, learn=True)

    assert _mismatches(expected, actual) == 0


def test_packed_scores_identically_to_the_body_it_came_from() -> None:
    train = _language_like_stream(200_000)
    held_out = _language_like_stream(50_000, seed=4242)

    reference = SequenceCortexOpponentLearner()
    native = native_cortex.NativeGrowableCortex()
    reference.observe_chunk_probabilities(train, learn=True)
    native.observe_chunk_probabilities(train, learn=True)

    packed = native.pack()

    reference.reset_episode()
    native.reset_episode()
    packed.reset_episode()

    expected = reference.observe_chunk_probabilities(held_out, learn=False)
    growable = native.observe_chunk_probabilities(held_out, learn=False)
    frozen = packed.observe_chunk_probabilities(held_out)

    assert _mismatches(expected, growable) == 0
    assert _mismatches(expected, frozen) == 0
    assert packed.live_sequence_count == reference.live_sequence_count
    assert packed.live_context_count == reference.live_context_count


def test_packed_body_is_smaller_than_the_growable_one() -> None:
    native = native_cortex.NativeGrowableCortex()
    native.observe_chunk_probabilities(_language_like_stream(200_000), learn=True)

    packed = native.pack()

    assert packed.memory_bytes < native.memory_bytes


def test_packed_cortex_refuses_to_learn() -> None:
    native = native_cortex.NativeGrowableCortex()
    native.observe_chunk_probabilities(_language_like_stream(20_000), learn=True)
    packed = native.pack()

    with pytest.raises(ValueError):
        packed.observe_chunk_probabilities(
            _language_like_stream(1_000), learn=True
        )


def test_episode_reset_matches_python() -> None:
    stream = _language_like_stream(60_000)
    tail = _language_like_stream(20_000, seed=11)

    reference = SequenceCortexOpponentLearner()
    native = native_cortex.NativeGrowableCortex()
    reference.observe_chunk_probabilities(stream, learn=True)
    native.observe_chunk_probabilities(stream, learn=True)

    reference.reset_episode()
    native.reset_episode()

    expected = reference.observe_chunk_probabilities(tail, learn=True)
    actual = native.observe_chunk_probabilities(tail, learn=True)

    assert _mismatches(expected, actual) == 0


def test_predict_next_distribution_matches_python_bit_for_bit() -> None:
    """The generation-time query: full 256-way, no side effects."""
    stream = _language_like_stream(150_000, seed=17)
    prompt = b"Once upon a time"

    reference = SequenceCortexOpponentLearner()
    native = native_cortex.NativeGrowableCortex()
    reference.observe_chunk_probabilities(stream, learn=True)
    native.observe_chunk_probabilities(stream, learn=True)
    reference.reset_episode()
    native.reset_episode()
    prompt_bytes = np.frombuffer(prompt, dtype=np.uint8)
    reference.observe_chunk_probabilities(prompt_bytes, learn=False)
    native.observe_chunk_probabilities(prompt_bytes, learn=False)

    expected = reference.predict_next_distribution()
    actual = native.predict_next_distribution()
    assert _mismatches(expected, actual) == 0
    assert abs(float(actual.sum()) - 1.0) < 1e-9


def test_advance_observation_matches_python_and_moves_state() -> None:
    stream = _language_like_stream(100_000, seed=18)

    reference = SequenceCortexOpponentLearner()
    native = native_cortex.NativeGrowableCortex()
    reference.observe_chunk_probabilities(stream, learn=True)
    native.observe_chunk_probabilities(stream, learn=True)
    reference.reset_episode()
    native.reset_episode()

    for byte_id in (111, 32, 84, 104, 101):
        expected = reference.advance_observation(byte_id, learn=False)
        actual = native.advance_observation(byte_id, learn=False)
        assert expected == actual

    # State actually advanced: predictions differ from a freshly reset cortex.
    fresh = native_cortex.NativeGrowableCortex.from_bytes(native.to_bytes())
    fresh.reset_episode()
    moved = native.predict_next_distribution()
    reset = fresh.predict_next_distribution()
    assert _mismatches(moved, reset) > 0


def test_shed_sequence_nursery_matches_python() -> None:
    stream = _language_like_stream(150_000)

    reference = SequenceCortexOpponentLearner()
    native = native_cortex.NativeGrowableCortex()
    reference.observe_chunk_probabilities(stream, learn=True)
    native.observe_chunk_probabilities(stream, learn=True)

    assert native.shed_sequence_nursery() == reference.shed_sequence_nursery()
    assert native.sequence_nursery_count == 0
    assert native.live_sequence_count == reference.live_sequence_count
