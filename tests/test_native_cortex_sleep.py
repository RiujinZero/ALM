"""consolidate_to_byte_budget: the sleep/pruning step.

Byte budgets are not directly comparable between backends (native's
"embodied bytes" is its own serialized size, not a Python pickle byte
count -- see native_cortex.py's docstring), so these test properties of the
pruning decision itself rather than byte-for-byte agreement with Python.
"""

from __future__ import annotations

import numpy as np
import pytest

native_cortex = pytest.importorskip("manifoldflow.dendritic_basin.native_cortex")

pytestmark = pytest.mark.skipif(
    not native_cortex.native_available(),
    reason="alm_native shared library is not built (see cpp/README.md)",
)

_WORDS = [
    b"the", b"little", b"girl", b"named", b"lily", b"went", b"to", b"the",
    b"park", b"and", b"played", b"with", b"her", b"dog", b"tom", b"was",
    b"very", b"happy", b"they", b"found", b"a", b"red", b"ball", b"then",
]


def _language_like_stream(byte_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = bytearray()
    while len(out) < byte_count:
        for _ in range(int(rng.integers(4, 12))):
            out += _WORDS[int(rng.integers(0, len(_WORDS)))]
            out += b" "
        out += b".\n"
    return np.frombuffer(bytes(out[:byte_count]), dtype=np.uint8)


def test_sleep_lands_at_or_under_the_requested_budget() -> None:
    stream = _language_like_stream(500_000, seed=1)
    cortex = native_cortex.NativeGrowableCortex()
    cortex.observe_chunk_probabilities(stream, learn=True)
    before = cortex.embodied_bytes

    result = cortex.consolidate_to_byte_budget(before // 2)

    assert result["body_bytes_before_sleep"] == before
    assert result["body_bytes_after_sleep"] <= before // 2
    assert cortex.embodied_bytes == result["body_bytes_after_sleep"]
    assert result["body_bytes_after_nursery_shed"] <= before
    assert result["shed_nursery_paths"] >= 0
    assert result["pruned_mature_paths"] >= 0


def test_sleep_no_longer_sheds_the_nursery() -> None:
    """The nursery used to be wiped unconditionally on every sleep, which
    discarded provisional candidates before consolidate_delta's cross-column
    merge (merge_sequence_bank in consolidate.hpp) ever gave them a real
    chance to accumulate sequence_confirmations worth of observations -- a
    long run's mature anatomy would go static almost as soon as it first hit
    the byte cap, with no mechanism to keep replacing weak mature paths with
    stronger new candidates the way the vision organism's utility-ranked
    eviction does. Now the nursery persists across sleep, and any byte
    reduction has to come from pruning mature paths directly."""
    stream = _language_like_stream(300_000, seed=2)
    cortex = native_cortex.NativeGrowableCortex()
    cortex.observe_chunk_probabilities(stream, learn=True)
    nursery_before = cortex.sequence_nursery_count
    assert nursery_before > 0  # otherwise this test proves nothing

    # A budget just under the pre-sleep size: with no nursery shed to lean
    # on, even this small a reduction must prune mature anatomy directly.
    result = cortex.consolidate_to_byte_budget(max(cortex.embodied_bytes - 1, 1))
    assert result["shed_nursery_paths"] == 0
    assert result["body_bytes_after_nursery_shed"] == result["body_bytes_before_sleep"]
    assert result["pruned_mature_paths"] > 0
    assert cortex.sequence_nursery_count == nursery_before


def test_sleep_keeps_the_organism_functional() -> None:
    """Predictions after a hard prune should still be sane, not degenerate."""
    train = _language_like_stream(500_000, seed=3)
    held_out = _language_like_stream(100_000, seed=4)

    cortex = native_cortex.NativeGrowableCortex()
    cortex.observe_chunk_probabilities(train, learn=True)
    cortex.consolidate_to_byte_budget(cortex.embodied_bytes // 3)

    cortex.reset_episode()
    probabilities = cortex.observe_chunk_probabilities(held_out, learn=False)
    finite = probabilities[np.isfinite(probabilities)]
    bits_per_byte = float(np.mean(-np.log2(np.maximum(finite, 1e-300))))

    assert 0.0 < bits_per_byte < 8.0  # a real distribution, not NaN/degenerate


def test_a_repeatedly_reinforced_context_survives_a_light_prune() -> None:
    """The documented ranking (strongest support first) should show up in
    practice: a heavily repeated pattern must outlive a rarely seen one."""

    def repeat(pattern: bytes, times: int) -> bytes:
        return pattern * times

    strong = repeat(b"AAAB", 400)
    weak = repeat(b"CCCD", 5)
    stream = np.frombuffer(strong + weak, dtype=np.uint8)

    cortex = native_cortex.NativeGrowableCortex(
        confirmations=1, context_confirmations=1, sequence_confirmations=3
    )
    cortex.observe_chunk_probabilities(stream, learn=True)

    # Find the smallest post-nursery-shed size, then prune moderately below
    # it so some (not all) mature sequence anatomy is actually removed.
    probe = native_cortex.NativeGrowableCortex.from_bytes(cortex.to_bytes())
    probe.consolidate_to_byte_budget(10**9)
    after_shed = probe.embodied_bytes

    cortex = native_cortex.NativeGrowableCortex.from_bytes(cortex.to_bytes())
    try:
        result = cortex.consolidate_to_byte_budget(int(after_shed * 0.85))
    except ValueError:
        pytest.skip("scenario too small to force a partial mature prune")
    if result["pruned_mature_paths"] == 0:
        pytest.skip("scenario too small to force a partial mature prune")

    def probe_probability(source, prefix: bytes, byte_id: int) -> float:
        clone = native_cortex.NativeGrowableCortex.from_bytes(source.to_bytes())
        clone.reset_episode()
        clone.observe_chunk_probabilities(
            np.frombuffer(prefix, dtype=np.uint8), learn=False
        )
        probe_stream = np.array([byte_id], dtype=np.uint8)
        return float(clone.observe_chunk_probabilities(probe_stream, learn=False)[0])

    p_strong = probe_probability(cortex, b"AAA", ord("B"))
    assert p_strong > 0.9
