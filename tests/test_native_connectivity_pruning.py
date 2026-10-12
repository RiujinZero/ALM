"""connectivity_pruning: native vs Python agreement on pruning behavior.

Byte budgets are not directly comparable between backends (native's
"embodied bytes" is its own serialized size, not a Python pickle byte
count -- see native_cortex.py's docstring), so this follows
test_native_cortex_sleep.py's established style: check the pruning
*decision* through predicted probabilities on both backends, rather than
byte-for-byte agreement. connectivity_pruning does not touch growth,
promotion, or the probability math at all -- only which mature rows a
byte-budget overflow evicts first -- so both backends should keep a
"hub" context (one many longer contexts back off through) alive through
an aggressive prune, on top of the counts machinery already verified
bit-exact elsewhere.
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


def _hub_and_filler_stream(seed: int) -> np.ndarray:
    """"the" is followed by 20 distinct bytes (many depth-4 children back
    off through it -- a hub); "zzq" is always followed by the same single
    byte (no depth-4 variety, effectively isolated at depth 4+), with the
    same total observation count as "the"."""
    rng = np.random.default_rng(seed)
    filler = bytes(rng.integers(97, 122, size=8000, dtype=np.uint8))
    hub_pieces = b"".join(b"the" + bytes([97 + (i % 20)]) for i in range(60))
    isolated_pieces = b"".join(b"zzq" + bytes([120]) for _ in range(60))
    return np.frombuffer(filler + hub_pieces + isolated_pieces, dtype=np.uint8)


def test_native_still_favors_the_hub_context_after_an_aggressive_prune() -> None:
    stream = _hub_and_filler_stream(seed=7)
    settings = dict(
        confirmations=1,
        context_confirmations=1,
        sequence_confirmations=2,
        max_context_bytes=4,
        connectivity_pruning=True,
        max_sequence_nursery_paths=500,
    )

    py_learner = SequenceCortexOpponentLearner(**settings)
    py_learner.observe_chunk_probabilities(stream, learn=True)
    nat_learner = native_cortex.NativeGrowableCortex(**settings)
    nat_learner.observe_chunk_probabilities(stream, learn=True)

    assert py_learner.live_sequence_count == nat_learner.live_sequence_count, (
        "growth/promotion must be identical regardless of connectivity_pruning"
    )

    py_before = py_learner.embodied_bytes
    nat_before = nat_learner.embodied_bytes
    py_learner.consolidate_to_byte_budget(max(int(py_before * 0.92), 1))
    nat_learner.consolidate_to_byte_budget(max(int(nat_before * 0.92), 1))

    def probe(learner, prefix: bytes) -> np.ndarray:
        learner.reset_episode()
        learner.observe_chunk_probabilities(
            np.frombuffer(prefix, dtype=np.uint8), learn=False
        )
        # 'a' through 't' cover every hub continuation byte used above.
        probe_stream = np.frombuffer(b"a", dtype=np.uint8)
        return learner.observe_chunk_probabilities(probe_stream, learn=False)

    py_prob = probe(py_learner, b"the")
    nat_prob = probe(nat_learner, b"the")

    # Both backends should have kept enough of the hub row's structure that
    # predicting after "the" is well above a uniform 1/256 guess -- the
    # concrete, checkable form of "connectivity pruning favored the hub."
    assert float(py_prob[0]) > 1.0 / 256, "python should still predict through the hub"
    assert float(nat_prob[0]) > 1.0 / 256, "native should still predict through the hub"
