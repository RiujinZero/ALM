"""consolidate_delta, native checkpoints, and the adaptive-competition port.

These extend test_native_cortex_matches_python.py's bit-exactness contract
to the pieces needed for the actual multi-column wake/sleep harness: merging
several columns cloned from one ancestor, native save/load, and the
trust/TD-credit/attribution mechanism layered on the residual/context level.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)

native_cortex = pytest.importorskip("manifoldflow.dendritic_basin.native_cortex")
native_import = pytest.importorskip("manifoldflow.dendritic_basin.native_cortex_import")

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


def _mismatches(left: np.ndarray, right: np.ndarray) -> int:
    both_nan = np.isnan(left) & np.isnan(right)
    return int(np.sum(left[~both_nan] != right[~both_nan]))


def test_native_checkpoint_round_trips_mid_episode() -> None:
    data = _language_like_stream(150_000, seed=1)
    tail = _language_like_stream(80_000, seed=2)

    native = native_cortex.NativeGrowableCortex()
    native.observe_chunk_probabilities(data, learn=True)
    reloaded = native_cortex.NativeGrowableCortex.from_bytes(native.to_bytes())

    # Neither side is reset: this checks that episode state (previous/older
    # byte, packed history) survives the round trip, not just anatomy.
    expected = native.observe_chunk_probabilities(tail, learn=False)
    actual = reloaded.observe_chunk_probabilities(tail, learn=False)
    assert _mismatches(expected, actual) == 0


def test_consolidate_delta_matches_python_on_two_columns() -> None:
    """The actual wake/sleep merge shape: N columns cloned from one ancestor.

    ``max_sequence_nursery_paths`` is left at a realistic (large) capacity
    on purpose: a small cap lets a column evict a key its ancestor still
    remembers mid-nursery, which fails with "descendant anatomy predates
    its ancestor" in the pure-Python reference too -- a pre-existing edge
    case in the merge algorithm itself, not something this test is for.
    """
    ancestor_stream = _language_like_stream(60_000, seed=7)
    settings = {"max_sequence_nursery_paths": 50_000}

    py_ancestor = SequenceCortexOpponentLearner(**settings)
    py_ancestor.observe_chunk_probabilities(ancestor_stream, learn=True)
    py_ancestor.reset_episode()
    nat_ancestor = native_cortex.NativeGrowableCortex(**settings)
    nat_ancestor.observe_chunk_probabilities(ancestor_stream, learn=True)
    nat_ancestor.reset_episode()

    py_master = copy.deepcopy(py_ancestor)
    nat_master = native_cortex.NativeGrowableCortex.from_bytes(nat_ancestor.to_bytes())

    for seed in (10, 11, 12):
        column_stream = _language_like_stream(25_000, seed=seed)

        py_column = copy.deepcopy(py_ancestor)
        py_column.begin_lineage_tracking()
        py_column.observe_chunk_probabilities(column_stream, learn=True)

        nat_column = native_cortex.NativeGrowableCortex.from_bytes(
            nat_ancestor.to_bytes()
        )
        nat_column.begin_lineage_tracking()
        nat_column.observe_chunk_probabilities(column_stream, learn=True)

        py_master.consolidate_delta(py_column, py_ancestor)
        nat_master.consolidate_delta(nat_column, nat_ancestor)

    assert nat_master.live_sequence_count == py_master.live_sequence_count
    assert nat_master.live_context_count == py_master.live_context_count
    assert nat_master.sequence_births == py_master.sequence_births
    assert nat_master.context_births == py_master.context_births
    assert nat_master.sequence_nursery_evictions == py_master.sequence_nursery_evictions
    assert nat_master.steps == py_master.steps

    held_out = _language_like_stream(40_000, seed=99)
    py_master.reset_episode()
    nat_master.reset_episode()
    expected = py_master.observe_chunk_probabilities(held_out, learn=False)
    actual = nat_master.observe_chunk_probabilities(held_out, learn=False)
    assert _mismatches(expected, actual) == 0


def test_persistent_nursery_lets_two_columns_jointly_promote_a_context() -> None:
    """The mechanism the vision-parity fix depends on: consolidate_delta
    already merges nursery observations across columns (merge_sequence_bank
    in consolidate.hpp), so a context two columns each see below
    sequence_confirmations individually can cross the threshold once merged.
    That capability was previously undermined by wiping the whole nursery on
    every sleep before it ever got a chance to matter -- this test locks in
    the merge behavior itself, independent of the sleep-side fix."""
    settings = {"sequence_confirmations": 4}
    warm = _language_like_stream(60_000, seed=3)

    def build(kind):
        learner = kind(**settings)
        learner.observe_chunk_probabilities(warm, learn=True)
        learner.reset_episode()
        return learner

    py_ancestor = build(SequenceCortexOpponentLearner)
    nat_ancestor = native_cortex.NativeGrowableCortex(**settings)
    nat_ancestor.observe_chunk_probabilities(warm, learn=True)
    nat_ancestor.reset_episode()

    # 'k', 'p', 'd', and 'e' are already live from the warm stream.
    novel = np.frombuffer(b"kpde", dtype=np.uint8)

    def half_column(kind, ancestor):
        column = copy.deepcopy(ancestor) if kind is SequenceCortexOpponentLearner else (
            native_cortex.NativeGrowableCortex.from_bytes(ancestor.to_bytes())
        )
        column.begin_lineage_tracking()
        column.observe_chunk_probabilities(novel, learn=True)
        column.reset_episode()
        column.observe_chunk_probabilities(novel, learn=True)
        return column

    py_a = half_column(SequenceCortexOpponentLearner, py_ancestor)
    py_b = half_column(SequenceCortexOpponentLearner, py_ancestor)
    nat_a = half_column(native_cortex.NativeGrowableCortex, nat_ancestor)
    nat_b = half_column(native_cortex.NativeGrowableCortex, nat_ancestor)

    py_master = copy.deepcopy(py_ancestor)
    py_master.consolidate_delta(py_a, py_ancestor)
    py_master.consolidate_delta(py_b, py_ancestor)

    nat_master = native_cortex.NativeGrowableCortex.from_bytes(nat_ancestor.to_bytes())
    nat_master.consolidate_delta(nat_a, nat_ancestor)
    nat_master.consolidate_delta(nat_b, nat_ancestor)

    # Neither column alone reached sequence_confirmations=4 (2 observations
    # each), but merged they total 4 and should have jointly promoted.
    assert py_master.live_sequence_count > py_ancestor.live_sequence_count
    assert nat_master.live_sequence_count > nat_ancestor.live_sequence_count
    assert nat_master.live_sequence_count == py_master.live_sequence_count


def test_lineage_tracking_survives_a_checkpoint_round_trip() -> None:
    """Regression: the real harness always saves a column to disk and
    reloads it before merging. CortexIO used to silently drop
    begin_lineage_tracking()'s touched-key state on that round trip, which
    made consolidate_delta index an empty, wrongly-sized internal vector --
    a real access violation, not a Python exception. This is the exact
    save-then-reload-then-merge shape that crashed.
    """
    ancestor = native_cortex.NativeGrowableCortex()
    ancestor.observe_chunk_probabilities(
        _language_like_stream(40_000, seed=20), learn=True
    )
    ancestor.reset_episode()

    reloaded_ancestor = native_cortex.NativeGrowableCortex.from_bytes(
        ancestor.to_bytes()
    )

    column = native_cortex.NativeGrowableCortex.from_bytes(ancestor.to_bytes())
    column.begin_lineage_tracking()
    column.observe_chunk_probabilities(
        _language_like_stream(20_000, seed=21), learn=True
    )
    column.reset_episode()

    # The crash-triggering step: round-trip the column through a checkpoint
    # (as _atomic_pickle/_load_any_cortex do) before merging it.
    descendant = native_cortex.NativeGrowableCortex.from_bytes(column.to_bytes())

    master = native_cortex.NativeGrowableCortex.from_bytes(ancestor.to_bytes())
    master.consolidate_delta(descendant, reloaded_ancestor)  # must not crash
    assert master.live_sequence_count >= ancestor.live_sequence_count


def test_consolidate_delta_rejects_missing_lineage_tracking() -> None:
    """A descendant that never had begin_lineage_tracking() called (or lost
    it some other way) must fail with a clear error, not corrupt memory."""
    ancestor = native_cortex.NativeGrowableCortex()
    ancestor.observe_chunk_probabilities(
        _language_like_stream(10_000, seed=22), learn=True
    )
    ancestor.reset_episode()

    descendant = native_cortex.NativeGrowableCortex.from_bytes(ancestor.to_bytes())
    descendant.observe_chunk_probabilities(
        _language_like_stream(5_000, seed=23), learn=True
    )  # no begin_lineage_tracking()

    master = native_cortex.NativeGrowableCortex.from_bytes(ancestor.to_bytes())
    with pytest.raises(ValueError):
        master.consolidate_delta(descendant, ancestor)


def test_consolidate_delta_rejects_mismatched_configuration() -> None:
    ancestor = native_cortex.NativeGrowableCortex(max_sequence_nursery_paths=100)
    descendant = native_cortex.NativeGrowableCortex(max_sequence_nursery_paths=100)
    self_cortex = native_cortex.NativeGrowableCortex(max_sequence_nursery_paths=999)

    with pytest.raises(ValueError):
        self_cortex.consolidate_delta(descendant, ancestor)


def test_adaptive_competition_development_matches_python_bit_for_bit() -> None:
    stream = _language_like_stream(200_000, seed=3)

    reference = SequenceCortexOpponentLearner(adaptive_competition=True)
    native = native_cortex.NativeGrowableCortex(adaptive_competition=True)

    expected = reference.observe_chunk_probabilities(stream, learn=True)
    actual = native.observe_chunk_probabilities(stream, learn=True)

    assert _mismatches(expected, actual) == 0
    assert len(reference.context_trust) > 0


def test_frozen_inference_adaptation_makes_counterfactual_scoring_read_only() -> None:
    """Candidate branches must not credit trust or decay burst state.

    ``learn=False`` intentionally remains adaptive during normal deployment,
    but that behavior contaminates a counterfactual assay that reuses one
    cortex for several candidates. The runtime debug freeze retains history
    advance while leaving every serialized adaptive-state bit untouched.
    """
    cortex = native_cortex.NativeGrowableCortex(
        adaptive_competition=True,
        burst_surprise_signal=True,
        confirmations=1,
        context_confirmations=1,
        sequence_confirmations=1,
    )
    cortex.observe_chunk_probabilities(
        _language_like_stream(20_000, seed=31), learn=True
    )
    cortex.reset_episode()
    assert cortex.burst_fraction > 0.0

    cortex.set_debug_freeze_inference_adaptation(True)
    burst_before = cortex.burst_fraction
    serialized_before = cortex.to_bytes()
    cortex.observe_chunk_probabilities(
        _language_like_stream(10_000, seed=32), learn=False
    )
    cortex.reset_episode()

    assert cortex.burst_fraction == burst_before
    assert cortex.to_bytes() == serialized_before


def test_import_preserves_learned_trust_bit_for_bit() -> None:
    """The actual scenario this was built for: importing a real checkpoint
    whose predictions depend on previously learned trust values, and
    continuing to score identically afterward."""
    train = _language_like_stream(150_000, seed=4)
    held_out = _language_like_stream(60_000, seed=5)

    reference = SequenceCortexOpponentLearner(adaptive_competition=True)
    reference.observe_chunk_probabilities(train, learn=True)
    assert len(reference.context_trust) > 0  # the case actually worth testing

    imported = native_import.native_cortex_from_python(reference)

    reference.reset_episode()
    imported.reset_episode()
    expected = reference.observe_chunk_probabilities(held_out, learn=False)
    actual = imported.observe_chunk_probabilities(held_out, learn=False)
    assert _mismatches(expected, actual) == 0


def test_import_matches_with_adaptive_competition_forced_off_too() -> None:
    """A checkpoint predating this mechanism (no context_trust at all) must
    still import and score correctly."""
    train = _language_like_stream(100_000, seed=6)
    held_out = _language_like_stream(30_000, seed=8)

    reference = SequenceCortexOpponentLearner()  # adaptive_competition defaults off
    reference.observe_chunk_probabilities(train, learn=True)

    imported = native_import.native_cortex_from_python(reference)

    reference.reset_episode()
    imported.reset_episode()
    expected = reference.observe_chunk_probabilities(held_out, learn=False)
    actual = imported.observe_chunk_probabilities(held_out, learn=False)
    assert _mismatches(expected, actual) == 0
