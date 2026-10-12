from __future__ import annotations

from manifoldflow.experiments.dendritic_basin_phase72_index_diagnostic import (
    _matching_prefix,
    scramble_distant_prefix,
)


def test_scramble_distant_prefix_preserves_recent_evidence_and_byte_counts() -> None:
    prefix = bytes(range(64))
    scrambled = scramble_distant_prefix(prefix, preserved_recent_bytes=16, seed=7)

    assert scrambled[-16:] == prefix[-16:]
    assert sorted(scrambled[:-16]) == sorted(prefix[:-16])
    assert scrambled[:-16] != prefix[:-16]


def test_matching_prefix_locates_first_autonomous_divergence() -> None:
    assert _matching_prefix(b"abcdef", b"abcxef") == 3
    assert _matching_prefix(b"abc", b"abcde") == 3
