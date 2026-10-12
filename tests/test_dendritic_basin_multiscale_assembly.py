from __future__ import annotations

import pytest

from manifoldflow.dendritic_basin.multiscale_assembly import MultiscaleAssemblyField
from manifoldflow.experiments.dendritic_basin_phase76_recurrent_field import (
    _temporal_cortical_context,
)


def test_constituent_and_larger_motif_can_coactivate() -> None:
    field = MultiscaleAssemblyField(activation_threshold=1.0)
    field.bind(1, b"ball")
    field.bind(2, b"red ball")

    at_ball = ()
    for byte_id in b"a red ball":
        active = field.observe(byte_id)
        if bytes(field.history).endswith(b"red ball"):
            at_ball = active

    assert [item.assembly for item in at_ball] == [2, 1]


def test_partial_overlap_is_graded_but_thresholded() -> None:
    field = MultiscaleAssemblyField(activation_threshold=0.75)
    field.bind(7, b"kite")

    active = ()
    for byte_id in b"bite":
        active = field.observe(byte_id)

    assert active[0].assembly == 7
    assert active[0].support == 0.75


def test_unrelated_activity_does_not_activate_pattern() -> None:
    field = MultiscaleAssemblyField(activation_threshold=0.75)
    field.bind(3, b"blue")

    active = ()
    for byte_id in b"tree":
        active = field.observe(byte_id)

    assert active == ()


def test_factorized_context_preserves_identity_across_temporal_lags() -> None:
    early = _temporal_cortical_context((((7, 3),),), include_identity=True)
    late = _temporal_cortical_context(
        (((7, 3),), ((8, 4),)), include_identity=True
    )

    assert (7, 2_000_000) in early
    assert (7, 2_000_000) in late
    assert set(early) != set(late)


def test_recent_assembly_remains_accessible_without_staying_sensory() -> None:
    field = MultiscaleAssemblyField(
        activation_threshold=1.0,
        trace_half_life_bytes=2.0,
        minimum_trace=0.2,
    )
    field.bind(9, b"cat")
    for byte_id in b"cat":
        field.observe(byte_id)

    one_later = field.observe(ord("x"))
    two_later = field.observe(ord("y"))

    assert one_later[0].assembly == 9
    assert one_later[0].matched_synapses == 0
    assert two_later[0].support == pytest.approx(0.5)
