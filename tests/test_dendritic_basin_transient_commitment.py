from __future__ import annotations

import pytest

from manifoldflow.dendritic_basin.transient_commitment import (
    TransientRecallCommitment,
)


def test_external_cue_recruits_then_activity_exhausts_commitment() -> None:
    trace = TransientRecallCommitment(half_life_steps=5.0, extinction=0.30)

    assert trace.recruit(external=True)
    trace.advance(8)
    assert trace.active
    trace.advance()
    assert not trace.active


def test_internal_activity_cannot_refresh_commitment() -> None:
    trace = TransientRecallCommitment(half_life_steps=5.0, extinction=0.30)
    trace.recruit(external=True)
    trace.advance(8)
    before = trace.level

    assert not trace.recruit(external=False)
    assert trace.level == before


def test_commitment_rejects_invalid_anatomy() -> None:
    with pytest.raises(ValueError, match="half-life"):
        TransientRecallCommitment(half_life_steps=0.0)
    with pytest.raises(ValueError, match="extinction"):
        TransientRecallCommitment(extinction=1.0)

    trace = TransientRecallCommitment()
    with pytest.raises(ValueError, match="steps"):
        trace.advance(-1)
