from __future__ import annotations

from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import (
    _lexical_outcome,
)


def test_lexical_outcome_credits_the_first_relational_completion() -> None:
    assert (
        _lexical_outcome(
            b" the red ball, not the blue kite", b" red ball", b" blue kite"
        )
        == "expected"
    )
    assert (
        _lexical_outcome(b" the blue kite", b" red ball", b" blue kite")
        == "contrast"
    )


def test_lexical_outcome_does_not_invent_evidence() -> None:
    assert _lexical_outcome(b" she smiled", b" red ball", b" blue kite") == "neither"
