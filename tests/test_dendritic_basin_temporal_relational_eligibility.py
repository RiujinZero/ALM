from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.temporal_relational_eligibility import (
    TemporalRelationalEligibility,
)
from manifoldflow.experiments import (
    dendritic_basin_temporal_relational_eligibility_smoke as phase133,
)


def test_trace_persists_without_persistent_content() -> None:
    learner = TemporalRelationalEligibility(half_lives=(8.0, 32.0))
    learner.participate("relation", 1.0)
    learner.advance(8)

    np.testing.assert_allclose(learner.trace("relation"), (0.5, 2**-0.25))


def test_only_external_consequence_changes_pathway() -> None:
    learner = TemporalRelationalEligibility()
    learner.participate("relation", 1.0)
    learner.confirm({"relation": 2.0}, external=False)
    np.testing.assert_array_equal(learner.strength("relation"), 0.0)

    learner.confirm({"relation": 2.0}, external=True)
    assert np.all(learner.strength("relation") > 0.0)


def test_signed_external_utility_can_depress_a_participating_pathway() -> None:
    learner = TemporalRelationalEligibility()
    learner.participate("relation", 1.0)

    learner.confirm({"relation": -0.5}, external=True)

    assert np.all(learner.strength("relation") < 0.0)
    assert learner.negative_updates == 1
    assert learner.permission("relation") == 0.0


def test_incompatible_relation_accelerates_old_trace_decay() -> None:
    learner = TemporalRelationalEligibility(conflict_decay=0.25)
    learner.participate("old", 1.0)
    learner.participate("new", 1.0, incompatible=("old",))

    np.testing.assert_allclose(learner.trace("old"), 0.25)
    np.testing.assert_allclose(learner.trace("new"), 1.0)


def test_delayed_sweep_is_smooth_and_no_match_abstains() -> None:
    result = phase133.run_seed(3)
    permission = [row["permission"] for row in result["delays"]]

    assert all(
        left > right
        for left, right in zip(permission, permission[1:], strict=False)
    )
    assert permission[-1] > 0.0
    assert result["no_match"]["active_relations"] == 0
    assert result["no_match"]["eligibility_relations"] == 0
