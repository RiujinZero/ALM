from types import SimpleNamespace

import numpy as np

from manifoldflow.dendritic_basin.recalled_expression import (
    RecalledCorticalExpression,
)
from manifoldflow.experiments.dendritic_basin_phase138_trajectory_shadow import (
    _scramble_active,
)
from manifoldflow.experiments.dendritic_basin_phase139_evidence_release import (
    _shadow_with_release,
)
from manifoldflow.experiments.dendritic_basin_phase140_cortical_route_return import (
    _route_overlay_distribution,
)
from manifoldflow.experiments.dendritic_basin_phase143_learned_route_permission import (
    _learned_support,
    _routable_support,
)


class _FrozenLearner:
    def predict_next_distribution(self) -> np.ndarray:
        distribution = np.full(256, 1e-6, dtype=np.float64)
        distribution[ord("x")] = 1.0
        distribution /= distribution.sum()
        return distribution

    def _observe_probability_only(self, byte_id: int, *, learn: bool) -> None:
        assert byte_id == ord("x")
        assert not learn


class _RouteLearner:
    temporal_bins = (1, 2, 4)
    temporal_clock = 10

    def __init__(self) -> None:
        self.temporal_last_seen = {3: 7}
        self.route_counts = {(1, 9): {}}
        self.route_utility = {(1, 9): 0.25}
        self.active_word_candidates = {3: 2}

    def predict_next_distribution(self) -> np.ndarray:
        distribution = np.zeros(256, dtype=np.float64)
        for (_receptor, assembly), utility in self.route_utility.items():
            distribution[assembly] = utility
        distribution[0] = 1.0
        distribution /= distribution.sum()
        return distribution


def test_identity_scramble_preserves_receptor_phase_and_strength() -> None:
    expression = RecalledCorticalExpression(minimum_evidence=1)
    expression.learn_sequence((1, 7), b"a")
    expression.learn_sequence((2, 7), b"b")
    expression.activate(((1, 7),), strengths=(0.4,))

    assert _scramble_active(expression)
    assert expression.active == {(2, 7): (0.4, 256)}


def test_external_mismatch_releases_without_teaching_the_return() -> None:
    expression = RecalledCorticalExpression(minimum_evidence=1)
    expression.learn_sequence((1, 7), b"yyyy")
    before = expression.embodied_bytes
    expression.activate(((1, 7),))
    arm = SimpleNamespace(
        learner=_FrozenLearner(),
        recalled_expression=expression,
        expression_gain=0.5,
        expression_cancellation_mode="none",
        expression_current_limit=0.0,
    )

    result = _shadow_with_release(arm, b"x" * 128, mismatch_streak=1)

    assert result["released_at"] == 1
    assert result["savings"]["1"] < 0.0
    assert result["savings"]["1"] == result["savings"]["128"]
    assert expression.embodied_bytes == before


def test_cortical_route_overlay_is_transient() -> None:
    learner = _RouteLearner()
    before_seen = learner.temporal_last_seen.copy()
    before_utility = learner.route_utility.copy()
    before_candidates = learner.active_word_candidates

    distribution = _route_overlay_distribution(
        learner,
        ((9, 0),),
        support=0.5,
    )

    assert distribution is not None
    assert distribution[9] > 0.25
    assert learner.temporal_last_seen == before_seen
    assert learner.route_utility == before_utility
    assert learner.active_word_candidates is before_candidates


def test_cortical_route_overlay_preserves_per_item_support() -> None:
    learner = _RouteLearner()
    learner.route_counts[(1, 10)] = {}
    learner.route_utility[(1, 10)] = 0.25

    distribution = _route_overlay_distribution(
        learner,
        ((9, 0), (10, 0)),
        support={(9, 0): 0.75, (10, 0): 0.25},
    )

    assert distribution is not None
    assert distribution[9] > distribution[10]


def test_learned_support_requires_permission_and_an_existing_cortical_route() -> None:
    class _Tissue:
        @staticmethod
        def permission(relation: tuple[int, int, int]) -> float:
            return {7: 0.5, 8: 0.0}[relation[0]]

    arm = SimpleNamespace(
        temporal_relational_eligibility=_Tissue(),
        pending_receptor_relational_population={
            (7, 9, 1): 0.4,
            (8, 10, 1): 0.9,
        },
    )
    support = _learned_support(arm, 2.0)

    assert support == {(9, 1): 0.4, (10, 1): 0.0}

    learner = _RouteLearner()
    assert _routable_support(learner, support) == {(9, 1): 0.4}
