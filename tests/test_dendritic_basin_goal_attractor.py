import pickle

import numpy as np

from manifoldflow.dendritic_basin.goal_attractor import (
    GoalConditionedLocalController,
    HomeostaticGoalAttractor,
    SparseHomeostaticGoalAttractor,
    SparsePredictiveBasinAttractor,
)


def _experience(
    learner: HomeostaticGoalAttractor, events: list[int], end: float
) -> None:
    learner.reset_episode()
    learner.observe_deviation(1.0)
    for event in events:
        learner.observe_event(event)
    learner.observe_deviation(end)


def test_restoration_selects_goal_without_target_label() -> None:
    learner = HomeostaticGoalAttractor(8, eligibility_decay=0.95)
    for _ in range(30):
        _experience(learner, [0, 2, 4], 0.0)
        _experience(learner, [1, 5, 5], 1.2)

    selected = learner.select_goal([0, 1])

    assert selected.goal == 0
    assert learner.value(0) > 0.0
    assert learner.value(1) < 0.0
    assert learner.current_goal == 0


def test_conditioned_restoration_propagates_to_earlier_predictor() -> None:
    learner = HomeostaticGoalAttractor(8, eligibility_decay=0.8)
    for _ in range(20):
        _experience(learner, [4], 0.0)
    before = learner.value(2)
    for _ in range(40):
        learner.reset_episode()
        learner.observe_deviation(1.0)
        learner.observe_event(2)
        learner.observe_event(4)

    assert learner.value(2) > before
    assert learner.value(2) > learner.minimum_goal_value


def test_goal_persists_through_neutral_steps_and_dissolves_on_restoration() -> None:
    learner = HomeostaticGoalAttractor(4)
    learner.restoration_sum[0] = 8.0
    learner.restoration_evidence[0] = 8.0
    learner.observe_deviation(1.0, learn=False)
    learner.select_goal([0, 1])

    for _ in range(8):
        learner.observe_deviation(1.0, learn=False)
        assert learner.current_goal == 0
    step = learner.observe_deviation(0.0, learn=False)

    assert step.changed
    assert learner.current_goal is None


def test_sparse_goal_ends_on_restoration_not_target_event() -> None:
    learner = SparseHomeostaticGoalAttractor(restoration_fraction=0.1)
    learner.restoration_sum[3] = 8.0
    learner.restoration_evidence[3] = 8.0
    learner.observe_deviation(1.0, learn=False)
    assert learner.select_goal({3: 1.0}) == 3

    learner.observe_event(3, learn=False)
    assert learner.current_goal == 3
    learner.observe_deviation(0.95, learn=False)
    assert learner.current_goal == 3
    learner.observe_deviation(0.89, learn=False)

    assert learner.current_goal is None


def test_sparse_goal_learns_component_restoration_signature() -> None:
    learner = SparseHomeostaticGoalAttractor(component_count=3)
    for _ in range(20):
        learner.reset_episode()
        learner.observe_components((1.0, 0.5, 0.5))
        learner.observe_event(3)
        learner.observe_components((0.0, 0.5, 0.5))

    signature = learner.restoration_signature(3)

    assert signature[0] > 0.5
    assert signature[1] == 0.0
    assert signature[2] == 0.0


def test_regulatory_deficit_selects_persistent_mode_until_restored() -> None:
    learner = SparseHomeostaticGoalAttractor(
        component_count=3, restoration_fraction=0.1
    )

    learner.observe_components((0.8, 0.2, 0.1), learn=False)
    assert learner.current_mode == 0
    learner.observe_components((0.75, 0.9, 0.1), learn=False)
    assert learner.current_mode == 0
    learner.observe_components((0.7, 0.9, 0.1), learn=False)
    assert learner.current_mode is None
    learner.observe_components((0.7, 0.9, 0.1), learn=False)

    assert learner.current_mode == 1


def test_explicit_goal_can_ignore_incidental_homeostatic_restoration() -> None:
    learner = SparseHomeostaticGoalAttractor(restoration_fraction=0.1)
    learner.current_goal = 7
    learner.goal_reference_deviation = 1.0
    learner.previous_deviation = 1.0

    learner.observe_deviation(0.5, learn=False, dissolve_on_restoration=False)

    assert learner.current_goal == 7
    assert learner.achieve(7)
    assert learner.current_goal is None


def test_goal_swap_changes_local_specialist_actions() -> None:
    controller = GoalConditionedLocalController(3, 2)
    for module in range(3):
        for goal in range(2):
            desired = (module + goal) % 2
            for action in range(2):
                for _ in range(20):
                    controller.observe(module, goal, action, action == desired)

    first = [controller.action(module, 0) for module in range(3)]
    swapped = [controller.action(module, 1) for module in range(3)]

    assert first == [0, 1, 0]
    assert swapped == [1, 0, 1]


def test_body_and_transient_state_are_counted_and_pickle_safe() -> None:
    learner = HomeostaticGoalAttractor(8)
    learner.observe_event(2)
    restored = pickle.loads(pickle.dumps(learner, protocol=5))

    np.testing.assert_array_equal(restored.eligibility, learner.eligibility)
    assert restored.embodied_bytes > 0
    assert restored.transient_bytes > 0


def test_predictive_signatures_self_organize_into_reusable_basins() -> None:
    learner = SparsePredictiveBasinAttractor(
        component_count=3, max_basins=4, birth_similarity=0.8
    )
    first = np.asarray((1.0, 0.0, 0.0))
    same_family = np.asarray((0.95, 0.05, 0.0))
    opponent = np.asarray((-1.0, 0.0, 0.0))

    first_basin = learner.observe(first)
    assert learner.observe(same_family) == first_basin
    opponent_basin = learner.observe(opponent)

    assert opponent_basin != first_basin
    assert learner.live_basins == 2
    assert learner.alignment(opponent) > 0.99


def test_predictive_basin_hysteresis_prevents_noisy_winner_flicker() -> None:
    learner = SparsePredictiveBasinAttractor(
        component_count=2,
        max_basins=2,
        birth_similarity=0.95,
        hysteresis=0.2,
    )
    left = learner.observe(np.asarray((1.0, 0.0)))
    right = learner.observe(np.asarray((0.0, 1.0)))
    assert left != right
    learner.current_basin = left

    selected = learner.observe(np.asarray((0.7, 0.72)), learn=False)

    assert selected == left


def test_predictive_basins_merge_by_structural_similarity() -> None:
    left = SparsePredictiveBasinAttractor(component_count=3, max_basins=4)
    right = SparsePredictiveBasinAttractor(component_count=3, max_basins=4)
    left.observe(np.asarray((1.0, 0.0, 0.0)))
    right.observe(np.asarray((0.98, 0.02, 0.0)))
    right.observe(np.asarray((0.0, 1.0, 0.0)))

    left.consolidate(right)

    assert left.live_basins == 2
    assert int(left.evidence.sum()) == 3
