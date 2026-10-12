from __future__ import annotations

import numpy as np

from manifoldflow.experiments.dendritic_basin_goal_horizon_ablation import (
    boundary_byte_mask,
    observe_with_episode_resets,
    observe_with_regulatory_trace,
    story_positions,
    summarize_complete_stories,
    summarize_horizons,
    summarize_state_persistence,
)


def test_story_positions_reset_after_complete_boundary() -> None:
    boundary = b"<>"
    stream = np.frombuffer(b"ab<>xyz", dtype=np.uint8)

    np.testing.assert_array_equal(
        story_positions(stream, boundary),
        np.asarray((0, 1, 2, 3, 0, 1, 2)),
    )


def test_boundary_mask_marks_only_complete_delimiters() -> None:
    stream = np.frombuffer(b"<>x<y<>z<", dtype=np.uint8)

    np.testing.assert_array_equal(
        boundary_byte_mask(stream, b"<>"),
        np.asarray((1, 1, 0, 0, 0, 1, 1, 0, 0), dtype=np.bool_),
    )


def test_oracle_episode_reset_prevents_transient_carryover() -> None:
    class CountingLearner:
        def __init__(self) -> None:
            self.count = 0

        def reset_episode(self) -> None:
            self.count = 0

        def observe_chunk_probabilities(
            self, stream: np.ndarray, *, learn: bool
        ) -> np.ndarray:
            values = np.arange(self.count, self.count + len(stream), dtype=np.float64)
            self.count += len(stream)
            return values

    stream = np.frombuffer(b"ab<>xyz", dtype=np.uint8)

    np.testing.assert_array_equal(
        observe_with_episode_resets(CountingLearner(), stream, b"<>"),
        np.asarray((0, 1, 2, 3, 0, 1, 2), dtype=np.float64),
    )


def test_regulatory_trace_records_mode_and_flow_onset() -> None:
    class GoalOrgan:
        current_mode = 2
        current_goal = None

    class TraceLearner:
        def __init__(self) -> None:
            self.goal_organ = GoalOrgan()
            self.goal_route_flow_uses = 0
            self.maximum_goal_route_byte_weight = 0.02
            self.observed_weights: list[float] = []

        def reset_episode(self) -> None:
            self.goal_organ.current_mode = 2

        def _observe_probability_only(self, byte_id: int, *, learn: bool) -> float:
            self.observed_weights.append(self.maximum_goal_route_byte_weight)
            if byte_id == 7:
                self.goal_route_flow_uses += 1
            self.goal_organ.current_mode = 0
            return 0.5

    probability, modes, goals, basins, flow = observe_with_regulatory_trace(
        TraceLearner(), np.asarray((7, 8), dtype=np.uint8)
    )

    np.testing.assert_array_equal(probability, np.asarray((0.5, 0.5)))
    np.testing.assert_array_equal(modes, np.asarray((2, 0), dtype=np.int8))
    np.testing.assert_array_equal(goals, np.asarray((-1, -1), dtype=np.int64))
    np.testing.assert_array_equal(basins, np.asarray((-1, -1), dtype=np.int16))
    np.testing.assert_array_equal(flow, np.asarray((True, False)))

    gated = TraceLearner()
    observe_with_regulatory_trace(gated, np.asarray((7, 8)), allowed_modes=(0,))
    assert gated.observed_weights == [0.0, 0.02]
    assert gated.maximum_goal_route_byte_weight == 0.02


def test_state_persistence_counts_only_active_runs() -> None:
    summary = summarize_state_persistence(
        np.asarray((-1, 2, 2, 2, -1, 2, 2, 1, 1, 1, 1), dtype=np.int64)
    )

    assert summary["active_fraction"] == 9 / 11
    assert summary["distinct_active_states"] == 2
    assert summary["active_runs"] == 3
    assert summary["mean_run_bytes"] == 3.0
    assert summary["median_run_bytes"] == 3.0
    assert summary["maximum_run_bytes"] == 4


def test_horizon_summary_reports_causal_saved_bits() -> None:
    delta = np.asarray((-1.0, 0.5, -0.25, 0.0))
    positions = np.asarray((0, 65, 300, 2_000))

    report = summarize_horizons(delta, positions)

    assert report["0:64"]["total_bits_saved_by_goal"] == 1.0
    assert report["64:128"]["total_bits_saved_by_goal"] == -0.5
    assert report["256:512"]["total_bits_saved_by_goal"] == 0.25
    assert report["1024+"]["changed_fraction"] == 0.0


def test_complete_story_summary_reports_breadth_not_only_total() -> None:
    stream = np.frombuffer(b"ab<>xyz<>", dtype=np.uint8)
    # Goal saves one bit in the first story and loses two in the second.
    delta = np.asarray((-0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0))

    report = summarize_complete_stories(delta, stream, b"<>")

    assert report["complete_story_count"] == 2
    assert report["goal_win_story_fraction"] == 0.5
    assert report["goal_loss_story_fraction"] == 0.5
    assert report["mean_bits_saved_per_story"] == -0.5
    assert report["median_bits_saved_per_story"] == -0.5
