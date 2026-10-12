from __future__ import annotations

from manifoldflow.dendritic_basin.temporal_difference_credit import (
    LocalValueEstimate,
    td_error,
)


def test_td_error_matches_hand_computed_delta() -> None:
    delta = td_error(outcome=1.0, value_next=2.0, value_current=0.5, gamma=0.9)
    assert abs(delta - (1.0 + 0.9 * 2.0 - 0.5)) < 1e-12


def test_gamma_zero_collapses_to_one_step_utility_only() -> None:
    delta_with_bootstrap = td_error(
        outcome=0.3, value_next=100.0, value_current=0.1, gamma=0.0
    )
    delta_without_next = td_error(
        outcome=0.3, value_next=-99.0, value_current=0.1, gamma=0.0
    )
    assert delta_with_bootstrap == delta_without_next
    assert abs(delta_with_bootstrap - (0.3 - 0.1)) < 1e-12


def test_local_value_estimate_converges_on_a_deterministic_chain() -> None:
    # x0 -> x1 -> terminal, with a fixed reward of 1.0 only on the final step.
    estimator = LocalValueEstimate(learning_rate=0.5)
    for _ in range(200):
        estimator.update("x0", outcome=0.0, next_key="x1", gamma=1.0)
        estimator.update("x1", outcome=1.0, next_key=None, gamma=1.0)

    assert abs(estimator.value("x1") - 1.0) < 1e-6
    assert abs(estimator.value("x0") - 1.0) < 1e-3


def test_update_returns_the_delta_actually_applied() -> None:
    estimator = LocalValueEstimate(learning_rate=0.1, initial_value=0.0)
    delta = estimator.update("k", outcome=2.0, next_key=None, gamma=0.5)
    assert abs(delta - 2.0) < 1e-12
    assert abs(estimator.value("k") - 0.2) < 1e-12


def test_rejects_out_of_range_gamma() -> None:
    estimator = LocalValueEstimate()
    try:
        estimator.update("k", outcome=1.0, next_key=None, gamma=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for gamma outside [0, 1]")
