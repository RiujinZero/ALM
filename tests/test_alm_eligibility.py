"""Known-delay sequence fixtures for forward eligibility traces.

Design doc section 17 (software architecture): learning/eligibility.py must
pass "known-delay sequence fixtures."
"""

from __future__ import annotations

import torch

from manifoldflow.alm.learning.eligibility import EligibilityConfig, TemporalPredictor


def test_trace_matches_geometric_decay_formula() -> None:
    """elig_ij,t = lambda_e*elig_ij,t-1 + pre_i,t*post'_j,t, checked directly
    against the closed-form geometric-weighted sum over a known sequence.
    """
    torch.manual_seed(0)
    d = 4
    decay = 0.7
    predictor = TemporalPredictor(d, EligibilityConfig(decay=decay))

    pres = [torch.randn(d) for _ in range(6)]
    posts = [torch.rand(d) for _ in range(6)]

    expected = torch.zeros(d, d)
    for pre, post in zip(pres, posts, strict=False):
        predictor.advance_trace(pre=pre, post_prime=post)
        expected = decay * expected + torch.outer(post, pre)
        torch.testing.assert_close(predictor.trace, expected)


def test_close_prediction_whitens_and_clips() -> None:
    config = EligibilityConfig(modulation_clip=1.0, whiten_ema=1.0)
    predictor = TemporalPredictor(3, config)
    predicted = torch.zeros(3)
    observed = torch.tensor([10.0, -10.0, 0.0])
    modulation = predictor.close_prediction(predicted, observed)
    assert float(modulation.abs().max()) <= 1.0 + 1e-6


def test_update_applies_delta_a_formula() -> None:
    predictor = TemporalPredictor(3, EligibilityConfig(eligibility_lr=0.1))
    predictor.trace = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [3.0, 0.0, 1.0]])
    A = torch.eye(3)
    modulation = torch.tensor([1.0, -1.0, 0.5])

    updated = predictor.update(A, modulation)

    expected_delta = 0.1 * modulation.unsqueeze(-1) * predictor.trace
    torch.testing.assert_close(updated, A + expected_delta)


def test_known_linear_dynamics_reduce_prediction_error_over_training() -> None:
    """Generate a sequence from a known one-step-delayed linear system
    h_{t+1} = A_true h_t, then verify repeated forward (never backward)
    eligibility-gated updates bring the predictor's closed-prediction error
    down as it re-visits the same known-delay trajectory.
    """
    torch.manual_seed(1)
    d = 5
    A_true = 0.6 * torch.eye(d) + 0.05 * torch.randn(d, d)

    h0 = torch.randn(d)
    trajectory = [h0]
    for _ in range(40):
        trajectory.append(A_true @ trajectory[-1])

    A = torch.randn(d, d) * 0.05
    predictor = TemporalPredictor(d, EligibilityConfig(decay=0.5, eligibility_lr=0.2))

    def epoch_mean_abs_residual() -> float:
        total = 0.0
        for t in range(len(trajectory) - 1):
            predicted = predictor.predict(A, trajectory[t])
            total += float((trajectory[t + 1] - predicted).abs().mean())
        return total / (len(trajectory) - 1)

    first_epoch_error = epoch_mean_abs_residual()

    for _epoch in range(30):
        for t in range(len(trajectory) - 1):
            support = torch.ones(d)  # identity post-derivative for this fixture
            A, _modulation = predictor.step(
                A, trajectory[t], trajectory[t + 1], support
            )

    last_epoch_error = epoch_mean_abs_residual()

    assert last_epoch_error < 0.25 * first_epoch_error


def test_predict_uses_only_current_A_and_h_no_future_leakage() -> None:
    """predict() must be a pure function of (A, h) -- no dependence on
    anything from a later time step (the forward-only credit invariant).
    """
    predictor = TemporalPredictor(3)
    A = torch.randn(3, 3)
    h = torch.randn(3)
    first = predictor.predict(A, h)
    predictor.advance_trace(pre=torch.randn(3), post_prime=torch.rand(3))
    second = predictor.predict(A, h)
    torch.testing.assert_close(first, second)
