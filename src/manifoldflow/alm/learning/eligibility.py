"""Temporal credit without BPTT: eligibility traces and one-event delayed
prediction (design doc section 9).

A column predicts its next state and stores an eligibility trace for the
synapses that produced that prediction. When the next token arrives, the
observed state closes the prediction; the residual combines with the trace
to update the recurrent map in the forward direction of time only -- no
token history is unrolled or replayed backward.

  elig_ij,t = lambda_e * elig_ij,t-1 + pre_i,t * post'_j,t
  m_t+1 = clip(whiten(h_t+1 - h_hat_t+1), -m_max, m_max)
  Delta A_ij = eta_A * m_j,t+1 * elig_ij,t
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EligibilityConfig:
    decay: float = 0.85  # lambda_e; doc's per-level default for a mid column
    eligibility_lr: float = 0.01  # eta_A
    modulation_clip: float = 3.0  # m_max
    whiten_ema: float = 0.01


class TemporalPredictor:
    """Predicts a column's next activity from its current state and closes
    that prediction one step later via an eligibility-gated local update to
    `A` (the same self-recurrent map used in the column's pre-activation).
    """

    def __init__(self, state_dim: int, config: EligibilityConfig | None = None):
        self.config = config or EligibilityConfig()
        self.trace = torch.zeros(state_dim, state_dim)  # elig_ij
        self.residual_mean = torch.zeros(state_dim)
        self.residual_var = torch.ones(state_dim)

    def predict(self, A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """h_hat_{t+1} = A h_t, the column's own one-step self-prediction."""
        return A @ h

    def advance_trace(self, pre: torch.Tensor, post_prime: torch.Tensor) -> None:
        """elig_ij,t = lambda_e * elig_ij,t-1 + pre_i,t * post'_j,t.

        `pre` is the presynaptic activity that produced the prediction
        (h_t); `post_prime` is the postsynaptic derivative gating which
        predicted units are eligible for credit (e.g. the kWTA support mask
        or a smooth surrogate).
        """
        cfg = self.config
        self.trace = cfg.decay * self.trace + torch.outer(post_prime, pre)

    def close_prediction(
        self, predicted: torch.Tensor, observed: torch.Tensor
    ) -> torch.Tensor:
        """m_{t+1} = clip(whiten(h_{t+1} - h_hat_{t+1}), -m_max, m_max)."""
        cfg = self.config
        raw_residual = observed - predicted
        delta = raw_residual - self.residual_mean
        self.residual_mean = self.residual_mean + cfg.whiten_ema * delta
        var = self.residual_var
        self.residual_var = (1 - cfg.whiten_ema) * var + cfg.whiten_ema * delta**2
        std = torch.sqrt(torch.clamp_min(self.residual_var, 1e-6))
        whitened = raw_residual / std
        return torch.clamp(whitened, -cfg.modulation_clip, cfg.modulation_clip)

    def update(self, A: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        """Delta A_ij = eta_A * m_j,t+1 * elig_ij,t; returns the updated A.

        `trace` and `A` share the (post, pre) row/column convention, so
        `modulation` (indexed by post-unit j) broadcasts down each row.
        """
        delta_a = self.config.eligibility_lr * modulation.unsqueeze(-1) * self.trace
        return A + delta_a

    def step(
        self,
        A: torch.Tensor,
        h_prev: torch.Tensor,
        h_now: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward temporal-credit step: close the prediction made last
        step, update A from it, then register this step's own prediction
        (and its eligibility trace) for the next call.

        Returns (updated_A, modulation) where `modulation` is this step's
        closed-prediction error signal (useful for logging/tests).
        """
        predicted = self.predict(A, h_prev)
        modulation = self.close_prediction(predicted, h_now)
        updated_A = self.update(A, modulation)
        self.advance_trace(pre=h_prev, post_prime=support)
        return updated_A, modulation
