"""The v0.2 generic nonmonotonic interaction dendrite (design doc
`Dendritic_Basin_Learning_Design_v0.2_Interaction_Dendrites.md`, section
10.1-10.4; consolidated in `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`
sections 6-9). Replaces the v0.1 prototype-distance dendrite
(`dendrites.py`), which was overproposing structure (see the
implementation journal).

  d(A, B) = exp[ -(w_A*A + w_B*B - mu)^2 / (2*sigma^2) ]
  J = Var(d) - Cov(d,A)^2/Var(A) - Cov(d,B)^2/Var(B)

No XOR/AND/DIFF primitive anywhere -- `d` is a generic bell response over
a weighted sum of its two parents, and `J` ("be informative, but don't
just copy either parent") is computed entirely from running local
statistics. Adaptation is local perturbation/selection (propose, evaluate
J, keep only if it improved) -- never a gradient of J.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-8


@dataclass
class RunningJointStats:
    """Streaming (Welford-style) mean/variance/covariance for (a, b, d).

    Supports one-at-a-time `update` (for a dendrite's lifetime statistics)
    and is cheap to build fresh from a batch (for local-search trials,
    where each candidate parameter set needs its own clean estimate).
    """

    n: int = 0
    mean_a: float = 0.0
    mean_b: float = 0.0
    mean_d: float = 0.0
    m2_a: float = 0.0
    m2_b: float = 0.0
    m2_d: float = 0.0
    cov_da_acc: float = 0.0
    cov_db_acc: float = 0.0

    def update(self, a: float, b: float, d: float) -> None:
        self.n += 1
        da_pre = a - self.mean_a
        db_pre = b - self.mean_b
        dd_pre = d - self.mean_d

        self.mean_a += da_pre / self.n
        self.mean_b += db_pre / self.n
        self.mean_d += dd_pre / self.n

        self.m2_a += da_pre * (a - self.mean_a)
        self.m2_b += db_pre * (b - self.mean_b)
        self.m2_d += dd_pre * (d - self.mean_d)

        self.cov_da_acc += da_pre * (d - self.mean_d)
        self.cov_db_acc += db_pre * (d - self.mean_d)

    def var_a(self) -> float:
        return self.m2_a / max(self.n, 1)

    def var_b(self) -> float:
        return self.m2_b / max(self.n, 1)

    def var_d(self) -> float:
        return self.m2_d / max(self.n, 1)

    def cov_d_a(self) -> float:
        return self.cov_da_acc / max(self.n, 1)

    def cov_d_b(self) -> float:
        return self.cov_db_acc / max(self.n, 1)

    def objective(self) -> float:
        """J = Var(d) - Cov(d,A)^2/Var(A) - Cov(d,B)^2/Var(B)."""
        if self.n == 0:
            return 0.0
        return (
            self.var_d()
            - (self.cov_d_a() ** 2) / (self.var_a() + EPS)
            - (self.cov_d_b() ** 2) / (self.var_b() + EPS)
        )


class InteractionDendrite:
    """A single generic nonmonotonic two-input dendrite with the
    interaction-maximizing local objective, adapted by local
    perturbation/selection. Four scalar parameters: `w_a, w_b, mu, sigma`.
    """

    def __init__(self, seed: int, init_scale: float = 1.0):
        rng = np.random.default_rng(seed)
        self.w_a = float(rng.normal(0.0, init_scale))
        self.w_b = float(rng.normal(0.0, init_scale))
        self.mu = float(rng.normal(0.0, init_scale))
        self.sigma = float(rng.uniform(0.5, 2.0))
        self.stats = RunningJointStats()

    def respond(self, a: float, b: float) -> float:
        drive = self.w_a * a + self.w_b * b
        return float(np.exp(-((drive - self.mu) ** 2) / (2 * self.sigma**2)))

    def observe(self, a: float, b: float) -> float:
        """Update lifetime running statistics with one more sample."""
        d = self.respond(a, b)
        self.stats.update(a, b, d)
        return d

    def local_objective(self) -> float:
        """J from this dendrite's own lifetime statistics (no label)."""
        return self.stats.objective()

    def evaluate_on(self, samples: list[tuple[float, float]]) -> float:
        """J computed fresh from `samples`, independent of lifetime stats
        -- used to score a candidate parameter set during local search.
        """
        stats = RunningJointStats()
        for a, b in samples:
            stats.update(a, b, self.respond(a, b))
        return stats.objective()

    def clone_params(self) -> InteractionDendrite:
        """A copy sharing parameters but with fresh (empty) statistics."""
        clone = InteractionDendrite.__new__(InteractionDendrite)
        clone.w_a, clone.w_b, clone.mu, clone.sigma = (
            self.w_a,
            self.w_b,
            self.mu,
            self.sigma,
        )
        clone.stats = RunningJointStats()
        return clone

    def perturbed(
        self, step_size: float, rng: np.random.Generator
    ) -> InteractionDendrite:
        candidate = self.clone_params()
        candidate.w_a += float(rng.normal(0.0, step_size))
        candidate.w_b += float(rng.normal(0.0, step_size))
        candidate.mu += float(rng.normal(0.0, step_size))
        candidate.sigma = max(1e-3, candidate.sigma + float(rng.normal(0.0, step_size)))
        return candidate

    def perturb_and_select(
        self,
        samples: list[tuple[float, float]],
        step_size: float,
        rng: np.random.Generator,
    ) -> float:
        """One local search step (section 3 of the v0.2 addendum): propose
        a perturbation, evaluate J for both current and candidate params
        on the same fresh `samples`, keep whichever scores higher. Returns
        the resulting (post-step) J.
        """
        candidate = self.perturbed(step_size, rng)
        current_j = self.evaluate_on(samples)
        candidate_j = candidate.evaluate_on(samples)
        if candidate_j > current_j:
            self.w_a, self.w_b, self.mu, self.sigma = (
                candidate.w_a,
                candidate.w_b,
                candidate.mu,
                candidate.sigma,
            )
            return candidate_j
        return current_j
