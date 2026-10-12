"""Dendritic feature population (design doc sections 5-9, 15-18, 21-23).

Each branch proposes a nonlinear conjunction over a sparse subset of the
literal-encoded input (see `model.encode_literals`). Activation depends
only on distance to a stored prototype over that branch's own masked
connections; eligibility tracks each connection's recent participation;
prototype and usefulness updates are driven by a scalar global modulator
gating that local eligibility. No branch ever receives a gradient
computed from another branch's error (section 3.3, 38.5).

Eligibility (section 15) and structural plasticity (sections 8-9, 23) are
implemented as methods here rather than in separate modules: both operate
directly on this population's own per-branch arrays (prototype, mask,
eligibility, usefulness), and splitting them out would just mean passing
all of that state across a module boundary for no encapsulation benefit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DendriticConfig:
    n_branches: int = 16
    input_dim: int = 4
    synapses_per_branch: int = 2
    beta: float = 6.0  # activation sharpness
    trace_decay: float = 0.7  # lambda_e
    usefulness_decay: float = 0.95  # lambda_u
    prototype_lr: float = 0.5  # eta_p / eta_d
    surprise_threshold: float = 0.3  # theta_s
    match_threshold: float = 0.5  # theta_m
    structural_plasticity: bool = True
    seed: int = 0


class DendriticPopulation:
    """A pool of K sparse, locally-updated nonlinear feature detectors."""

    def __init__(self, config: DendriticConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        k, d = config.n_branches, config.input_dim
        self.prototype = rng.uniform(0.0, 1.0, size=(k, d))
        self.mask = np.zeros((k, d), dtype=np.float64)
        self.eligibility = np.zeros((k, d), dtype=np.float64)
        self.usefulness = np.zeros(k, dtype=np.float64)
        self.active = np.zeros(k, dtype=bool)
        self._last_activity = np.zeros(k, dtype=np.float64)
        self._rng = rng

        if not config.structural_plasticity:
            # Ablation B ("dendrites without structural plasticity",
            # section 25): pre-wire all K branches to random sparse
            # subsets up front; grow_branch is never called afterward.
            for branch in range(k):
                self._randomly_wire_branch(branch)

    def _randomly_wire_branch(self, branch: int) -> None:
        cfg = self.config
        count = min(cfg.synapses_per_branch, cfg.input_dim)
        idx = self._rng.choice(cfg.input_dim, size=count, replace=False)
        self.mask[branch] = 0
        self.mask[branch, idx] = 1
        self.active[branch] = True

    def activate(self, z: np.ndarray) -> np.ndarray:
        """d_k = exp(-beta * delta_k) (section 7).

        A branch with an all-zero mask (never grown) always reads exactly
        zero activation rather than the formula's vacuous zero-distance
        "perfect match" -- an ungrown branch must never look like it
        already represents the input.
        """
        cfg = self.config
        mask_sum = self.mask.sum(axis=1)
        difference = np.abs(self.prototype - z[None, :])
        masked = difference * self.mask
        distance = masked.sum(axis=1) / (mask_sum + 1e-8)
        activity = np.exp(-cfg.beta * distance)
        activity = np.where(mask_sum > 0, activity, 0.0)
        self._last_activity = activity
        return activity

    def update_eligibility(self, z: np.ndarray, activity: np.ndarray) -> None:
        """elig_ki(t+1) = lambda_e*elig_ki(t) + z_i(t)*d_k(t) (section 15)."""
        cfg = self.config
        self.eligibility *= cfg.trace_decay
        self.eligibility += activity[:, None] * z[None, :] * self.mask

    def learn(self, modulator: float) -> None:
        """Delta p_ki = eta_d * M(t) * e_ki(t) (section 15/22, restricted
        to masked connections), clipped to the valid literal range [0, 1]
        (section 22's noted follow-up: "replaced by a bounded prototype
        update to prevent drift outside the valid feature range").
        """
        cfg = self.config
        self.prototype += cfg.prototype_lr * modulator * self.eligibility * self.mask
        np.clip(self.prototype, 0.0, 1.0, out=self.prototype)
        self.usefulness = (
            cfg.usefulness_decay * self.usefulness + modulator * self._last_activity
        )

    def adapt_structure(
        self, z: np.ndarray, activity: np.ndarray, modulator: float
    ) -> int | None:
        """Grow a new branch on (surprise AND poor representation) (section 8/23)."""
        cfg = self.config
        if not cfg.structural_plasticity:
            return None
        if activity.size:
            poorly_represented = float(activity.max()) < cfg.match_threshold
        else:
            poorly_represented = True
        surprising = abs(modulator) > cfg.surprise_threshold
        if surprising and poorly_represented:
            return self.grow_branch(z)
        return None

    def grow_branch(self, z: np.ndarray) -> int | None:
        """New branch connects to the currently-active literals (section
        9/23). For the literal encoding (section 4), the currently-active
        literals are exactly the sparse conjunction the input expresses,
        so this is a natural (not manually XOR-specific) growth rule.
        Returns the grown branch's index, or None if no branch was free.
        """
        cfg = self.config
        unused = np.flatnonzero(~self.active)
        if unused.size == 0:
            return None
        branch = int(unused[0])
        active_literals = np.flatnonzero(z > 0.5)
        if active_literals.size == 0:
            active_literals = np.arange(cfg.input_dim)
        count = min(cfg.synapses_per_branch, active_literals.size)
        chosen = self._rng.choice(active_literals, size=count, replace=False)
        self.mask[branch] = 0
        self.mask[branch, chosen] = 1
        self.prototype[branch] = z
        self.eligibility[branch] = 0
        self.usefulness[branch] = 0
        self.active[branch] = True
        return branch

    def n_active_branches(self) -> int:
        return int(self.active.sum())
