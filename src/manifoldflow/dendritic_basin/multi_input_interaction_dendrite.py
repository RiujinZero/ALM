"""K-input dendritic branch with Gaussian bell response and J adaptation.

Generalises the two-input ``InteractionDendrite`` from
``interaction_dendrite.py`` to an arbitrary number of selected input
dimensions.

  d = exp[ -(w · x[indices] - μ)² / (2σ²) ]

  J_K = Var(d) - Σ_i  Cov(d, xᵢ)² / Var(xᵢ)

J_K is maximised when the response is informative about the joint pattern
of the selected inputs but not redundant with any single one of them —
the K-input analogue of the AND-NOT coincidence detector in the two-input
design.

Adaptation is local perturbation / selection on (weights, μ, σ): propose
a Gaussian perturbation, evaluate J on a rolling buffer of recent
observations, keep the candidate only if it improves J.  Input dimension
indices are fixed at construction.  The caller is responsible for
initialising them from birth-context statistics (see
``from_birth_context``).
"""

from __future__ import annotations

from collections import deque

import numpy as np

_EPS = 1e-8


class MultiInputInteractionDendrite:
    """A K-input dendritic branch that adapts via local J perturbation/selection."""

    def __init__(
        self,
        input_indices: np.ndarray,
        weights: np.ndarray,
        mu: float,
        sigma: float,
        buffer_size: int = 64,
        seed: int | None = None,
        output_weights: np.ndarray | None = None,
    ) -> None:
        self.input_indices = np.asarray(input_indices, dtype=np.intp).copy()
        self.weights = np.asarray(weights, dtype=np.float64).copy()
        self.mu = float(mu)
        self.sigma = float(max(sigma, 1e-3))
        self._buffer_size = buffer_size
        self._buf_x: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._buf_d: deque[float] = deque(maxlen=buffer_size)
        self._rng = np.random.default_rng(seed)
        self.output_weights: np.ndarray | None = (
            np.asarray(output_weights, dtype=np.float64).copy()
            if output_weights is not None
            else None
        )

    @property
    def k(self) -> int:
        return len(self.input_indices)

    def _bell(self, x_slice: np.ndarray, w: np.ndarray, mu: float, sigma: float) -> float:
        drive = float(np.dot(w, x_slice))
        return float(np.exp(-((drive - mu) ** 2) / (2.0 * sigma ** 2)))

    def respond(self, context: np.ndarray) -> float:
        return self._bell(context[self.input_indices], self.weights, self.mu, self.sigma)

    def observe(self, context: np.ndarray) -> float:
        x = context[self.input_indices].copy()
        d = self._bell(x, self.weights, self.mu, self.sigma)
        self._buf_x.append(x)
        self._buf_d.append(d)
        return d

    def _j_on(self, xs: np.ndarray, ds: np.ndarray) -> float:
        if len(ds) < 4:
            return 0.0
        var_d = float(np.var(ds))
        j = var_d
        d_mean = ds.mean()
        for i in range(self.k):
            xi = xs[:, i]
            cov_di = float(np.mean((xi - xi.mean()) * (ds - d_mean)))
            var_xi = float(np.var(xi))
            j -= cov_di ** 2 / (var_xi + _EPS)
        return j

    def _eval_bell_batch(
        self, xs: np.ndarray, w: np.ndarray, mu: float, sigma: float
    ) -> np.ndarray:
        drives = xs @ w
        return np.exp(-((drives - mu) ** 2) / (2.0 * sigma ** 2))

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Return d(context) * output_weights — the mode's residual prediction."""
        if self.output_weights is None:
            raise AttributeError("predict() requires output_weights")
        return self.respond(context) * self.output_weights

    def adapt_output(self, context: np.ndarray, error: np.ndarray, lr: float) -> None:
        """Delta-rule update: output_weights += lr * d(context) * error."""
        if self.output_weights is None:
            raise AttributeError("adapt_output() requires output_weights")
        self.output_weights += lr * self.respond(context) * error

    def adapt(self, step_size: float) -> float:
        """One perturbation/selection step. Returns resulting J."""
        if len(self._buf_x) < 8:
            return 0.0
        xs = np.stack(list(self._buf_x))
        ds_cur = self._eval_bell_batch(xs, self.weights, self.mu, self.sigma)
        j_cur = self._j_on(xs, ds_cur)

        rng = self._rng
        new_w = self.weights + rng.normal(0.0, step_size, self.k)
        new_mu = self.mu + float(rng.normal(0.0, step_size))
        new_sigma = max(1e-3, self.sigma + float(rng.normal(0.0, step_size * 0.5)))

        ds_cand = self._eval_bell_batch(xs, new_w, new_mu, new_sigma)
        j_cand = self._j_on(xs, ds_cand)

        if j_cand > j_cur:
            self.weights = new_w
            self.mu = new_mu
            self.sigma = new_sigma
            return j_cand
        return j_cur

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_birth_context(
        cls,
        context_mean: np.ndarray,
        k: int,
        initial_sigma: float = 1.0,
        buffer_size: int = 64,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        output_weights: np.ndarray | None = None,
    ) -> "MultiInputInteractionDendrite":
        """Construct a new branch from the nursery candidate's context mean.

        Selects the K dimensions with the highest absolute value in the birth
        context (most "active" at birth), with a small random shuffle among
        the top-2K to introduce diversity across co-born assemblies.  Weights
        are initialised so the branch fires near its peak for the birth context
        pattern.
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        d = len(context_mean)
        k = min(k, d)

        scores = np.abs(context_mean)
        top2k = int(min(2 * k, d))
        candidate_dims = np.argsort(scores)[-top2k:]
        chosen = rng.choice(candidate_dims, size=k, replace=False)
        input_indices = np.sort(chosen)

        x_birth = context_mean[input_indices]
        norm = float(np.linalg.norm(x_birth)) + _EPS
        weights = x_birth / norm
        mu = float(np.dot(weights, x_birth))

        return cls(
            input_indices=input_indices,
            weights=weights,
            mu=mu,
            sigma=initial_sigma,
            buffer_size=buffer_size,
            seed=seed,
            output_weights=output_weights,
        )
