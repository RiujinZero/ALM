"""One plastic predictive column (design doc section 6, 6.1, 6.2, 8).

A column holds tied recognition/reconstruction weights, a self-recurrent
map, and a local next-token head, all randomly initialized and updated only
from local signals (presynaptic activity, postsynaptic state, and a local
error) -- never from a global backward pass. This module intentionally has
no `torch.nn.Parameter`/autograd machinery: every update below is the
literal local delta rule from the design doc, applied under `torch.no_grad`
semantics (plain tensor ops), so "local" is structurally true rather than
merely intended.

Equations implemented here:
  a = W @ h_in + A @ h_prev + b                      (6, eq 1, G_l m_t term
  h = kWTA(norm(tanh(a)))                             deferred to the MF
  reconstruction = W.T @ h                            integration milestone)
  q = H @ h; logits = E_token @ q + b_tok; p = softmax(logits)   (8)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PlasticColumnConfig:
    input_dim: int
    state_dim: int = 768
    token_dim: int = 384
    vocab_size: int = 4096
    target_density: float = 0.125
    weight_lr: float = 0.002
    token_head_lr: float = 0.004
    token_embed_lr: float = 0.004
    homeostasis_ema: float = 0.01
    density_ema: float = 0.05
    threshold_lr: float = 0.05
    variance_floor: float = 1e-3
    covariance_penalty: float = 1e-3
    weight_norm_cap: float = 3.0
    dead_unit_window: int = 200
    dead_unit_activity_threshold: float = 1e-6
    seed: int = 0
    # False only for linear-equivalence / finite-difference test fixtures
    # (learning/pc_alm.py, section 17): replaces kWTA(norm(tanh(.))) with the
    # identity so PC-ALM settling can be checked against closed-form linear
    # least squares. Real training always leaves this True.
    nonlinear: bool = True


@dataclass
class ColumnState:
    """Mutable per-column state: activity, homeostasis stats, dead-unit clock."""

    h: torch.Tensor  # (state_dim,) current settled activity
    running_mean: torch.Tensor  # (state_dim,) EMA of activity
    running_var: torch.Tensor  # (state_dim,) EMA of activity variance
    unit_density: torch.Tensor  # (state_dim,) EMA of per-unit firing rate
    threshold_bias: torch.Tensor  # (state_dim,) adaptive kWTA competitiveness
    steps_since_active: torch.Tensor  # (state_dim,) int, for dead-unit rescue
    covariance_ema: torch.Tensor  # (state_dim, state_dim) EMA outer product


class PlasticColumn:
    """A single 768-wide plastic column with tied weights and a local token head."""

    def __init__(
        self, config: PlasticColumnConfig, generator: torch.Generator | None = None
    ):
        self.config = config
        g = generator or torch.Generator().manual_seed(config.seed)
        d, din = config.state_dim, config.input_dim
        dtok, v = config.token_dim, config.vocab_size

        def init_matrix(rows: int, cols: int, scale: float) -> torch.Tensor:
            return torch.randn(rows, cols, generator=g) * scale

        # Recognition weight; reconstruction reuses W.T (tied, section 6).
        self.W = init_matrix(d, din, scale=(1.0 / din) ** 0.5)
        # Self-recurrent map (A_l h_{l,t-1} term in eq 1).
        self.A = init_matrix(d, d, scale=0.1 * (1.0 / d) ** 0.5)
        self.b = torch.zeros(d)
        # Local token head: state -> shared token space -> vocab logits.
        self.H = init_matrix(dtok, d, scale=(1.0 / d) ** 0.5)
        self.E_token = init_matrix(v, dtok, scale=(1.0 / dtok) ** 0.5)
        self.b_tok = torch.zeros(v)

        self.state = ColumnState(
            h=torch.zeros(d),
            running_mean=torch.zeros(d),
            running_var=torch.ones(d),
            unit_density=torch.full((d,), config.target_density),
            threshold_bias=torch.zeros(d),
            steps_since_active=torch.zeros(d, dtype=torch.long),
            covariance_ema=torch.zeros(d, d),
        )

    # -- forward / settling -------------------------------------------------

    def pre_activation(self, h_in: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """a = W h_in + A h_prev + b (eq 1, without the deferred MF read term)."""
        return self.W @ h_in + self.A @ h_prev + self.b

    def kwta(self, a: torch.Tensor) -> torch.Tensor:
        """kWTA(norm(tanh(a))): normalize, subtract adaptive per-unit threshold,
        keep only the top-k units (k set by `target_density`) among those with
        positive post-threshold drive.
        """
        cfg = self.config
        x = torch.tanh(a)
        norm = x.norm()
        if norm > 0:
            x = x / norm
        competitive = x - self.state.threshold_bias
        k = max(1, round(cfg.target_density * cfg.state_dim))
        topk_vals, topk_idx = torch.topk(competitive, k)
        h = torch.zeros_like(x)
        keep = topk_idx[topk_vals > 0]
        h[keep] = x[keep]
        return h

    def activation(self, a: torch.Tensor) -> torch.Tensor:
        """kWTA(norm(tanh(a))), or the identity in linear-fixture mode (see
        `PlasticColumnConfig.nonlinear`).
        """
        return self.kwta(a) if self.config.nonlinear else a

    def forward(self, h_in: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """One settling-free forward pass (initialization, section 13.1 step 1)."""
        a = self.pre_activation(h_in, h_prev)
        return self.activation(a)

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        return self.W.T @ h

    # -- local token head (section 8) ---------------------------------------

    def token_logits(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.H @ h
        logits = self.E_token @ q + self.b_tok
        return logits, q

    def token_probs(self, h: torch.Tensor) -> torch.Tensor:
        logits, _ = self.token_logits(h)
        return torch.softmax(logits, dim=-1)

    # -- local updates (no autograd; literal delta rules) --------------------

    def update_reconstruction(
        self, h_in: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """Delta W from local reconstruction residual (section 6.1 w_rec term).

        J_rec = ||h_in - W^T h||^2  =>  dJ/dW = -2 h (h_in - W^T h)^T
        Delta W = eta_w * h @ e_rec^T, a pre/post/residual-local Hebbian rule.
        """
        e_rec = h_in - self.reconstruct(h)
        delta_w = self.config.weight_lr * torch.outer(h, e_rec)
        self.W = self._clip_rows(self.W + delta_w)
        return e_rec

    def token_delta(self, h: torch.Tensor, next_token: int) -> torch.Tensor:
        """delta = onehot(y_next) - p, without applying any update (section 8)."""
        logits, _q = self.token_logits(h)
        p = torch.softmax(logits, dim=-1)
        target = torch.zeros_like(p)
        target[next_token] = 1.0
        return target - p

    def token_gradient_wrt_h(self, delta: torch.Tensor) -> torch.Tensor:
        """dCE/dh for this column's own head, given `delta` from `token_delta`.

        CE = -log p[y]; dCE/dlogits = p - onehot(y) = -delta;
        dCE/dq = E_token^T @ dCE/dlogits; dCE/dh = H^T @ dCE/dq.
        Used to fold a column's own local token loss into PC-ALM settling's
        activity pressure (`A = L_token + ...`, section 7) -- the mechanism
        by which downstream classification error can shape upstream
        columns' weights via the neighbor-exchange term in
        `learning/pc_alm.py`'s `_activity_gradient`, instead of only ever
        updating this column's own `H`/`E_token` in isolation.
        """
        return -(self.H.T @ (self.E_token.T @ delta))

    def update_token_head(self, h: torch.Tensor, next_token: int) -> torch.Tensor:
        """Local softmax delta updates at the head (section 8, eq 3)."""
        cfg = self.config
        _logits, q = self.token_logits(h)
        delta = self.token_delta(h, next_token)

        d_h = cfg.token_head_lr * torch.outer(self.E_token.T @ delta, h)
        d_e = cfg.token_embed_lr * torch.outer(delta, q)
        self.H = self._clip_rows(self.H + d_h)
        self.E_token = self._clip_rows(self.E_token + d_e)
        self.b_tok = self.b_tok + cfg.token_head_lr * delta
        return delta

    def update_homeostasis(self, h: torch.Tensor) -> None:
        """Track density/variance/covariance and nudge the kWTA threshold and
        variance floor toward the target activation density (section 6.2).
        """
        cfg, s = self.config, self.state
        fired = (h != 0).float()

        density_ema = cfg.density_ema
        s.unit_density = (1 - density_ema) * s.unit_density + density_ema * fired
        density_error = s.unit_density - cfg.target_density
        s.threshold_bias = s.threshold_bias + cfg.threshold_lr * density_error

        ema = cfg.homeostasis_ema
        delta_mean = h - s.running_mean
        s.running_mean = s.running_mean + ema * delta_mean
        s.running_var = (1 - ema) * s.running_var + ema * delta_mean**2
        s.running_var = torch.clamp_min(s.running_var, cfg.variance_floor)

        s.covariance_ema = (1 - ema) * s.covariance_ema + ema * torch.outer(h, h)

        zero = torch.zeros_like(s.steps_since_active)
        s.steps_since_active = torch.where(
            fired.bool(), zero, s.steps_since_active + 1
        )
        self._rescue_dead_units()

    def off_diagonal_covariance_energy(self) -> torch.Tensor:
        cov = self.state.covariance_ema
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag**2).mean()

    def effective_rank(self) -> float:
        """Participation-ratio effective rank of the recent covariance sketch."""
        eigvals = torch.linalg.eigvalsh(self.state.covariance_ema)
        eigvals = torch.clamp_min(eigvals, 0)
        total = eigvals.sum()
        if total <= 0:
            return 0.0
        p = eigvals / total
        entropy = -(p * torch.log(torch.clamp_min(p, 1e-12))).sum()
        return float(torch.exp(entropy))

    def activation_density(self) -> float:
        return float(self.state.unit_density.mean())

    def max_unit_share(self) -> float:
        return float(self.state.unit_density.max())

    # -- checkpointing --------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        """All learned weights and homeostasis state, as a flat, cloned dict
        (safe to `torch.save` directly for a resumable checkpoint).
        """
        return {
            "W": self.W.clone(),
            "A": self.A.clone(),
            "b": self.b.clone(),
            "H": self.H.clone(),
            "E_token": self.E_token.clone(),
            "b_tok": self.b_tok.clone(),
            "state.h": self.state.h.clone(),
            "state.running_mean": self.state.running_mean.clone(),
            "state.running_var": self.state.running_var.clone(),
            "state.unit_density": self.state.unit_density.clone(),
            "state.threshold_bias": self.state.threshold_bias.clone(),
            "state.steps_since_active": self.state.steps_since_active.clone(),
            "state.covariance_ema": self.state.covariance_ema.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore weights and homeostasis state saved by `state_dict()`."""
        self.W = state["W"].clone()
        self.A = state["A"].clone()
        self.b = state["b"].clone()
        self.H = state["H"].clone()
        self.E_token = state["E_token"].clone()
        self.b_tok = state["b_tok"].clone()
        self.state = ColumnState(
            h=state["state.h"].clone(),
            running_mean=state["state.running_mean"].clone(),
            running_var=state["state.running_var"].clone(),
            unit_density=state["state.unit_density"].clone(),
            threshold_bias=state["state.threshold_bias"].clone(),
            steps_since_active=state["state.steps_since_active"].clone(),
            covariance_ema=state["state.covariance_ema"].clone(),
        )

    # -- internals ------------------------------------------------------------

    def _rescue_dead_units(self) -> None:
        """Reinitialize the input row of any unit inactive past the window
        (section 6.2: "only after a measured inactivity window; never inject
        arbitrary activation into durable memory" -- this reseeds the weight
        row, not the activity trace itself).
        """
        cfg, s = self.config, self.state
        dead = s.steps_since_active >= cfg.dead_unit_window
        if not bool(dead.any()):
            return
        n_dead = int(dead.sum())
        scale = (1.0 / cfg.input_dim) ** 0.5
        self.W[dead] = torch.randn(n_dead, cfg.input_dim) * scale
        s.threshold_bias[dead] = 0.0
        s.steps_since_active[dead] = 0

    def _clip_rows(self, matrix: torch.Tensor) -> torch.Tensor:
        """Row-norm cap on recognition/head matrices (section 6.2)."""
        cap = self.config.weight_norm_cap
        row_norms = matrix.norm(dim=-1, keepdim=True)
        scale = torch.clamp(cap / torch.clamp_min(row_norms, cap), max=1.0)
        return matrix * scale
