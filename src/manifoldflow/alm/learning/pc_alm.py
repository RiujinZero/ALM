"""Layer-local spatial credit via PC-ALM settling (design doc section 7, 7.1).

Each column stores a local dual variable that accumulates unresolved
disagreement between its settled activity and what its neighbors' weights
predict for it. A small number of synchronous local activity/dual
iterations exchange information between neighboring columns; weights update
only after the settled state, and never by differentiating through the
settling iterations (section 7.1: "do not differentiate through the
settling iterations").

  e_l = h_l - f_l(W_l h_{l-1}, A_l h_{l,prev})
  A = L_token + sum_l [u_l^T e_l + (rho_l/2)||e_l||^2] + sum_l J_local,l
  Delta W_l = eta_w * [((u_l + rho_l e_l) elementwise f'_l) h_{l-1}^T] + Delta W_local

The activity step below descends the augmented-Lagrangian free energy `A`
with respect to each h_l, including the pull from layer l+1's residual
(the only piece of "neighboring column" information h_l needs) -- this is
what makes settling more than an isolated per-column fixed point.

`L_token` (a column's own local token-head loss, section 8) is a second,
optional source of activity pressure: pass `extra_pressure_fn` to `settle`
to fold `dCE/dh` (see `PlasticColumn.token_gradient_wrt_h`) into a given
layer's pressure. Because that pressure is exactly what the neighbor term
above pulls on the preceding layer, a downstream column's token-head error
can reshape an upstream column's weights via settling -- the local
stand-in for what backprop's chain rule does through hidden layers, and
the mechanism this module didn't yet exercise before `extra_pressure_fn`
existed (see the implementation journal's temporal-XOR probe).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from manifoldflow.alm.model.plastic_column import PlasticColumn


@dataclass
class PCALMConfig:
    settle_steps: int = 4
    max_settle_steps: int = 8
    activity_lr: float = 0.10
    dual_lr: float = 0.05
    rho: float = 1.0
    clip_local_error: float = 3.0


@dataclass
class SettleResult:
    h: list[torch.Tensor]
    dual: list[torch.Tensor]
    residuals: list[torch.Tensor]
    iterations: int


class PCALMHierarchy:
    """A stack of plastic columns settled jointly with PC-ALM local dual dynamics."""

    def __init__(self, columns: list[PlasticColumn], config: PCALMConfig | None = None):
        if not columns:
            raise ValueError("hierarchy needs at least one column")
        self.columns = columns
        self.config = config or PCALMConfig()

    @property
    def n_layers(self) -> int:
        return len(self.columns)

    def forward_initialize(
        self, x0: torch.Tensor, h_prev: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Section 13.1 step 1: one ordinary forward pass through all columns."""
        h = []
        layer_input = x0
        for col, prev in zip(self.columns, h_prev, strict=True):
            h_layer = col.forward(layer_input, prev)
            h.append(h_layer)
            layer_input = h_layer
        return h

    def _layer_input(
        self, layer: int, x0: torch.Tensor, h: list[torch.Tensor]
    ) -> torch.Tensor:
        return x0 if layer == 0 else h[layer - 1]

    def _target(
        self,
        layer: int,
        x0: torch.Tensor,
        h: list[torch.Tensor],
        h_prev: list[torch.Tensor],
    ) -> torch.Tensor:
        col = self.columns[layer]
        a = col.pre_activation(self._layer_input(layer, x0, h), h_prev[layer])
        return col.activation(a)

    def residual(
        self,
        layer: int,
        x0: torch.Tensor,
        h: list[torch.Tensor],
        h_prev: list[torch.Tensor],
    ) -> torch.Tensor:
        """e_l = h_l - f_l(...)."""
        return h[layer] - self._target(layer, x0, h, h_prev)

    def _all_residuals(
        self, x0: torch.Tensor, h: list[torch.Tensor], h_prev: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        return [self.residual(layer, x0, h, h_prev) for layer in range(self.n_layers)]

    def free_energy(
        self,
        x0: torch.Tensor,
        h: list[torch.Tensor],
        h_prev: list[torch.Tensor],
        dual: list[torch.Tensor],
        token_energy_fn: Callable[[list[torch.Tensor]], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The augmented-Lagrangian energy `A` -- the constraint term always;
        `L_token` (section 7) only if `token_energy_fn` is given (it takes
        the current `h` list and returns a scalar, e.g. a column's own CE).
        `J_local` otherwise is handled by the column-local update methods.
        Used directly by the finite-difference gradient-check test.
        """
        cfg = self.config
        residuals = self._all_residuals(x0, h, h_prev)
        energy = torch.zeros(())
        for u_layer, e_layer in zip(dual, residuals, strict=True):
            energy = energy + u_layer @ e_layer + 0.5 * cfg.rho * (e_layer @ e_layer)
        if token_energy_fn is not None:
            energy = energy + token_energy_fn(h)
        return energy

    def _activity_gradient(
        self,
        layer: int,
        x0: torch.Tensor,
        h: list[torch.Tensor],
        h_prev: list[torch.Tensor],
        dual: list[torch.Tensor],
        residuals: list[torch.Tensor],
        extra_pressure: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Analytic dA/dh_l: this layer's own constraint pressure (plus any
        `extra_pressure`, e.g. dL_token/dh_l) minus the pull from layer
        l+1's *total* pressure (the neighbor-exchange term) -- so a
        downstream layer's `extra_pressure` reaches this layer too.
        """
        pressure = self._pressure(dual, residuals, extra_pressure)
        grad = pressure[layer]
        if layer + 1 < self.n_layers:
            nxt = self.columns[layer + 1]
            a_next = nxt.pre_activation(h[layer], h_prev[layer + 1])
            if nxt.config.nonlinear:
                support = (h[layer + 1] != 0).float()
                f_prime = (1 - torch.tanh(a_next) ** 2) * support
            else:
                f_prime = torch.ones_like(a_next)
            grad = grad - nxt.W.T @ (pressure[layer + 1] * f_prime)
        return grad

    def _pressure(
        self,
        dual: list[torch.Tensor],
        residuals: list[torch.Tensor],
        extra_pressure: list[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        """u_l + rho*e_l (+ extra_pressure_l if given) for every layer --
        the total per-layer force used by both the activity step and the
        post-settling weight update.
        """
        cfg = self.config
        base = [u_l + cfg.rho * e_l for u_l, e_l in zip(dual, residuals, strict=True)]
        if extra_pressure is None:
            return base
        return [b + e for b, e in zip(base, extra_pressure, strict=True)]

    def settle(
        self,
        x0: torch.Tensor,
        h_prev: list[torch.Tensor],
        steps: int | None = None,
        initial_h: list[torch.Tensor] | None = None,
        extra_pressure_fn: (
            Callable[[list[torch.Tensor]], list[torch.Tensor]] | None
        ) = None,
    ) -> SettleResult:
        """Section 7.1: initialize, then run `steps` local activity/dual iterations.

        `initial_h` overrides the ordinary forward-pass initialization (used
        by convergence tests that start away from the constraint fixed
        point); real training always leaves it unset.

        `extra_pressure_fn`, if given, is called with the current `h` list
        at each iteration and must return a same-length list of additional
        per-layer pressure (zero tensors for layers with none) -- e.g. a
        column's own `dL_token/dh` (section 7's `L_token` term). It is
        recomputed each iteration since it generally depends on the
        (changing) settled activity, unlike the constraint pressure's
        dual/residual terms which are already iteration-local.
        """
        cfg = self.config
        steps = cfg.settle_steps if steps is None else steps
        steps = min(steps, cfg.max_settle_steps)

        if initial_h is not None:
            h = [t.clone() for t in initial_h]
        else:
            h = self.forward_initialize(x0, h_prev)
        dual = [torch.zeros_like(h_layer) for h_layer in h]

        for _ in range(steps):
            residuals = self._all_residuals(x0, h, h_prev)
            extra = extra_pressure_fn(h) if extra_pressure_fn is not None else None
            grads = [
                self._activity_gradient(layer, x0, h, h_prev, dual, residuals, extra)
                for layer in range(self.n_layers)
            ]
            for layer in range(self.n_layers):
                grad = torch.clamp(
                    grads[layer], -cfg.clip_local_error, cfg.clip_local_error
                )
                h[layer] = h[layer] - cfg.activity_lr * grad
            residuals = self._all_residuals(x0, h, h_prev)
            for layer in range(self.n_layers):
                dual[layer] = dual[layer] + cfg.dual_lr * residuals[layer]

        final_residuals = self._all_residuals(x0, h, h_prev)
        return SettleResult(h=h, dual=dual, residuals=final_residuals, iterations=steps)

    def update_weights(
        self,
        x0: torch.Tensor,
        result: SettleResult,
        h_prev: list[torch.Tensor],
        extra_pressure: list[torch.Tensor] | None = None,
    ) -> None:
        """Delta W_l = eta_w*[(pressure_l elementwise f'_l) h_{l-1}^T] (section 7),
        where pressure_l = u_l + rho_l*e_l, plus `extra_pressure[l]` if given
        (e.g. a column's own dL_token/dh -- the design's `Delta W_local`).

        Column-local objectives (reconstruction/token head/homeostasis) are
        applied separately via `PlasticColumn.update_*`, matching the
        design's split between the PC-ALM constraint term and `J_local,l`.
        """
        pressure = self._pressure(result.dual, result.residuals, extra_pressure)
        for layer, col in enumerate(self.columns):
            h_in = self._layer_input(layer, x0, result.h)
            a = col.pre_activation(h_in, h_prev[layer])
            if col.config.nonlinear:
                support = (result.h[layer] != 0).float()
                f_prime = (1 - torch.tanh(a) ** 2) * support
            else:
                f_prime = torch.ones_like(a)
            signal = pressure[layer] * f_prime
            delta_w = col.config.weight_lr * torch.outer(signal, h_in)
            col.W = col._clip_rows(col.W + delta_w)
