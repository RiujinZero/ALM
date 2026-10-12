"""Linear-equivalence and finite-difference checks for PC-ALM settling.

Design doc section 17 (software architecture): learning/pc_alm.py must pass
"linear equivalence and finite-difference checks."
"""

from __future__ import annotations

import torch

from manifoldflow.alm.learning.pc_alm import PCALMConfig, PCALMHierarchy
from manifoldflow.alm.model.plastic_column import PlasticColumn, PlasticColumnConfig


def _make_hierarchy(
    n_layers: int, dim: int = 6, nonlinear: bool = False, seed: int = 0
) -> PCALMHierarchy:
    g = torch.Generator().manual_seed(seed)
    columns = []
    for layer in range(n_layers):
        # token embedding dim == state dim for these small fixtures
        cfg = PlasticColumnConfig(
            input_dim=dim,
            state_dim=dim,
            token_dim=4,
            vocab_size=8,
            nonlinear=nonlinear,
            seed=seed + layer,
        )
        columns.append(PlasticColumn(cfg, generator=g))
    pc_alm_config = PCALMConfig(settle_steps=4, activity_lr=0.1, dual_lr=0.05, rho=1.0)
    return PCALMHierarchy(columns, pc_alm_config)


def test_finite_difference_gradient_matches_analytic() -> None:
    """dA/dh_l (the analytic activity-step gradient) must match numerical
    differentiation of `free_energy` for a middle layer with both an
    upstream (own residual) and downstream (neighbor) contribution.
    """
    torch.manual_seed(0)
    hierarchy = _make_hierarchy(n_layers=3, dim=5, nonlinear=False)
    x0 = torch.randn(5)
    h_prev = [torch.randn(5) for _ in range(3)]
    h = hierarchy.forward_initialize(x0, h_prev)
    h = [h_l + 0.1 * torch.randn_like(h_l) for h_l in h]  # perturb off the fixed point
    dual = [0.3 * torch.randn_like(h_l) for h_l in h]

    layer = 1  # has both a same-layer term and a layer-2 neighbor term
    residuals = hierarchy._all_residuals(x0, h, h_prev)
    analytic = hierarchy._activity_gradient(layer, x0, h, h_prev, dual, residuals)

    eps = 1e-4
    numeric = torch.zeros_like(analytic)
    for i in range(h[layer].numel()):
        h_plus = [t.clone() for t in h]
        h_minus = [t.clone() for t in h]
        h_plus[layer][i] += eps
        h_minus[layer][i] -= eps
        e_plus = hierarchy.free_energy(x0, h_plus, h_prev, dual)
        e_minus = hierarchy.free_energy(x0, h_minus, h_prev, dual)
        numeric[i] = (e_plus - e_minus) / (2 * eps)

    torch.testing.assert_close(analytic, numeric, atol=1e-3, rtol=1e-3)


def test_finite_difference_gradient_matches_with_token_pressure_at_head_layer() -> None:
    """At the layer that owns the token head, `extra_pressure` is a direct,
    unambiguous addition to that layer's own gradient -- CE(h_top) is
    literally a function of h_top alone, so finite-differencing
    `free_energy(..., token_energy_fn=...)` with respect to h_top is a
    faithful check. (Checking earlier layers this way would conflate two
    different, both legitimate, notions of "gradient": see
    `test_extra_pressure_propagates_to_earlier_layers_via_neighbor_term`
    for why that's tested algebraically instead.)
    """
    torch.manual_seed(5)
    hierarchy = _make_hierarchy(n_layers=3, dim=5, nonlinear=False)
    top = hierarchy.columns[-1]
    next_token = 3

    x0 = torch.randn(5)
    h_prev = [torch.randn(5) for _ in range(3)]
    init = hierarchy.forward_initialize(x0, h_prev)
    h = [h_l + 0.1 * torch.randn_like(h_l) for h_l in init]
    dual = [0.3 * torch.randn_like(h_l) for h_l in h]

    def token_energy_fn(h_list):
        logits, _q = top.token_logits(h_list[-1])
        return -torch.log_softmax(logits, dim=-1)[next_token]

    residuals = hierarchy._all_residuals(x0, h, h_prev)
    zero_extra = [torch.zeros_like(h_l) for h_l in h]
    delta = top.token_delta(h[-1], next_token)
    extra = list(zero_extra)
    extra[-1] = top.token_gradient_wrt_h(delta)

    layer = 2
    analytic = hierarchy._activity_gradient(
        layer, x0, h, h_prev, dual, residuals, extra
    )

    eps = 1e-4
    numeric = torch.zeros_like(analytic)
    for i in range(h[layer].numel()):
        h_plus = [t.clone() for t in h]
        h_minus = [t.clone() for t in h]
        h_plus[layer][i] += eps
        h_minus[layer][i] -= eps
        e_plus = hierarchy.free_energy(x0, h_plus, h_prev, dual, token_energy_fn)
        e_minus = hierarchy.free_energy(x0, h_minus, h_prev, dual, token_energy_fn)
        numeric[i] = (e_plus - e_minus) / (2 * eps)

    torch.testing.assert_close(analytic, numeric, atol=1e-3, rtol=1e-3)


def test_extra_pressure_propagates_to_earlier_layers_via_neighbor_term() -> None:
    """`extra_pressure` at layer l+1 must change layer l's gradient by
    exactly `-W_{l+1}^T @ (extra_pressure[l+1] * f'_{l+1})` -- the same
    neighbor-exchange formula the constraint pressure already uses,
    algebraically checked directly (not via finite-difference: propagating
    a downstream layer's *own* local loss to an upstream layer through the
    settling constraint is a total-derivative-through-the-constraint-
    manifold statement, not a partial derivative of a snapshot energy
    function with every h_l held as an independent variable, so a raw
    finite-difference check of the latter would not actually confirm the
    former -- which is exactly the mechanism this propagation is for).
    """
    torch.manual_seed(6)
    hierarchy = _make_hierarchy(n_layers=3, dim=5, nonlinear=True)
    x0 = torch.randn(5)
    h_prev = [torch.randn(5) for _ in range(3)]
    h = hierarchy.forward_initialize(x0, h_prev)
    dual = [0.2 * torch.randn_like(h_l) for h_l in h]
    residuals = hierarchy._all_residuals(x0, h, h_prev)

    layer = 1
    zero_extra = [torch.zeros_like(h_l) for h_l in h]
    without = hierarchy._activity_gradient(
        layer, x0, h, h_prev, dual, residuals, zero_extra
    )

    injected = torch.randn(5)
    with_extra = list(zero_extra)
    with_extra[layer + 1] = injected
    with_pressure = hierarchy._activity_gradient(
        layer, x0, h, h_prev, dual, residuals, with_extra
    )

    nxt = hierarchy.columns[layer + 1]
    a_next = nxt.pre_activation(h[layer], h_prev[layer + 1])
    support = (h[layer + 1] != 0).float()
    f_prime = (1 - torch.tanh(a_next) ** 2) * support
    expected_delta = -nxt.W.T @ (injected * f_prime)

    actual_delta = with_pressure - without
    torch.testing.assert_close(actual_delta, expected_delta, atol=1e-6, rtol=1e-6)


def test_finite_difference_gradient_matches_at_last_layer() -> None:
    """The last layer has no downstream neighbor term; check that edge case too."""
    torch.manual_seed(1)
    hierarchy = _make_hierarchy(n_layers=3, dim=5, nonlinear=False)
    x0 = torch.randn(5)
    h_prev = [torch.randn(5) for _ in range(3)]
    init = hierarchy.forward_initialize(x0, h_prev)
    h = [h_l + 0.1 * torch.randn_like(h_l) for h_l in init]
    dual = [0.3 * torch.randn_like(h_l) for h_l in h]

    layer = 2
    residuals = hierarchy._all_residuals(x0, h, h_prev)
    analytic = hierarchy._activity_gradient(layer, x0, h, h_prev, dual, residuals)

    eps = 1e-4
    numeric = torch.zeros_like(analytic)
    for i in range(h[layer].numel()):
        h_plus = [t.clone() for t in h]
        h_minus = [t.clone() for t in h]
        h_plus[layer][i] += eps
        h_minus[layer][i] -= eps
        e_plus = hierarchy.free_energy(x0, h_plus, h_prev, dual)
        e_minus = hierarchy.free_energy(x0, h_minus, h_prev, dual)
        numeric[i] = (e_plus - e_minus) / (2 * eps)

    torch.testing.assert_close(analytic, numeric, atol=1e-3, rtol=1e-3)


def test_linear_settling_converges_to_forward_fixed_point() -> None:
    """With linear (identity) activations, driving the augmented-Lagrangian
    residual to zero via repeated activity/dual steps must recover the same
    state as the direct closed-form forward computation -- i.e. PC-ALM
    settling is equivalent to ordinary forward evaluation in the degenerate
    linear, single-pass case. Starts away from the fixed point (via
    `initial_h`) so the check is non-trivial: the forward pass alone is
    already a zero-residual fixed point, so settling from there wouldn't
    exercise the activity/dual dynamics at all.
    """
    torch.manual_seed(2)
    hierarchy = _make_hierarchy(n_layers=2, dim=6, nonlinear=False)
    # Settling's own operational cap (max_settle_steps, default 8, section
    # 7.1's edge-efficiency gate) is irrelevant to this math-correctness
    # check, so it's raised here to let the primal-dual dynamics fully
    # converge; real training never overrides it.
    hierarchy.config.max_settle_steps = 5000
    x0 = torch.randn(6)
    h_prev = [torch.zeros(6), torch.zeros(6)]
    closed_form = hierarchy.forward_initialize(x0, h_prev)
    perturbed = [h_l + torch.randn_like(h_l) for h_l in closed_form]

    result = hierarchy.settle(x0, h_prev, steps=3000, initial_h=perturbed)

    for h_l, target_l in zip(result.h, closed_form, strict=False):
        torch.testing.assert_close(h_l, target_l, atol=1e-2, rtol=1e-2)
    for e_l in result.residuals:
        assert float(e_l.norm()) < 1e-2


def test_settle_respects_max_settle_steps_cap() -> None:
    hierarchy = _make_hierarchy(n_layers=2, dim=4, nonlinear=False)
    hierarchy.config.max_settle_steps = 3
    x0 = torch.randn(4)
    h_prev = [torch.zeros(4), torch.zeros(4)]
    result = hierarchy.settle(x0, h_prev, steps=100)
    assert result.iterations == 3


def test_update_weights_changes_only_W_of_each_column() -> None:
    hierarchy = _make_hierarchy(n_layers=3, dim=5, nonlinear=True)
    x0 = torch.randn(5)
    h_prev = [torch.zeros(5) for _ in range(3)]
    before = {
        id(col): (col.A.clone(), col.H.clone(), col.b.clone())
        for col in hierarchy.columns
    }

    result = hierarchy.settle(x0, h_prev)
    hierarchy.update_weights(x0, result, h_prev)

    for col in hierarchy.columns:
        a_before, h_before, b_before = before[id(col)]
        assert torch.equal(col.A, a_before)
        assert torch.equal(col.H, h_before)
        assert torch.equal(col.b, b_before)


def test_settle_reduces_residual_norm_relative_to_initialization() -> None:
    hierarchy = _make_hierarchy(n_layers=4, dim=8, nonlinear=True)
    x0 = torch.randn(8)
    h_prev = [torch.randn(8) * 0.1 for _ in range(4)]

    init = hierarchy.forward_initialize(x0, h_prev)
    init_residuals = hierarchy._all_residuals(x0, init, h_prev)
    init_residual = sum(float(r.norm()) for r in init_residuals)

    result = hierarchy.settle(x0, h_prev)
    final_residual = sum(float(r.norm()) for r in result.residuals)

    assert final_residual <= init_residual + 1e-6
