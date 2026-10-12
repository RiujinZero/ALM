"""Shape, locality, norm, and dead-unit tests for the plastic column.

Design doc section 17 (software architecture): model/plastic_column.py must
pass "shape, locality, norm, dead-unit tests."
"""

from __future__ import annotations

import torch

from manifoldflow.alm.model.plastic_column import PlasticColumn, PlasticColumnConfig


def _make_column(**overrides) -> PlasticColumn:
    defaults = dict(input_dim=16, state_dim=32, token_dim=12, vocab_size=20, seed=0)
    defaults.update(overrides)
    config = PlasticColumnConfig(**defaults)
    return PlasticColumn(config, generator=torch.Generator().manual_seed(config.seed))


def test_weight_shapes_match_config() -> None:
    col = _make_column()
    cfg = col.config
    assert col.W.shape == (cfg.state_dim, cfg.input_dim)
    assert col.A.shape == (cfg.state_dim, cfg.state_dim)
    assert col.b.shape == (cfg.state_dim,)
    assert col.H.shape == (cfg.token_dim, cfg.state_dim)
    assert col.E_token.shape == (cfg.vocab_size, cfg.token_dim)
    assert col.b_tok.shape == (cfg.vocab_size,)


def test_forward_output_shape_and_sparsity() -> None:
    col = _make_column()
    h_in = torch.randn(col.config.input_dim)
    h_prev = torch.zeros(col.config.state_dim)
    h = col.forward(h_in, h_prev)
    assert h.shape == (col.config.state_dim,)
    k = round(col.config.target_density * col.config.state_dim)
    assert (h != 0).sum() <= k


def test_reconstruction_shape_matches_input() -> None:
    col = _make_column()
    h = torch.zeros(col.config.state_dim)
    h[0] = 1.0
    recon = col.reconstruct(h)
    assert recon.shape == (col.config.input_dim,)


def test_activation_density_converges_toward_target() -> None:
    col = _make_column(state_dim=64, target_density=0.125)
    g = torch.Generator().manual_seed(1)
    h_prev = torch.zeros(col.config.state_dim)
    for _ in range(300):
        h_in = torch.randn(col.config.input_dim, generator=g)
        h = col.forward(h_in, h_prev)
        col.update_homeostasis(h)
    assert abs(col.activation_density() - col.config.target_density) < 0.05


def test_update_reconstruction_is_local_to_W() -> None:
    """A reconstruction update must touch only W -- not A, H, E_token, or b_tok."""
    col = _make_column()
    before = {
        "A": col.A.clone(),
        "H": col.H.clone(),
        "E_token": col.E_token.clone(),
        "b_tok": col.b_tok.clone(),
        "b": col.b.clone(),
    }
    h_in = torch.randn(col.config.input_dim)
    h = col.forward(h_in, torch.zeros(col.config.state_dim))
    col.update_reconstruction(h_in, h)

    assert torch.equal(col.A, before["A"])
    assert torch.equal(col.H, before["H"])
    assert torch.equal(col.E_token, before["E_token"])
    assert torch.equal(col.b_tok, before["b_tok"])
    assert torch.equal(col.b, before["b"])


def test_update_token_head_is_local_to_head_params() -> None:
    """A token-head update must touch only H, E_token, b_tok -- not W, A, b."""
    col = _make_column()
    before = {"W": col.W.clone(), "A": col.A.clone(), "b": col.b.clone()}
    h_in, h_prev = torch.randn(col.config.input_dim), torch.zeros(col.config.state_dim)
    h = col.forward(h_in, h_prev)
    col.update_token_head(h, next_token=3)

    assert torch.equal(col.W, before["W"])
    assert torch.equal(col.A, before["A"])
    assert torch.equal(col.b, before["b"])


def test_reconstruction_error_decreases_with_repeated_presentation() -> None:
    col = _make_column(weight_lr=0.05)
    h_in = torch.randn(col.config.input_dim, generator=torch.Generator().manual_seed(2))
    h_prev = torch.zeros(col.config.state_dim)

    errors = []
    for _ in range(200):
        h = col.forward(h_in, h_prev)
        e_rec = col.update_reconstruction(h_in, h)
        errors.append(float(e_rec.norm()))

    assert errors[-1] < errors[0]


def test_token_head_loss_decreases_with_repeated_target() -> None:
    col = _make_column(token_head_lr=0.1, token_embed_lr=0.1)
    h_in, h_prev = torch.randn(col.config.input_dim), torch.zeros(col.config.state_dim)
    h = col.forward(h_in, h_prev)
    target = 5

    def ce_loss() -> float:
        p = col.token_probs(h)
        return float(-torch.log(torch.clamp_min(p[target], 1e-12)))

    losses = [ce_loss()]
    for _ in range(100):
        col.update_token_head(h, next_token=target)
        losses.append(ce_loss())

    assert losses[-1] < losses[0]


def test_row_norm_cap_is_enforced_after_updates() -> None:
    cap = 1.0
    col = _make_column(
        weight_norm_cap=cap, weight_lr=5.0, token_head_lr=5.0, token_embed_lr=5.0
    )
    h_prev = torch.zeros(col.config.state_dim)
    g = torch.Generator().manual_seed(3)
    for _ in range(20):
        h_in = torch.randn(col.config.input_dim, generator=g) * 10
        h = col.forward(h_in, h_prev)
        col.update_reconstruction(h_in, h)
        next_token = int(torch.randint(0, col.config.vocab_size, (1,), generator=g))
        col.update_token_head(h, next_token=next_token)

    assert float(col.W.norm(dim=-1).max()) <= cap + 1e-4
    assert float(col.H.norm(dim=-1).max()) <= cap + 1e-4
    assert float(col.E_token.norm(dim=-1).max()) <= cap + 1e-4


def test_dead_unit_rescue_reinitializes_inactive_rows() -> None:
    col = _make_column(state_dim=8, dead_unit_window=5, target_density=0.125)
    original_row0 = col.W[0].clone()

    # Never activate unit 0 by always presenting zero input except a strong,
    # always-active drive elsewhere; unit 0's threshold bias stays neutral so
    # it should still occasionally fire from tanh(0)=0 -> never wins topk here
    # since all inputs are zero for it deterministically once its column of W
    # is zeroed out below.
    col.W[0] = 0.0
    h_prev = torch.zeros(col.config.state_dim)
    for _ in range(col.config.dead_unit_window + 2):
        h_in = torch.ones(col.config.input_dim)
        h = col.forward(h_in, h_prev)
        col.update_homeostasis(h)

    assert not torch.equal(col.W[0], original_row0)
    assert int(col.state.steps_since_active[0]) < col.config.dead_unit_window


def test_effective_rank_and_covariance_energy_are_finite() -> None:
    col = _make_column()
    g = torch.Generator().manual_seed(4)
    h_prev = torch.zeros(col.config.state_dim)
    for _ in range(50):
        h_in = torch.randn(col.config.input_dim, generator=g)
        h = col.forward(h_in, h_prev)
        col.update_homeostasis(h)
    assert col.effective_rank() >= 0.0
    assert torch.isfinite(col.off_diagonal_covariance_energy())


def test_deterministic_given_seed() -> None:
    col_a = _make_column(seed=42)
    col_b = _make_column(seed=42)
    assert torch.equal(col_a.W, col_b.W)
    assert torch.equal(col_a.A, col_b.A)
    assert torch.equal(col_a.H, col_b.H)
