"""Smoke tests for the multi-column PC-ALM trainer (train/pc_alm_trainer.py).

Checks the training loop's plumbing -- shapes, determinism, resumability,
locality of what gets updated -- not whether it solves any particular
task (that's the XOR probe's job; see test_alm_xor_probe.py and
docs/ManifoldFlow_ALM_Journal.md).
"""

from __future__ import annotations

import numpy as np
import torch

from manifoldflow.alm.learning.pc_alm import PCALMConfig, PCALMHierarchy
from manifoldflow.alm.model.plastic_column import PlasticColumn, PlasticColumnConfig
from manifoldflow.alm.train.pc_alm_trainer import train_pc_alm_hierarchy

VOCAB = 10
EOS = 0


def _make_hierarchy(
    seed: int = 0, n_layers: int = 2, state_dim: int = 12
) -> PCALMHierarchy:
    generator = torch.Generator().manual_seed(seed)
    columns = []
    for layer in range(n_layers):
        input_dim = 6 if layer == 0 else state_dim
        config = PlasticColumnConfig(
            input_dim=input_dim,
            state_dim=state_dim,
            token_dim=6,
            vocab_size=VOCAB,
            seed=seed + layer,
        )
        columns.append(PlasticColumn(config, generator=generator))
    return PCALMHierarchy(columns, PCALMConfig(settle_steps=2))


def _random_stream(n_tokens: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, VOCAB, size=n_tokens).astype(np.int64)


def test_training_runs_and_produces_a_log() -> None:
    hierarchy = _make_hierarchy(seed=1)
    stream = _random_stream(300, seed=1)
    stats, h_prev = train_pc_alm_hierarchy(hierarchy, stream, eos_id=EOS, log_every=100)

    assert stats.tokens_seen == len(stream) - 1
    assert len(stats.log) == (len(stream) - 1) // 100
    assert len(h_prev) == hierarchy.n_layers


def test_training_is_deterministic_given_the_same_seed() -> None:
    stream = _random_stream(200, seed=2)

    hierarchy_a = _make_hierarchy(seed=3)
    stats_a, _ = train_pc_alm_hierarchy(hierarchy_a, stream, eos_id=EOS, log_every=50)

    hierarchy_b = _make_hierarchy(seed=3)
    stats_b, _ = train_pc_alm_hierarchy(hierarchy_b, stream, eos_id=EOS, log_every=50)

    assert stats_a.log == stats_b.log
    for col_a, col_b in zip(hierarchy_a.columns, hierarchy_b.columns, strict=True):
        torch.testing.assert_close(col_a.W, col_b.W)
        torch.testing.assert_close(col_a.H, col_b.H)


def test_only_head_layer_H_and_E_token_change_other_layers_keep_theirs_fixed() -> None:
    """H/E_token (the token-head parameters) should only ever change for
    the head layer -- other layers' heads are never trained or scored.
    """
    hierarchy = _make_hierarchy(seed=4, n_layers=2)
    before = [(col.H.clone(), col.E_token.clone()) for col in hierarchy.columns]
    stream = _random_stream(200, seed=4)
    train_pc_alm_hierarchy(hierarchy, stream, eos_id=EOS, log_every=1000, head_layer=-1)

    h0_before, e0_before = before[0]
    assert torch.equal(hierarchy.columns[0].H, h0_before)
    assert torch.equal(hierarchy.columns[0].E_token, e0_before)
    h1_before, e1_before = before[1]
    assert not torch.equal(hierarchy.columns[1].H, h1_before)
    assert not torch.equal(hierarchy.columns[1].E_token, e1_before)


def test_resume_continues_the_same_trajectory_as_an_uninterrupted_run() -> None:
    stream = _random_stream(300, seed=5)
    half = len(stream) // 2

    continuous = _make_hierarchy(seed=6)
    continuous_stats, _ = train_pc_alm_hierarchy(
        continuous, stream, eos_id=EOS, log_every=1000
    )

    first_half = _make_hierarchy(seed=6)
    stats_a, h_prev_a = train_pc_alm_hierarchy(
        first_half, stream[: half + 1], eos_id=EOS, log_every=1000
    )
    stats_b, _ = train_pc_alm_hierarchy(
        first_half,
        stream,
        eos_id=EOS,
        log_every=1000,
        start_index=half,
        stats=stats_a,
        h_prev=h_prev_a,
    )

    for col_a, col_b in zip(first_half.columns, continuous.columns, strict=True):
        torch.testing.assert_close(col_a.W, col_b.W)
        torch.testing.assert_close(col_a.H, col_b.H)
    assert stats_b.tokens_seen == continuous_stats.tokens_seen
