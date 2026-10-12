"""Deterministic replay, checkpoint/resume, and loss-trend smoke tests for
the M1 trainer.

Design doc section 17: train/stream_trainer.py must pass "deterministic
replay and crash recovery." Both halves are covered here: deterministic
replay (same seed -> same log) and crash recovery (checkpoint + resume
continues the same trajectory).
"""

from __future__ import annotations

import numpy as np
import torch

from manifoldflow.alm.model.plastic_column import PlasticColumn, PlasticColumnConfig
from manifoldflow.alm.train.stream_trainer import (
    load_checkpoint,
    resume_or_init,
    sanity_gate_report,
    save_checkpoint,
    train_single_column,
)

VOCAB = 24
EOS_ID = 2


def _make_column(seed: int = 0, **overrides) -> PlasticColumn:
    defaults = dict(
        input_dim=12,
        state_dim=48,
        token_dim=12,
        vocab_size=VOCAB,
        weight_lr=0.02,
        token_head_lr=0.05,
        token_embed_lr=0.05,
        seed=seed,
    )
    defaults.update(overrides)
    config = PlasticColumnConfig(**defaults)
    return PlasticColumn(config, generator=torch.Generator().manual_seed(seed))


def _synthetic_stream(n_repeats: int = 300) -> np.ndarray:
    """A short, deterministic "sentence" repeated many times, framed by EOS --
    trivially learnable so a working local rule should clearly beat unigram.
    """
    sentence = [5, 9, 14, 9, 20, EOS_ID]
    stream = sentence * n_repeats
    return np.asarray(stream, dtype=np.int64)


def test_training_reduces_token_ce_across_the_run() -> None:
    column = _make_column(seed=1)
    stream = _synthetic_stream()
    stats, _h_prev = train_single_column(column, stream, eos_id=EOS_ID, log_every=200)

    assert len(stats.log) >= 2
    assert stats.log[-1]["token_ce"] < stats.log[0]["token_ce"]


def test_training_is_reported_as_learning_on_a_trivially_learnable_stream() -> None:
    column = _make_column(seed=2)
    stream = _synthetic_stream()
    stats, _h_prev = train_single_column(column, stream, eos_id=EOS_ID, log_every=200)

    report = sanity_gate_report(stats, stream, VOCAB)
    assert report["is_learning"]
    assert report["loss_trend_ratio"] < 1.0
    # also clears unigram baseline handily -- informative, not a gate
    assert report["cross_entropy_reduction"] > 0.10


def test_training_is_deterministic_given_the_same_seed() -> None:
    stream = _synthetic_stream(n_repeats=100)

    column_a = _make_column(seed=7)
    stats_a, _ = train_single_column(column_a, stream, eos_id=EOS_ID, log_every=100)

    column_b = _make_column(seed=7)
    stats_b, _ = train_single_column(column_b, stream, eos_id=EOS_ID, log_every=100)

    assert stats_a.log == stats_b.log
    torch.testing.assert_close(column_a.W, column_b.W)
    torch.testing.assert_close(column_a.E_token, column_b.E_token)


def test_training_does_not_collapse_representation() -> None:
    """Non-collapse readouts (section 6.2/18.2): density near target, no
    single unit dominating, nonzero effective rank.
    """
    column = _make_column(seed=3)
    stream = _synthetic_stream()
    stats, _h_prev = train_single_column(column, stream, eos_id=EOS_ID, log_every=200)

    last = stats.log[-1]
    assert 0.0 < last["activation_density"] < 0.5
    assert last["max_unit_share"] < 0.35
    assert last["effective_rank"] > 0.0


def test_sanity_gate_report_contains_expected_fields() -> None:
    column = _make_column(seed=4)
    stream = _synthetic_stream(n_repeats=50)
    stats, _h_prev = train_single_column(column, stream, eos_id=EOS_ID, log_every=50)
    report = sanity_gate_report(stats, stream, VOCAB)

    for key in (
        "model_token_ce",
        "unigram_token_ce",
        "cross_entropy_reduction",
        "first_window_token_ce",
        "last_window_token_ce",
        "loss_trend_ratio",
        "is_learning",
        "final_activation_density",
        "final_max_unit_share",
        "final_effective_rank_fraction",
    ):
        assert key in report


def test_sanity_gate_report_is_learning_false_when_no_log_yet() -> None:
    from manifoldflow.alm.train.stream_trainer import TrainStats

    report = sanity_gate_report(TrainStats(), _synthetic_stream(n_repeats=1), VOCAB)
    assert report["is_learning"] is False


# -- checkpointing / resume ---------------------------------------------------


def test_checkpoint_round_trips_weights_and_stats(tmp_path) -> None:
    column = _make_column(seed=5)
    stream = _synthetic_stream(n_repeats=50)
    stats, h_prev = train_single_column(column, stream, eos_id=EOS_ID, log_every=50)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, column, stats, position=len(stream) - 1, h_prev=h_prev)
    checkpoint = load_checkpoint(path)

    assert checkpoint["position"] == len(stream) - 1
    assert checkpoint["stats"]["tokens_seen"] == stats.tokens_seen
    torch.testing.assert_close(checkpoint["column"]["W"], column.W)
    torch.testing.assert_close(checkpoint["h_prev"], h_prev)


def test_resume_or_init_returns_fresh_column_when_no_checkpoint(tmp_path) -> None:
    config = PlasticColumnConfig(
        input_dim=12, state_dim=48, token_dim=12, vocab_size=VOCAB
    )
    column, stats, start_index, h_prev = resume_or_init(
        tmp_path / "missing.pt", config, generator=torch.Generator().manual_seed(0)
    )
    assert start_index == 0
    assert stats.tokens_seen == 0
    torch.testing.assert_close(h_prev, torch.zeros(config.state_dim))
    assert isinstance(column, PlasticColumn)


def test_resume_or_init_rejects_mismatched_config(tmp_path) -> None:
    column = _make_column(seed=6)
    stream = _synthetic_stream(n_repeats=20)
    stats, h_prev = train_single_column(column, stream, eos_id=EOS_ID, log_every=20)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, column, stats, position=len(stream) - 1, h_prev=h_prev)

    wrong_config = PlasticColumnConfig(
        input_dim=12, state_dim=999, token_dim=12, vocab_size=VOCAB
    )
    try:
        resume_or_init(path, wrong_config)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError on config mismatch")


def test_periodic_checkpointing_writes_a_resumable_file(tmp_path) -> None:
    column = _make_column(seed=8)
    stream = _synthetic_stream(n_repeats=50)
    path = tmp_path / "checkpoint.pt"

    stats, h_prev = train_single_column(
        column,
        stream,
        eos_id=EOS_ID,
        log_every=50,
        checkpoint_path=path,
        checkpoint_every=100,
    )

    assert path.exists()
    checkpoint = load_checkpoint(path)
    assert checkpoint["position"] > 0
    assert checkpoint["position"] <= len(stream) - 1


def test_resume_continues_the_same_trajectory_as_an_uninterrupted_run() -> None:
    """Train continuously for N steps vs. train N/2, checkpoint, reload into
    a fresh column, and train the remaining N/2 -- final weights must match.
    `dead_unit_window` is set larger than the run length so the only
    non-deterministic piece (dead-unit rescue's unseeded RNG draw) never
    fires, making the two paths bit-for-bit comparable.
    """
    stream = _synthetic_stream(n_repeats=80)
    half = len(stream) // 2

    continuous = _make_column(seed=9, dead_unit_window=10_000)
    continuous_stats, _ = train_single_column(
        continuous, stream, eos_id=EOS_ID, log_every=1000
    )

    first_half_column = _make_column(seed=9, dead_unit_window=10_000)
    stats_a, h_prev_a = train_single_column(
        first_half_column, stream[: half + 1], eos_id=EOS_ID, log_every=1000
    )

    resumed_column = _make_column(seed=9, dead_unit_window=10_000)
    resumed_column.load_state_dict(first_half_column.state_dict())
    stats_b, _ = train_single_column(
        resumed_column,
        stream,
        eos_id=EOS_ID,
        log_every=1000,
        start_index=half,
        stats=stats_a,
        h_prev=h_prev_a,
    )

    torch.testing.assert_close(resumed_column.W, continuous.W)
    torch.testing.assert_close(resumed_column.E_token, continuous.E_token)
    torch.testing.assert_close(resumed_column.H, continuous.H)
    assert stats_b.tokens_seen == continuous_stats.tokens_seen
