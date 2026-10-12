"""Harness-correctness tests for the temporal-XOR probe.

These check the probe's plumbing (label correctness, scoring logic) only
-- NOT whether the local learning rule actually solves XOR. That's an
open empirical question answered by running
`experiments/manifoldflow_alm_xor_probe.py` and recorded in
docs/ManifoldFlow_ALM_Journal.md, not something to hard-code a pass/fail
assertion for here.
"""

from __future__ import annotations

import torch

from manifoldflow.alm.eval.xor_probe import (
    EOS,
    LABEL0,
    LABEL1,
    ONE,
    SEP,
    VOCAB_SIZE,
    ZERO,
    build_xor_stream,
    evaluate_at_label_positions,
)
from manifoldflow.alm.model.plastic_column import PlasticColumn, PlasticColumnConfig


def _decode_example(tokens, start: int, with_separator: bool) -> tuple[int, int, int]:
    """Read (a, b, label) back out of the stream starting at a bit_a token."""
    a = 1 if tokens[start] == ONE else 0
    cursor = start + (2 if with_separator else 1)
    b = 1 if tokens[cursor] == ONE else 0
    label_token = tokens[cursor + 1]
    label = 1 if label_token == LABEL1 else 0
    return a, b, label


def test_build_xor_stream_labels_are_correct_xor_no_separator() -> None:
    tokens, label_positions = build_xor_stream(200, seed=0, with_separator=False)
    for pos in label_positions:
        a, b, label = _decode_example(tokens, pos - 1, with_separator=False)
        assert label == (a ^ b)
        assert tokens[pos + 2] == EOS


def test_build_xor_stream_labels_are_correct_xor_with_separator() -> None:
    tokens, label_positions = build_xor_stream(200, seed=1, with_separator=True)
    for pos in label_positions:
        a, b, label = _decode_example(tokens, pos - 2, with_separator=True)
        assert label == (a ^ b)
        assert tokens[pos - 1] == SEP


def test_build_xor_stream_uses_only_expected_vocabulary() -> None:
    tokens, _ = build_xor_stream(100, seed=2, with_separator=True)
    assert set(tokens.tolist()) <= {SEP, ZERO, ONE, LABEL0, LABEL1, EOS}
    assert VOCAB_SIZE == 6


def test_build_xor_stream_labels_are_balanced() -> None:
    tokens, label_positions = build_xor_stream(2000, seed=3, with_separator=False)
    labels = [1 if tokens[pos + 1] == LABEL1 else 0 for pos in label_positions]
    fraction_true = sum(labels) / len(labels)
    assert 0.4 < fraction_true < 0.6  # a^b over uniform iid bits is exactly balanced


def test_label_positions_count_matches_n_examples() -> None:
    _tokens, label_positions = build_xor_stream(500, seed=4, with_separator=True)
    assert len(label_positions) == 500


def test_evaluate_scores_exactly_n_examples() -> None:
    config = PlasticColumnConfig(
        input_dim=8, state_dim=16, token_dim=8, vocab_size=VOCAB_SIZE
    )
    column = PlasticColumn(config, generator=torch.Generator().manual_seed(0))
    tokens, label_positions = build_xor_stream(50, seed=5, with_separator=False)
    result = evaluate_at_label_positions(column, tokens, label_positions)
    assert result["n_scored"] == 50
    assert 0.0 <= result["accuracy"] <= 1.0


def test_evaluate_reports_perfect_accuracy_for_a_hand_crafted_perfect_head() -> None:
    """Validates the scoring path against ground truth: force the token
    head to always output the correct label logit and check accuracy=1.0,
    so a future bug in the scoring loop (e.g. off-by-one on positions)
    would be caught even though no real training happens here.
    """
    config = PlasticColumnConfig(
        input_dim=8, state_dim=16, token_dim=8, vocab_size=VOCAB_SIZE
    )
    column = PlasticColumn(config, generator=torch.Generator().manual_seed(0))
    tokens, label_positions = build_xor_stream(30, seed=6, with_separator=True)

    # Monkeypatch token_probs to always return the true next label token
    # with probability 1: token_probs is only called at label positions
    # (see evaluate_at_label_positions), and those occur in the same
    # increasing order as `label_positions` itself, so a call counter can
    # index straight into the true labels in order. This isolates "is the
    # scoring loop correct" from "can the model learn."
    true_labels_in_order = [int(tokens[pos + 1]) for pos in label_positions]
    call_count = {"n": 0}

    def fake_token_probs(h: torch.Tensor) -> torch.Tensor:
        p = torch.full((VOCAB_SIZE,), 1e-9)
        p[true_labels_in_order[call_count["n"]]] = 1.0
        call_count["n"] += 1
        return p

    column.token_probs = fake_token_probs

    result = evaluate_at_label_positions(column, tokens, label_positions)
    assert result["accuracy"] == 1.0
    assert call_count["n"] == len(label_positions)
