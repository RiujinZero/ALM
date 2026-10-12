"""Unit tests for the unigram-baseline and reduction helpers used by the
D0 sanity gate (design doc section 18.2)."""

from __future__ import annotations

import math

import numpy as np

from manifoldflow.alm.eval.metrics import (
    bits_per_token,
    cross_entropy_reduction,
    unigram_cross_entropy,
)


def test_unigram_cross_entropy_of_uniform_stream_is_log_vocab() -> None:
    vocab = 8
    stream = np.arange(vocab)  # each token appears exactly once
    ce = unigram_cross_entropy(stream, vocab)
    assert math.isclose(ce, math.log(vocab), rel_tol=1e-6)


def test_unigram_cross_entropy_of_constant_stream_is_zero() -> None:
    stream = np.full(100, 3)
    assert unigram_cross_entropy(stream, vocab_size=10) == 0.0


def test_unigram_cross_entropy_of_empty_stream_is_zero() -> None:
    assert unigram_cross_entropy(np.array([], dtype=np.int64), vocab_size=10) == 0.0


def test_cross_entropy_reduction_matches_definition() -> None:
    assert math.isclose(cross_entropy_reduction(model_loss=3.0, unigram_loss=4.0), 0.25)


def test_cross_entropy_reduction_zero_unigram_is_safe() -> None:
    assert cross_entropy_reduction(model_loss=1.0, unigram_loss=0.0) == 0.0


def test_bits_per_token_matches_log2_conversion() -> None:
    nats = math.log(4)
    assert math.isclose(bits_per_token(nats), 2.0, rel_tol=1e-9)
