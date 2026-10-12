"""Tests for bounded Phase 12 experiment accounting and telemetry."""

from __future__ import annotations

from io import StringIO

import pytest

from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticConfig,
    OnlineDendriticStreamLearner,
)
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    StreamingNGramControls,
    StreamMetrics,
    _run_stream,
)


def test_stream_metrics_keep_only_the_bounded_tail() -> None:
    metrics = StreamMetrics(tail_size=7)
    for index in range(1000):
        metrics.add(0.5, 0.25 if index % 2 else 0.5)

    assert metrics.count == 1000
    assert metrics.model_bits_per_byte == pytest.approx(1.0)
    assert len(metrics.tail_model_bits) == 7
    assert len(metrics.tail_base_bits) == 7

    restored = StreamMetrics.from_state_dict(metrics.state_dict())
    assert restored.state_dict() == metrics.state_dict()


def test_stream_telemetry_reports_byte_clock_rate_eta_and_anatomy() -> None:
    learner = OnlineDendriticStreamLearner(
        OnlineDendriticConfig(
            vocab_size=4,
            timescales=(1, 2),
            max_relations=8,
            max_nursery_candidates=8,
            max_active_relations=2,
            surprise_birth_bits=1.0,
            birth_cooldown=1,
            seed=2,
        )
    )
    output = StringIO()
    metrics = _run_stream(
        learner,
        [0, 1, 0, 2] * 4,
        learn=True,
        tail_size=5,
        progress_every=8,
        label="train",
        progress_output=output,
    )

    rendered = output.getvalue()
    assert metrics.count == 15
    assert len(metrics.tail_model_bits) == 5
    assert "8/16 bytes ( 50.0%)" in rendered
    assert "bytes/s" in rendered
    assert "ETA" in rendered
    assert "rolling" in rendered
    assert "relations" in rendered
    assert "nursery" in rendered
    assert "births/prunes" in rendered


def test_streaming_ngram_controls_learn_without_corpus_lists() -> None:
    controls = StreamingNGramControls(max_order=3)
    train = [1, 2, 1, 2] * 100
    for token_id in iter(train):
        controls.observe_train(token_id)
    controls.begin_evaluation()
    for token_id in iter([1, 2] * 20):
        controls.observe_eval(token_id)

    assert controls.bits_per_byte(2) < controls.bits_per_byte(1)
    assert controls.bits_per_byte(3) < controls.bits_per_byte(1)
    assert abs(controls.bits_per_byte(3) - controls.bits_per_byte(2)) < 0.01
