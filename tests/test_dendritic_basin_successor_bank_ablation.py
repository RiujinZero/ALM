from __future__ import annotations

import json

import numpy as np

from manifoldflow.experiments.dendritic_basin_successor_bank_ablation import (
    Episode,
    fit_endpoint_codebook,
    fit_predictive_code,
    generate_identity_episodes,
    generate_training_episodes,
    make_world,
    run_ablation,
)


def test_identity_generator_crosses_order_and_novel_content() -> None:
    world = make_world(dimensions=12, train_contents=6, seed=2)
    episodes, audits = generate_identity_episodes(
        world,
        groups=5,
        min_delay=12,
        max_delay=24,
        local_window=4,
        seed=3,
    )

    assert len(episodes) == len(audits) == 20
    assert len({audit.delay for audit in audits}) > 1
    assert not hasattr(episodes[0], "direction")
    indexed = {
        (audit.group, audit.content, audit.direction): episode
        for episode, audit in zip(episodes, audits, strict=True)
    }
    for group in range(5):
        for content in ("familiar", "novel"):
            negative = indexed[(group, content, -1)]
            positive = indexed[(group, content, 1)]
            np.testing.assert_allclose(negative.events[0], positive.events[1])
            np.testing.assert_allclose(negative.events[1], positive.events[0])
            np.testing.assert_allclose(negative.events[2:-1], positive.events[2:-1])
            assert not np.allclose(negative.events[-1], positive.events[-1])


def test_predictive_code_is_fit_only_from_observed_episode_arrays() -> None:
    world = make_world(dimensions=12, train_contents=8, seed=4)
    train = generate_training_episodes(
        world,
        repeats_per_content=2,
        min_delay=12,
        max_delay=28,
        local_window=4,
        seed=5,
    )
    assert all(isinstance(episode, Episode) for episode in train)
    codebook = fit_endpoint_codebook(train, codes=2, iterations=10, seed=6)
    code = fit_predictive_code(train, codebook, ridge_alpha=0.5, temperature=0.2)

    first = train[0].events
    reversed_cue = first[0] - first[1]
    forward_cue = first[1] - first[0]
    assert np.linalg.norm(code.evidence(forward_cue)) > 0.05
    assert np.dot(code.evidence(forward_cue), code.evidence(reversed_cue)) < 0.0


def test_ablation_passes_variable_delay_and_identity_gates() -> None:
    report = run_ablation(
        dimensions=16,
        train_contents=20,
        repeats_per_content=3,
        identity_groups=32,
        min_delay=16,
        max_delay=80,
        half_lives=(8.0, 32.0, 128.0),
        fixed_lags=(16, 32, 64),
        ridge_alpha=0.5,
        seed=7,
    )

    assert report["learner_observations"]["trajectory_labels"] is False
    assert report["learner_observations"]["modality_ids"] is False
    assert report["task"]["distinct_delays"] > 20
    assert report["task"]["cue_content_set_max_absolute_error"] == 0.0
    assert report["gates"]["bank_beats_local"]
    assert report["gates"]["bank_joint_identity_above_90_percent"]
    assert report["gates"]["reset_removes_identity"]
    assert report["gates"]["shuffle_removes_identity"]

    conditions = report["conditions"]
    assert conditions["multi_timescale_predictive_bank"]["endpoint"]["accuracy"] > 0.9
    assert conditions["local"]["endpoint"]["accuracy"] <= 0.55
    assert (
        conditions["multi_timescale_predictive_bank"]["identity_2x2"][
            "same_trajectory_novel_content_preservation_rate"
        ]["positive"]
        > 0.9
    )
    assert json.loads(json.dumps(report))["schema_version"] == 1
