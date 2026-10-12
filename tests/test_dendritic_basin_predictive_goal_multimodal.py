from __future__ import annotations

import json

import numpy as np

from manifoldflow.experiments.dendritic_basin_predictive_goal_multimodal_smoke import (
    CONSEQUENCE_MODALITY,
    CUE_MODALITY,
    LaggedSuccessorPredictor,
    generate_multimodal_episodes,
    make_vocabulary,
    run_ablation,
    stratified_shuffle_consequences,
)


def test_generator_is_cross_modal_delayed_and_locally_ambiguous() -> None:
    vocabulary = make_vocabulary(goals=4, distractors=6)
    episodes = generate_multimodal_episodes(
        40,
        horizon=24,
        vocabulary=vocabulary,
        rng=np.random.default_rng(3),
        local_window=4,
    )

    assert all(len(episode.events) == 25 for episode in episodes)
    assert all(episode.modalities[0] == CUE_MODALITY for episode in episodes)
    assert all(episode.modalities[-1] == CONSEQUENCE_MODALITY for episode in episodes)
    assert all(
        np.all(episode.events[-5:-1] == vocabulary.query_symbol) for episode in episodes
    )
    assert set(int(episode.events[-1]) for episode in episodes) == set(
        vocabulary.consequence_symbols
    )
    # The learner-facing record intentionally contains no latent goal field.
    assert not hasattr(episodes[0], "goal")


def test_stratified_shuffle_removes_observable_cue_consequence_mapping() -> None:
    vocabulary = make_vocabulary(goals=4, distractors=6)
    episodes = generate_multimodal_episodes(
        64,
        horizon=16,
        vocabulary=vocabulary,
        rng=np.random.default_rng(4),
    )
    shuffled = stratified_shuffle_consequences(
        episodes,
        consequence_symbols=vocabulary.consequence_symbols,
        rng=np.random.default_rng(5),
    )

    for cue in vocabulary.cue_symbols:
        outcomes = [
            int(episode.events[-1])
            for episode in shuffled
            if int(episode.events[0]) == cue
        ]
        assert set(outcomes) == set(vocabulary.consequence_symbols)
        assert len(set(outcomes.count(value) for value in outcomes)) == 1


def test_successor_forecast_requires_its_persistent_delay_queue() -> None:
    vocabulary = make_vocabulary(goals=4, distractors=6)
    train = generate_multimodal_episodes(
        80,
        horizon=16,
        vocabulary=vocabulary,
        rng=np.random.default_rng(6),
    )
    learner = LaggedSuccessorPredictor(vocabulary.size, lag=16)
    learner.fit(train)
    episode = train[0]
    target = int(episode.events[-1])
    target_index = vocabulary.consequence_symbols.index(target)

    intact = learner.probabilities(episode.events[:-1], vocabulary.consequence_symbols)
    reset = learner.probabilities(
        episode.events[:-1], vocabulary.consequence_symbols, reset_index=8
    )

    assert int(np.argmax(intact)) == target_index
    assert intact[target_index] > 0.85
    np.testing.assert_allclose(reset, np.full(4, 0.25))


def test_ablation_separates_persistent_prediction_from_plausibility_controls() -> None:
    report = run_ablation(
        horizons=(8, 32, 128),
        train_episodes=320,
        eval_episodes=160,
        seed=11,
        distractors=8,
    )

    assert report["learner_observations"]["latent_goal_labels"] is False
    assert report["learner_observations"]["modality_ids"] is False
    assert report["aggregate"]["all_horizons_successor_better_than_local"]
    assert json.loads(json.dumps(report))["schema_version"] == 1

    for row in report["horizons"]:
        conditions = row["conditions"]
        assert row["final_local_context_conditional_entropy_bits"] == 2.0
        assert conditions["local_immediate"]["accuracy"] == 0.25
        assert conditions["far_future_successor"]["accuracy"] == 1.0
        assert conditions["far_future_successor"]["cross_entropy_bits"] < 0.1
        assert conditions["shuffled_successor"]["accuracy"] <= 0.5
        assert conditions["mid_episode_reset_successor"]["accuracy"] == 0.25
        assert conditions["far_future_successor"]["gain_over_local_bits"] > 1.8

    theme_accuracy = [
        row["conditions"]["slow_theme"]["accuracy"] for row in report["horizons"]
    ]
    assert theme_accuracy[-1] < theme_accuracy[0]
