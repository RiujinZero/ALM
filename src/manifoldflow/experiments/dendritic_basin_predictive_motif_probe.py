"""Cross-domain gates for prediction-error-segmented motif assemblies."""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    PredictiveMotifOpponentLearner,
    SequenceCortexOpponentLearner,
)

StreamGenerator = Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]]


def _delayed_symbols(
    episodes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    values: list[int] = []
    targets: list[bool] = []
    for _ in range(episodes):
        positive = bool(rng.integers(0, 2))
        episode = [1 if positive else 2] + [9] * 8 + [50, 60 if positive else 70]
        values.extend(episode)
        targets.extend([False] * (len(episode) - 1) + [True])
    return np.asarray(values, dtype=np.uint8), np.asarray(targets, dtype=np.bool_)


def _occluded_ring(
    episodes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    values: list[int] = []
    targets: list[bool] = []
    ring_size = 8
    for _ in range(episodes):
        direction = 1 if rng.integers(0, 2) else -1
        position = int(rng.integers(0, ring_size))
        episode = [10 if direction > 0 else 11]
        for _ in range(4):
            episode.append(100 + position)
            position = (position + direction) % ring_size
        hidden_steps = int(rng.integers(7, 13))
        position = (position + direction * hidden_steps) % ring_size
        episode.extend([90] * hidden_steps)
        episode.extend((91, 100 + position))
        values.extend(episode)
        targets.extend([False] * (len(episode) - 1) + [True])
    return np.asarray(values, dtype=np.uint8), np.asarray(targets, dtype=np.bool_)


def _time_warped_audio(
    episodes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    values: list[int] = []
    targets: list[bool] = []
    for _ in range(episodes):
        high = bool(rng.integers(0, 2))
        episode = [20 if high else 21]
        episode.extend([30 if high else 31] * int(rng.integers(1, 5)))
        episode.extend([40] * int(rng.integers(7, 13)))
        episode.extend((50, 51, 60 if high else 61))
        values.extend(episode)
        targets.extend([False] * (len(episode) - 1) + [True])
    return np.asarray(values, dtype=np.uint8), np.asarray(targets, dtype=np.bool_)


def _spider_cross_modal(
    episodes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    values: list[int] = []
    targets: list[bool] = []
    for _ in range(episodes):
        prey_left = bool(rng.integers(0, 2))
        episode = [70 if prey_left else 71]
        episode.extend([80] * int(rng.integers(7, 13)))
        episode.extend((90, 91, 92 if prey_left else 93))
        values.extend(episode)
        targets.extend([False] * (len(episode) - 1) + [True])
    return np.asarray(values, dtype=np.uint8), np.asarray(targets, dtype=np.bool_)


def _target_loss(probability: np.ndarray, targets: np.ndarray) -> float:
    selected = probability[targets & np.isfinite(probability)]
    return float(np.mean(-np.log2(np.maximum(selected, 1e-300))))


def _evaluate(name: str, generator: StreamGenerator) -> dict[str, float | int | str]:
    train, _ = generator(3000, np.random.default_rng(35))
    valid, targets = generator(1000, np.random.default_rng(36))
    settings = {
        "confirmations": 1,
        "context_confirmations": 2,
        "surprise_bits": 0.0,
        "sequence_confirmations": 2,
    }
    local = SequenceCortexOpponentLearner(**settings)
    motif = PredictiveMotifOpponentLearner(**settings, event_confirmations=2)
    local.observe_chunk_probabilities(train, learn=True)
    motif.observe_chunk_probabilities(train, learn=True)
    local.reset_episode()
    motif.reset_episode()
    local_probability = local.observe_chunk_probabilities(valid, learn=False)
    pruned_branches, pruned_synapses = motif.consolidate_event_anatomy(
        minimum_observations=3
    )
    motif_probability = motif.observe_chunk_probabilities(valid, learn=False)
    local_loss = _target_loss(local_probability, targets)
    motif_loss = _target_loss(motif_probability, targets)
    return {
        "task": name,
        "train_events": len(train),
        "valid_events": len(valid),
        "sequence_target_bits": local_loss,
        "motif_target_bits": motif_loss,
        "motif_gain_bits": local_loss - motif_loss,
        "pruned_event_branches": pruned_branches,
        "pruned_event_synapses": pruned_synapses,
        "motif_assemblies": motif.live_motif_count,
        "event_assemblies": motif.live_event_count,
        "learned_boundaries": motif.learned_boundaries,
    }


def run_probe() -> list[dict[str, float | int | str]]:
    return [
        _evaluate("delayed_symbols", _delayed_symbols),
        _evaluate("occluded_ring", _occluded_ring),
        _evaluate("time_warped_audio", _time_warped_audio),
        _evaluate("spider_cross_modal", _spider_cross_modal),
    ]


def main() -> None:
    print(json.dumps(run_probe(), indent=2))


if __name__ == "__main__":
    main()
