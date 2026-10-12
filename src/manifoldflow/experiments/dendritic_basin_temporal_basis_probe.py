"""Compare exact-lag and event-specific temporal-basis motif routing."""

from __future__ import annotations

import argparse
import json
from statistics import mean

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    PredictiveMotifOpponentLearner,
)
from manifoldflow.dendritic_basin.routed_event_memory import (
    RoutedMotifOpponentLearner,
    TemporalBasisMotifOpponentLearner,
)

TEMPORAL_BINS = (1, 2, 4, 8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--minimum-gap", type=int, default=3)
    parser.add_argument("--maximum-gap", type=int, default=48)
    return parser.parse_args()


def _episode(
    rng: np.random.Generator,
    *,
    minimum_gap: int,
    maximum_gap: int,
) -> tuple[np.ndarray, int]:
    cue_left = (1, 2, 3, 4)
    cue_right = (5, 6, 7, 8)
    fillers = tuple(tuple(range(start, start + 4)) for start in range(20, 52, 4))
    target_left = (60, 70, 80, 90)
    target_right = (60, 71, 80, 90)
    tail = (100, 101, 102, 103)
    left = bool(rng.integers(0, 2))
    gap = int(rng.integers(minimum_gap, maximum_gap + 1))
    values = list(cue_left if left else cue_right)
    for _ in range(gap):
        values.extend(fillers[int(rng.integers(0, len(fillers)))])
    target_start = len(values)
    values.extend(target_left if left else target_right)
    values.extend(tail)
    return np.asarray(values, dtype=np.uint8), target_start + 1


def _settings() -> dict[str, object]:
    return {
        "confirmations": 1,
        "context_confirmations": 2,
        "surprise_bits": 0.0,
        "sequence_confirmations": 2,
        "event_confirmations": 2,
        "boundary_surprise_bits": 100.0,
        "minimum_motif_events": 4,
        "maximum_motif_events": 4,
    }


def _score(
    learner: PredictiveMotifOpponentLearner,
    episodes: list[tuple[np.ndarray, int]],
    *,
    learn: bool,
) -> float:
    bits = []
    for stream, target_index in episodes:
        probability = learner.observe_chunk_probabilities(stream, learn=learn)
        if not learn:
            bits.append(-np.log2(max(float(probability[target_index]), 1e-300)))
        learner.reset_episode()
    return float(mean(bits)) if bits else float("nan")


def run_probe(
    *,
    seeds: int = 5,
    train_episodes: int = 3000,
    eval_episodes: int = 500,
    minimum_gap: int = 3,
    maximum_gap: int = 48,
) -> dict[str, object]:
    rows = []
    settings = _settings()
    for seed in range(seeds):
        train_rng = np.random.default_rng(seed)
        eval_rng = np.random.default_rng(10_000 + seed)
        train = [
            _episode(
                train_rng,
                minimum_gap=minimum_gap,
                maximum_gap=maximum_gap,
            )
            for _ in range(train_episodes)
        ]
        valid = [
            _episode(
                eval_rng,
                minimum_gap=minimum_gap,
                maximum_gap=maximum_gap,
            )
            for _ in range(eval_episodes)
        ]
        local = PredictiveMotifOpponentLearner(**settings)
        exact = RoutedMotifOpponentLearner(
            **settings,
            max_alm_bytes=None,
            route_lags=TEMPORAL_BINS,
            route_confirmations=2,
            route_utility_decay=0.9,
            minimum_route_utility_bits=0.01,
        )
        temporal = TemporalBasisMotifOpponentLearner(
            **settings,
            max_alm_bytes=None,
            temporal_bins=TEMPORAL_BINS,
            route_confirmations=2,
            route_utility_decay=0.9,
            minimum_route_utility_bits=0.01,
        )
        for learner in (local, exact, temporal):
            _score(learner, train, learn=True)
        row = {
            "seed": seed,
            "local_target_bits": _score(local, valid, learn=False),
            "exact_lag_target_bits": _score(exact, valid, learn=False),
            "temporal_basis_target_bits": _score(temporal, valid, learn=False),
            "exact_routes": exact.live_route_count,
            "temporal_routes": temporal.live_route_count,
            "temporal_route_synapses": temporal.live_route_synapse_count,
            "temporal_embodied_bytes": temporal.embodied_bytes,
            "temporal_trace_capacity": temporal.max_active_temporal_traces,
        }
        rows.append(row)
    return {
        "seeds": seeds,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "minimum_gap": minimum_gap,
        "maximum_gap": maximum_gap,
        "local_target_bits_mean": mean(row["local_target_bits"] for row in rows),
        "exact_lag_target_bits_mean": mean(
            row["exact_lag_target_bits"] for row in rows
        ),
        "temporal_basis_target_bits_mean": mean(
            row["temporal_basis_target_bits"] for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_probe(
                seeds=args.seeds,
                train_episodes=args.train_episodes,
                eval_episodes=args.eval_episodes,
                minimum_gap=args.minimum_gap,
                maximum_gap=args.maximum_gap,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
