"""Pressure-test bounded temporal-role routing above raw-byte motifs."""

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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--pressure-bytes", type=int, default=5000)
    parser.add_argument("--probe-alm-bytes", type=int, default=400_000)
    return parser.parse_args()


def _role_stream(
    episodes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    motifs = {
        "left": (1, 2, 3, 4),
        "right": (5, 6, 7, 8),
        "filler_a": (20, 21, 22, 23),
        "filler_b": (24, 25, 26, 27),
        "filler_c": (28, 29, 30, 31),
        "target_left": (50, 60, 80, 90),
        "target_right": (50, 70, 80, 90),
    }
    values: list[int] = []
    targets: list[bool] = []
    for _ in range(episodes):
        left = bool(rng.integers(0, 2))
        episode = (
            motifs["left" if left else "right"]
            + motifs["filler_a"]
            + motifs["filler_b"]
            + motifs["filler_c"]
            + motifs["target_left" if left else "target_right"]
        )
        values.extend(episode)
        targets.extend([False] * 17 + [True] + [False] * 2)
    return np.asarray(values, dtype=np.uint8), np.asarray(targets, dtype=np.bool_)


def _target_bits(probability: np.ndarray, mask: np.ndarray) -> float:
    selected = probability[mask & np.isfinite(probability)]
    return float(-np.log2(np.maximum(selected, 1e-300)).mean())


def run_probe(
    *,
    seeds: int = 5,
    train_episodes: int = 2000,
    eval_episodes: int = 500,
    pressure_bytes: int = 5000,
    probe_alm_bytes: int = 400_000,
) -> dict[str, object]:
    rows = []
    settings = dict(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        sequence_confirmations=2,
        event_confirmations=2,
        boundary_surprise_bits=100.0,
        minimum_motif_events=4,
        maximum_motif_events=4,
    )
    for seed in range(seeds):
        train, _ = _role_stream(train_episodes, np.random.default_rng(seed))
        valid, targets = _role_stream(
            eval_episodes, np.random.default_rng(10_000 + seed)
        )
        pressure = np.random.default_rng(20_000 + seed).integers(
            0, 256, size=pressure_bytes, dtype=np.uint8
        )
        local = PredictiveMotifOpponentLearner(**settings)
        routed = RoutedMotifOpponentLearner(
            **settings,
            max_alm_bytes=probe_alm_bytes,
            byte_budget_check_interval=max(1000, pressure_bytes),
            route_lags=(1, 2, 4),
            route_confirmations=2,
            route_utility_decay=0.9,
            minimum_route_utility_bits=0.01,
        )
        local.observe_chunk_probabilities(train, learn=True)
        routed.observe_chunk_probabilities(pressure, learn=True)
        routed.observe_chunk_probabilities(train, learn=True)
        routed.enforce_byte_budget(target_fraction=1.0)
        local.reset_episode()
        routed.reset_episode()
        local_probability = local.observe_chunk_probabilities(valid, learn=False)
        routed_probability = routed.observe_chunk_probabilities(valid, learn=False)
        rows.append(
            {
                "seed": seed,
                "local_target_bits": _target_bits(local_probability, targets),
                "routed_target_bits": _target_bits(routed_probability, targets),
                "embodied_bytes": routed.embodied_bytes,
                "serialized_alm_bytes": routed.serialized_size_bytes(),
                "byte_occupancy": routed.byte_occupancy,
                "live_routes": routed.live_route_count,
                "route_synapses": routed.live_route_synapse_count,
                "budget_checks": routed.budget_checks,
                "budget_evictions": routed.budget_evictions,
                "budget_mature_replacements": routed.budget_mature_replacements,
                "budget_nursery_losses": routed.budget_nursery_losses,
                "budget_evictions_by_tissue": routed.budget_evictions_by_tissue,
                "competition_history": routed.competition_history,
                "motif_recycles": routed.motif_recycles,
                "growth_frozen": routed.growth_frozen,
            }
        )
    return {
        "seeds": seeds,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "pressure_bytes": pressure_bytes,
        "probe_alm_bytes": probe_alm_bytes,
        "local_target_bits_mean": mean(row["local_target_bits"] for row in rows),
        "routed_target_bits_mean": mean(row["routed_target_bits"] for row in rows),
        "routed_gain_bits_mean": mean(
            row["local_target_bits"] - row["routed_target_bits"] for row in rows
        ),
        "embodied_bytes_mean": mean(row["embodied_bytes"] for row in rows),
        "serialized_alm_bytes_mean": mean(row["serialized_alm_bytes"] for row in rows),
        "byte_occupancy_mean": mean(row["byte_occupancy"] for row in rows),
        "budget_evictions_mean": mean(row["budget_evictions"] for row in rows),
        "budget_mature_replacements_mean": mean(
            row["budget_mature_replacements"] for row in rows
        ),
        "budget_nursery_losses_mean": mean(
            row["budget_nursery_losses"] for row in rows
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
                pressure_bytes=args.pressure_bytes,
                probe_alm_bytes=args.probe_alm_bytes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
