"""Phase 6 (design doc section 54, "Adaptive bell dendrites on generic
vectors"; section 50's computational-architecture goal of
`O(K_active * K_local)` rather than dense `O(D^2)` scanning; section 51's
bounded "dendritic units" / "active trace candidates" memory budget):
replace `relation_discovery.py`'s full `(vocab_size * n_timescales)`
enumeration with a bounded, self-organizing population of dendritic
slots.

`discover_candidate_relations` trains one `RelationUnit` for *every*
(trace_channel, timescale_index) pair against the anchor -- exhaustive,
and exactly the dense scanning section 50 says to avoid once the
vocabulary is not a handful of toy tokens. Here, a small fixed number of
population *slots* each sample a random candidate pair, train it with the
same `RelationUnit` / `relation_information_gain` machinery already
validated in Phases 2-5, and -- if the candidate fails the structural or
usefulness gate -- reallocate to a fresh random candidate rather than
giving up. Total candidate evaluations are bounded by
`population_size * max_candidates_per_slot`, independent of how large the
vocabulary grows; this is the "self-organizing" part -- population
pressure moves toward whichever candidates turn out to matter instead of
a designer (or an exhaustive scan) deciding up front.

Building the per-anchor trace snapshot array (`anchor_values`) below is
still `O(n_anchor_positions * n_timescales * vocab_size)` -- that cost
belongs to `build_stream_diary`'s trace representation itself, not to
candidate search, and is unchanged from `relation_discovery.py`. What
this module removes is the `O(vocab_size * n_timescales)` *search* cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldflow.dendritic_basin.relation_discovery import (
    _make_draw_batch,
    relation_information_gain,
)
from manifoldflow.dendritic_basin.relation_unit import RelationUnit


@dataclass
class DiscoveredRelation:
    key: tuple[int, int]
    unit: RelationUnit
    gain: float


@dataclass
class PopulationStats:
    total_candidate_evaluations: int
    slots_converged: int
    population_size: int
    max_candidates_per_slot: int


def discover_relations_by_self_organizing_population(
    diary: list[tuple[int, np.ndarray, int]],
    anchor: int,
    vocab_size: int,
    n_timescales: int,
    seed: int,
    population_size: int = 8,
    max_candidates_per_slot: int = 8,
    min_gain: float = 0.1,
    acceptance_threshold: float = 0.05,
    n_candidates: int = 2,
    rounds: int = 40,
    perturbations_per_round: int = 12,
    step_size: float = 0.3,
    episodes_per_candidate: int = 1,
    min_coefficient_of_variation: float = 0.1,
) -> tuple[list[DiscoveredRelation], PopulationStats]:
    """`population_size` independent slots each try up to
    `max_candidates_per_slot` random (trace_channel, timescale_index)
    candidates (never repeating a candidate another slot already tried),
    training a `RelationUnit` on each with the same coefficient-of-
    variation guard, quality gate, and retry-episodes logic as
    `discover_candidate_relations` -- a slot keeps a candidate only if it
    clears both the structural (`RelationUnit.is_accepted`) and
    usefulness (`relation_information_gain >= min_gain`) gates, otherwise
    it reallocates to a new random candidate. Returns the discovered
    relations sorted by descending gain, plus `PopulationStats` recording
    how many candidates were actually evaluated (for confirming the
    bounded-cost claim, not just the discovery quality).
    """
    rng = np.random.default_rng(seed)
    anchor_values = np.array([q for tok, q, _next in diary if tok == anchor])
    candidate_channels = [c for c in range(vocab_size) if c != anchor]

    discovered: dict[tuple[int, int], DiscoveredRelation] = {}
    tried: set[tuple[int, int]] = set()
    uid = 0
    total_evaluations = 0
    slots_converged = 0

    max_distinct_candidates = len(candidate_channels) * n_timescales

    for _slot in range(population_size):
        for _attempt in range(max_candidates_per_slot):
            if len(tried) >= max_distinct_candidates:
                break  # every possible candidate has already been tried
            trace_channel = int(rng.choice(candidate_channels))
            timescale_index = int(rng.integers(0, n_timescales))
            key = (trace_channel, timescale_index)
            while key in tried:
                trace_channel = int(rng.choice(candidate_channels))
                timescale_index = int(rng.integers(0, n_timescales))
                key = (trace_channel, timescale_index)
            tried.add(key)
            total_evaluations += 1

            channel_series = anchor_values[:, timescale_index, trace_channel]
            mean = float(np.mean(channel_series))
            std = float(np.std(channel_series))
            coefficient_of_variation = std / (abs(mean) + 1e-8)
            if coefficient_of_variation < min_coefficient_of_variation:
                continue  # reallocate: this candidate can't discriminate here

            draw_batch = _make_draw_batch(diary, anchor, trace_channel, timescale_index)
            best_unit: RelationUnit | None = None
            best_gain = -np.inf
            for _episode in range(episodes_per_candidate):
                unit = RelationUnit(
                    n_candidates=n_candidates,
                    seed=seed + uid,
                    acceptance_threshold=acceptance_threshold,
                )
                unit.adapt(draw_batch, rounds, perturbations_per_round, step_size, rng)
                uid += 1
                if not unit.is_accepted():
                    continue
                gain = relation_information_gain(unit, key, diary, anchor, vocab_size)
                if gain > best_gain:
                    best_gain = gain
                    best_unit = unit

            if best_unit is not None and best_gain >= min_gain:
                discovered[key] = DiscoveredRelation(
                    key=key, unit=best_unit, gain=best_gain
                )
                slots_converged += 1
                break  # this slot has converged; move on to the next slot

    results = sorted(discovered.values(), key=lambda item: item.gain, reverse=True)
    stats = PopulationStats(
        total_candidate_evaluations=total_evaluations,
        slots_converged=slots_converged,
        population_size=population_size,
        max_candidates_per_slot=max_candidates_per_slot,
    )
    return results, stats
