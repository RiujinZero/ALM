"""Controlled raw-byte assay for transient role/filler reinstatement."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.routed_event_memory import (
    GoalConditionedTemporalBasisMotifOpponentLearner,
)

SUFFIXES = ("aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh")
FILLER_UNITS = tuple(f"f{suffix}" for suffix in SUFFIXES)
RESULT_UNITS = tuple(f"r{suffix}" for suffix in SUFFIXES)
FILLERS = tuple(unit * 4 for unit in FILLER_UNITS)
RESULTS = tuple("result" + unit * 2 for unit in RESULT_UNITS)
DISTRACTORS = tuple(
    f"d{suffix}"
    for suffix in (
        "aa",
        "bb",
        "cc",
        "dd",
        "ee",
        "ff",
        "gg",
        "hh",
        "ii",
        "jj",
        "kk",
        "ll",
        "mm",
        "nn",
        "oo",
        "pp",
    )
)


@dataclass(frozen=True)
class Episode:
    stream: bytes
    answers: tuple[tuple[int, int, str, str], ...]
    writes: tuple[tuple[int, int, str], ...]


def _append(parts: bytearray, text: str) -> tuple[int, int]:
    start = len(parts)
    parts.extend(text.encode("ascii"))
    return start, len(parts)


def _episode(rng: np.random.Generator, distractors: int) -> Episode:
    selected = rng.choice(len(FILLERS), size=3, replace=False)
    alpha, beta, reassigned = (int(value) for value in selected)
    parts = bytearray()
    answers: list[tuple[int, int, str, str]] = []
    writes: list[tuple[int, int, str]] = []

    def assignment(key: str, index: int) -> None:
        _append(parts, f"{key}set")
        start, end = _append(parts, FILLERS[index])
        writes.append((start, end, FILLERS[index]))
        _append(parts, f"{RESULTS[index]}sep")

    def distract() -> None:
        # Long but learnably regular interference. Random high-entropy noise is
        # a separate relevance-gating test, not part of the minimal binder gate.
        phase = int(rng.integers(0, len(DISTRACTORS)))
        words = [
            DISTRACTORS[(phase + offset) % len(DISTRACTORS)]
            for offset in range(distractors)
        ]
        _append(parts, "".join(str(word) for word in words) + "sep")

    def query(key: str, index: int) -> None:
        _append(parts, f"{key}ask")
        start, end = _append(parts, RESULTS[index])
        answers.append((start, end, FILLERS[index], RESULTS[index]))
        _append(parts, "sep")

    assignment("kaa", alpha)
    assignment("kbb", beta)
    distract()
    query("kaa", alpha)
    query("kbb", beta)
    assignment("kaa", reassigned)
    distract()
    query("kaa", reassigned)
    query("kbb", beta)
    return Episode(bytes(parts), tuple(answers), tuple(writes))


def _learner(
    cells: int = 256,
    address_context: str = "structural_basin",
    binding_surprise_bits: float = 4.0,
) -> GoalConditionedTemporalBasisMotifOpponentLearner:
    return GoalConditionedTemporalBasisMotifOpponentLearner(
        goal_apical_weight=8.0,
        max_goal_assemblies=256,
        max_alm_bytes=64_000_000,
        temporal_bins=(1, 2, 4, 8, 16, 32, 64),
        max_active_temporal_traces=64,
        route_confirmations=3,
        max_event_motifs=2,
        event_confirmations=2,
        boundary_surprise_bits=3.0,
        minimum_motif_events=3,
        maximum_motif_events=16,
        goal_permission_trial_limit=0,
        goal_third_component="temporal_disagreement",
        goal_route_context="restorative_basin",
        goal_permission_context="direction",
        maximum_goal_route_bonus=0.5,
        maximum_goal_route_byte_weight=0.02,
        goal_route_projection_policy="aligned_flow",
        goal_route_timing="learned_hazard",
        goal_route_alignment="restoration_signature",
        goal_flow_polarity="regulatory_basin",
        predictive_goal_basins=16,
        predictive_goal_birth_similarity=0.75,
        predictive_goal_hysteresis=0.5,
        predictive_goal_weight=1.0,
        binding_index_cells=cells,
        binding_role_assemblies=32,
        binding_cells_per_association=2,
        binding_activity_decay=0.98,
        binding_minimum_activity=0.05,
        binding_role_birth_similarity=0.8,
        binding_surprise_bits=binding_surprise_bits,
        binding_address_context=address_context,
        goal_flow_mode_gate="learned_utility",
        goal_flow_mode_trials=128,
        goal_surprise_scope="all_bytes",
    )


def _oracle_route(
    learner: GoalConditionedTemporalBasisMotifOpponentLearner,
    fillers: set[int],
) -> tuple[int, int] | None:
    candidates = [
        (receptor, filler)
        for filler in fillers
        for receptor in learner.temporal_bins
        if learner.route_utility.get((receptor, filler), 0.0)
        > learner.minimum_route_utility_bits
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda key: (learner.route_utility[key], learner.route_totals[key]),
    )


def _inject(
    learner: GoalConditionedTemporalBasisMotifOpponentLearner,
    filler: int,
    receptor: int,
) -> None:
    learner.temporal_last_seen[filler] = learner.temporal_clock - receptor + 1
    learner.temporal_recalled_sources.add(filler)
    learner._goal_route_projection_cache = None


def _evaluate(
    frozen: bytes,
    episodes: list[Episode],
    mode: str,
) -> dict[str, float | int]:
    learner = pickle.loads(frozen)
    if mode != "learned_binder":
        learner.binding_index = None
    bits = 0.0
    scored = 0
    correct = 0
    decision_bits = 0.0
    decision_correct = 0
    decisions = 0
    oracle_available = 0
    oracle_anatomical = 0
    reinstatements_before = learner.binding_reinstatements
    formations_before = learner.binding_formation_events
    for episode in episodes:
        learner.reset_episode()
        observed_fillers: dict[str, set[int]] = {}
        writes_by_position = {
            position: filler
            for start, end, filler in episode.writes
            for position in range(start, end)
        }
        answer_by_start = {
            start: (filler, result)
            for start, _, filler, result in episode.answers
        }
        answer_positions = {
            position
            for start, end, _, _ in episode.answers
            for position in range(start, end)
        }
        for position, byte_id in enumerate(episode.stream):
            if position in answer_by_start and mode.startswith("oracle"):
                filler_form, result_form = answer_by_start[position]
                del result_form
                correct_sources = observed_fillers.get(filler_form, set())
                selected = _oracle_route(learner, correct_sources)
                oracle_anatomical += int(selected is not None)
                if selected is not None:
                    receptor, filler = selected
                    if filler not in learner.temporal_last_seen:
                        oracle_available += 1
                        chosen_filler = filler
                        chosen_receptor = receptor
                        if mode == "oracle_wrong_receptor":
                            receptor_index = learner.temporal_bins.index(receptor)
                            chosen_receptor = learner.temporal_bins[
                                (receptor_index + 1) % len(learner.temporal_bins)
                            ]
                        elif mode == "oracle_wrong_filler":
                            alternative_sources = set().union(
                                *(
                                    sources
                                    for name, sources in observed_fillers.items()
                                    if name != filler_form
                                )
                            ) if len(observed_fillers) > 1 else set()
                            alternatives = [
                                source
                                for source in alternative_sources
                                if learner.route_utility.get((receptor, source), 0.0)
                                > learner.minimum_route_utility_bits
                                and source not in learner.temporal_last_seen
                            ]
                            chosen_filler = alternatives[0] if alternatives else None
                        if chosen_filler is not None:
                            _inject(learner, chosen_filler, chosen_receptor)
            distribution = learner.predict_next_distribution()
            probability = float(distribution[int(byte_id)])
            if position in answer_positions:
                bits += -log2(max(probability, 1e-300))
                correct += int(int(np.argmax(distribution)) == int(byte_id))
                scored += 1
            # The common six-byte prefix is "result" and the following "r"
            # begins the identity unit; its second byte is the first decision.
            if any(position == start + 7 for start, _, _, _ in episode.answers):
                decision_bits += -log2(max(probability, 1e-300))
                decision_correct += int(int(np.argmax(distribution)) == int(byte_id))
                decisions += 1
            previous_clock = learner.temporal_clock
            learner._observe_probability_only(int(byte_id), learn=False)
            filler_write = writes_by_position.get(position)
            if filler_write is not None and learner.temporal_clock > previous_clock:
                completed = {
                    source
                    for source, timestamp in learner.temporal_last_seen.items()
                    if timestamp == learner.temporal_clock
                    and source not in learner.temporal_recalled_sources
                }
                observed_fillers.setdefault(filler_write, set()).update(completed)
    return {
        "answer_bpb": bits / max(1, scored),
        "answer_byte_accuracy": correct / max(1, scored),
        "answer_bytes": scored,
        "identity_byte_bpb": decision_bits / max(1, decisions),
        "identity_byte_accuracy": decision_correct / max(1, decisions),
        "identity_bytes": decisions,
        "oracle_anatomical_routes": oracle_anatomical,
        "oracle_available_reads": oracle_available,
        "binding_formations": learner.binding_formation_events - formations_before,
        "binding_reinstatements": (
            learner.binding_reinstatements - reinstatements_before
        ),
    }


def _run_seed(
    seed: int,
    train_episodes: int,
    eval_episodes: int,
    distractors: int,
    address_context: str,
    binding_surprise_bits: float,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    train = [_episode(rng, distractors) for _ in range(train_episodes)]
    evaluate = [_episode(rng, distractors) for _ in range(eval_episodes)]
    learner = _learner(
        address_context=address_context,
        binding_surprise_bits=binding_surprise_bits,
    )
    started = time.perf_counter()
    train_bytes = 0
    for episode in train:
        learner.reset_episode()
        learner.observe_chunk_probabilities(
            np.frombuffer(episode.stream, dtype=np.uint8), learn=True
        )
        train_bytes += len(episode.stream)
    train_seconds = time.perf_counter() - started
    learner.reset_episode()
    frozen = pickle.dumps(learner, protocol=pickle.HIGHEST_PROTOCOL)
    conditions = {
        mode: _evaluate(frozen, evaluate, mode)
        for mode in (
            "off",
            "learned_binder",
            "oracle_correct",
            "oracle_wrong_receptor",
            "oracle_wrong_filler",
        )
    }
    return {
        "seed": seed,
        "binding_address_context": address_context,
        "train_bytes": train_bytes,
        "train_bytes_per_second": train_bytes / max(train_seconds, 1e-12),
        "embodied_bytes": learner.embodied_bytes,
        "binding_reinstatements": learner.binding_reinstatements,
        "conditions": conditions,
    }


def run_probe(
    *,
    seeds: tuple[int, ...],
    train_episodes: int,
    eval_episodes: int,
    distractors: int,
    workers: int,
    address_context: str,
    binding_surprise_bits: float,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_seed,
                seed,
                train_episodes,
                eval_episodes,
                distractors,
                address_context,
                binding_surprise_bits,
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: int(result["seed"]))
    return {
        "task": "phase72_controlled_raw_byte_binding",
        "description": (
            "two randomized associations, long distractors, within-episode "
            "reassignment, and repeated structure with different fillers"
        ),
        "binding_address_context": address_context,
        "binding_surprise_bits": binding_surprise_bits,
        "seeds": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--distractors", type=int, default=70)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--address-context",
        choices=(
            "structural_basin",
            "query_only",
            "cortical_query",
            "raw_query",
        ),
        default="structural_basin",
    )
    parser.add_argument("--binding-surprise-bits", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe(
        seeds=tuple(args.seeds),
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        distractors=args.distractors,
        workers=args.workers,
        address_context=args.address_context,
        binding_surprise_bits=args.binding_surprise_bits,
    )
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
