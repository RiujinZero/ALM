"""Phase 133 delayed relational confirmation and no-match smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin import temporal_relational_eligibility as temporal

DELAYS = (8, 16, 32, 64, 128, 256, 512)


def _patterns(seed: int, count: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dimensions = 384
    common = rng.choice(64, size=12, replace=False)
    patterns = np.zeros((count, dimensions), dtype=np.float64)
    consequences = np.zeros((count, count), dtype=np.float64)
    for index in range(count):
        patterns[index, common] = 1.0
        start = 64 + index * 10
        patterns[index, start : start + 10] = 1.0
        consequences[index, index] = 1.0
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
    return patterns, consequences


def _read(
    patterns: np.ndarray,
    query: np.ndarray,
    *,
    threshold: float = 0.58,
    power: int = 8,
) -> np.ndarray:
    norm = float(np.linalg.norm(query))
    if norm <= 0.0:
        return np.zeros(len(patterns), dtype=np.float64)
    drive = patterns @ (query / norm)
    activity = np.maximum(0.0, drive - threshold) ** power
    total = float(np.sum(activity))
    return activity / total if total > 0.0 else activity


def _causal_utility(
    baseline: np.ndarray, predicted: np.ndarray, observed: int
) -> float:
    on = 0.5 * baseline + 0.5 * predicted
    return float(
        np.log2(max(on[observed], 1e-300))
        - np.log2(max(baseline[observed], 1e-300))
    )


def run_seed(seed: int) -> dict[str, object]:
    patterns, consequences = _patterns(seed)
    baseline = np.full(len(patterns), 1.0 / len(patterns))
    rng = np.random.default_rng(seed + 10_000)
    rows = []
    for delay in DELAYS:
        learner = temporal.TemporalRelationalEligibility()
        target = delay % len(patterns)
        active = np.flatnonzero(patterns[target] > 0.0)
        retained = rng.choice(
            active,
            size=max(1, round(0.65 * len(active))),
            replace=False,
        )
        query = np.zeros(patterns.shape[1], dtype=np.float64)
        query[retained] = 1.0
        weights = _read(patterns, query)
        for relation in np.flatnonzero(weights > 0.0):
            learner.participate(int(relation), float(weights[relation]))
        learner.advance(delay)
        utilities = {
            int(relation): _causal_utility(
                baseline, consequences[int(relation)], target
            )
            for relation in np.flatnonzero(weights > 0.0)
        }
        learner.confirm(utilities, external=True)
        rows.append(
            {
                "delay": delay,
                "correct_read_mass": float(weights[target]),
                "trace": learner.trace(target).tolist(),
                "learned_strength": learner.strength(target).tolist(),
                "permission": learner.permission(target),
            }
        )

    # Structurally familiar common context plus an entity never stored.
    common = np.flatnonzero(np.sum(patterns > 0.0, axis=0) == len(patterns))
    unused = np.flatnonzero(np.sum(patterns, axis=0) == 0.0)
    no_match_query = np.zeros(patterns.shape[1], dtype=np.float64)
    no_match_query[common] = 1.0
    no_match_query[rng.choice(unused, size=10, replace=False)] = 1.0
    no_match = _read(patterns, no_match_query)
    no_match_learner = temporal.TemporalRelationalEligibility()
    for relation in np.flatnonzero(no_match > 0.0):
        no_match_learner.participate(int(relation), float(no_match[relation]))
    no_match_learner.confirm({}, external=True)
    return {
        "seed": seed,
        "delays": rows,
        "no_match": {
            "active_relations": int(np.count_nonzero(no_match)),
            "total_read_mass": float(np.sum(no_match)),
            "maximum_read_mass": float(np.max(no_match, initial=0.0)),
            "eligibility_relations": len(no_match_learner.traces),
            "external_updates": no_match_learner.external_updates,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = [run_seed(index) for index in range(args.seeds)]
    aggregate = []
    for index, delay in enumerate(DELAYS):
        aggregate.append(
            {
                "delay": delay,
                "mean_permission": float(
                    np.mean([row["delays"][index]["permission"] for row in seeds])
                ),
                "mean_trace": np.mean(
                    [row["delays"][index]["trace"] for row in seeds], axis=0
                ).tolist(),
            }
        )
    report = {
        "task": "phase133_temporally_eligible_relational_coupling",
        "half_lives": [8, 32, 128, 512],
        "aggregate": aggregate,
        "no_match": {
            key: float(np.mean([row["no_match"][key] for row in seeds]))
            for key in seeds[0]["no_match"]
        },
        "seeds": seeds,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
