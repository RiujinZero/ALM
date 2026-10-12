"""Test whether restoration can autonomously organize a persistent goal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.goal_attractor import (
    GoalConditionedLocalController,
    HomeostaticGoalAttractor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _begin(learner: HomeostaticGoalAttractor) -> None:
    learner.reset_episode()
    learner.observe_deviation(1.0)


def _train(
    learner: HomeostaticGoalAttractor,
    *,
    episodes: int,
    gap: int,
    rng: np.random.Generator,
) -> None:
    # Curriculum is exposure, not a target label: first experience restoration,
    # then increasingly remote predictors of the same restorative event.
    for _ in range(episodes):
        _begin(learner)
        learner.observe_event(4)
        learner.observe_deviation(0.0)

        _begin(learner)
        learner.observe_event(2)
        for event in rng.integers(6, 16, size=gap):
            learner.observe_event(int(event))
        learner.observe_event(4)
        learner.observe_deviation(0.0)

        _begin(learner)
        learner.observe_event(0)
        for event in rng.integers(6, 16, size=gap):
            learner.observe_event(int(event))
        learner.observe_event(2)
        for event in rng.integers(6, 16, size=gap):
            learner.observe_event(int(event))
        learner.observe_event(4)
        learner.observe_deviation(0.0)

        # Candidate 1 is maximally predictable but never restores the drive.
        _begin(learner)
        learner.observe_event(1)
        for _ in range(2 * gap + 2):
            learner.observe_event(5)
        learner.observe_deviation(1.2)


def _controller_causal_swap() -> dict[str, object]:
    controller = GoalConditionedLocalController(4, 2)
    for module in range(4):
        for goal in range(2):
            desired = (module + goal) % 2
            for action in range(2):
                for _ in range(32):
                    controller.observe(module, goal, action, action == desired)
    goal_zero = [controller.action(module, 0) for module in range(4)]
    goal_one = [controller.action(module, 1) for module in range(4)]
    return {
        "goal_zero_actions": goal_zero,
        "goal_one_actions": goal_one,
        "changed_fraction": float(np.mean(np.asarray(goal_zero) != goal_one)),
        "embodied_bytes": controller.embodied_bytes,
    }


def _conditional_transition_entropy(sequences: list[list[int]]) -> float:
    counts: dict[int, dict[int, int]] = {}
    for sequence in sequences:
        for source, target in zip(sequence, sequence[1:], strict=False):
            row = counts.setdefault(source, {})
            row[target] = row.get(target, 0) + 1
    weighted_entropy = 0.0
    transitions = 0
    for row in counts.values():
        values = np.asarray(tuple(row.values()), dtype=np.float64)
        probabilities = values / values.sum()
        weighted_entropy += float(-np.sum(values * np.log2(probabilities)))
        transitions += int(values.sum())
    return weighted_entropy / max(1, transitions)


def _predictability_control(*, episodes: int, gap: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    restorative = []
    dead_loop = []
    for _ in range(episodes):
        restorative.append(
            [0]
            + rng.integers(6, 16, size=gap).tolist()
            + [2]
            + rng.integers(6, 16, size=gap).tolist()
            + [4]
        )
        dead_loop.append([1] + [5] * (2 * gap + 2))
    entropy = {
        0: _conditional_transition_entropy(restorative),
        1: _conditional_transition_entropy(dead_loop),
    }
    return {
        "selected_goal": min(entropy, key=entropy.get),
        "conditional_transition_bits": {
            str(key): value for key, value in entropy.items()
        },
    }


def run_probe(*, episodes: int = 200, gap: int = 8, seed: int = 0) -> dict[str, object]:
    conditions: dict[str, object] = {}
    for name, conditioned_gain in (("direct_only", 0.0), ("conditioned", 0.6)):
        learner = HomeostaticGoalAttractor(
            16,
            eligibility_decay=0.6,
            conditioned_gain=conditioned_gain,
            minimum_goal_value=0.05,
        )
        _train(
            learner,
            episodes=episodes,
            gap=gap,
            rng=np.random.default_rng(seed),
        )
        learner.observe_deviation(1.0, learn=False)
        selected = learner.select_goal([0, 1])
        persistence = []
        for _ in range(2 * gap):
            learner.observe_deviation(1.0, learn=False)
            persistence.append(learner.current_goal)
        completion = learner.observe_deviation(0.0, learn=False)
        conditions[name] = {
            "selected_goal": selected.goal,
            "restorative_candidate_value": learner.value(0),
            "predictable_loop_value": learner.value(1),
            "intermediate_value": learner.value(2),
            "terminal_value": learner.value(4),
            "persisted_neutral_steps": int(sum(goal == 0 for goal in persistence)),
            "dissolved_on_restoration": completion.goal is None,
            "local_updates": learner.local_updates,
            "embodied_bytes": learner.embodied_bytes,
            "peak_transient_bytes": learner.transient_bytes,
        }
    return {
        "task": "modality_neutral_restoration_conditioned_goal_chain",
        "episodes": episodes,
        "gap": gap,
        "primitive_drive": "non-negative deviation from one viable condition",
        "target_labels_supplied": False,
        "future_routes_supplied": False,
        "prediction_only_control": _predictability_control(
            episodes=episodes, gap=gap, seed=seed
        ),
        "conditions": conditions,
        "causal_goal_swap": _controller_causal_swap(),
    }


def main() -> None:
    args = parse_args()
    report = run_probe(episodes=args.episodes, gap=args.gap, seed=args.seed)
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
