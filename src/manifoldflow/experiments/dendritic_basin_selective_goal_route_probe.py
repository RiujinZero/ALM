"""Test deficit-gated route selection with delayed modality-neutral restoration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.routed_event_memory import (
    GoalConditionedTemporalBasisMotifOpponentLearner,
)


class _ProbeSelector(GoalConditionedTemporalBasisMotifOpponentLearner):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(max_alm_bytes=None, **kwargs)
        self.probe_active_keys: tuple[tuple[int, int], ...] = ()

    def _active_route_keys(self) -> tuple[tuple[int, int], ...]:
        return self.probe_active_keys


def _teach_delayed_restoration(
    learner: _ProbeSelector,
    key: tuple[int, int],
    mode: int,
    signal: float,
    delay_steps: int,
) -> None:
    learner.goal_route_eligibility.clear()
    learner._advance_goal_route_credit((key,), (0.0, 0.0, 0.0), learn=True)
    for _ in range(delay_steps):
        learner._advance_goal_route_credit((), (0.0, 0.0, 0.0), learn=True)
    outcome = [0.0, 0.0, 0.0]
    outcome[mode] = signal
    learner._advance_goal_route_credit((), tuple(outcome), learn=True)


def _accuracy(
    learner: _ProbeSelector,
    route_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[float, list[list[int]]]:
    correct = 0
    selected: list[list[int]] = []
    for module, pair in enumerate(route_pairs):
        module_selection = []
        learner.probe_active_keys = pair
        for mode in range(2):
            learner.goal_organ.current_mode = mode
            winner = learner._selected_route_key()
            choice = pair.index(winner) if winner in pair else -1
            module_selection.append(choice)
            correct += int(choice == (module + mode) % 2)
        selected.append(module_selection)
    return correct / (2 * len(route_pairs)), selected


def run_probe(
    *,
    modules: int = 16,
    training_cycles: int = 20,
    delay_steps: int = 8,
    goal_gain: float = 4.0,
    goal_permission_trials: int = 0,
) -> dict[str, object]:
    if modules < 1 or training_cycles < 1 or delay_steps < 0:
        raise ValueError("invalid probe dimensions")
    learner = _ProbeSelector(
        goal_apical_weight=goal_gain,
        maximum_goal_route_bonus=0.5,
        goal_route_eligibility_decay=0.9,
        goal_permission_trial_limit=goal_permission_trials,
    )
    route_pairs = []
    for module in range(modules):
        pair = ((1, 2 * module), (1, 2 * module + 1))
        route_pairs.append(pair)
        for choice, key in enumerate(pair):
            learner.route_counts[key] = {2 * module + choice: 8}
            learner.route_totals[key] = 8
            learner.route_utility[key] = 1.0
    for _ in range(training_cycles):
        for module, pair in enumerate(route_pairs):
            for mode in range(2):
                desired = (module + mode) % 2
                for choice, key in enumerate(pair):
                    signal = 1.0 if choice == desired else -1.0
                    _teach_delayed_restoration(learner, key, mode, signal, delay_steps)
    learner.goal_route_eligibility.clear()
    for _ in range(training_cycles):
        for module, pair in enumerate(route_pairs):
            for mode in range(2):
                desired = (module + mode) % 2
                if goal_permission_trials:
                    learner.probe_active_keys = pair
                    learner.goal_organ.current_mode = mode
                    learner._goal_learning_active = True
                    selected = learner._selected_route_key()
                    signal = 1.0 if selected == pair[desired] else -1.0
                    outcome = [0.0, 0.0, 0.0]
                    outcome[mode] = signal
                    learner._advance_goal_route_credit((), tuple(outcome), learn=True)
                else:
                    learner._update_goal_selection_utility(
                        pair, 2 * module + desired, mode
                    )
    learner._goal_learning_active = False
    accuracy, selected = _accuracy(learner, route_pairs)
    learner.goal_apical_weight = 0.0
    control_accuracy, control_selected = _accuracy(learner, route_pairs)
    changed = np.asarray(selected)[:, 0] != np.asarray(selected)[:, 1]
    return {
        "task": "modality_neutral_delayed_restoration_route_selection",
        "modules": modules,
        "training_cycles": training_cycles,
        "delay_steps": delay_steps,
        "goal_gain": goal_gain,
        "goal_permission_trials": goal_permission_trials,
        "goal_conditioned_accuracy": accuracy,
        "zero_goal_control_accuracy": control_accuracy,
        "goal_swap_changed_route_fraction": float(np.mean(changed)),
        "selected_routes": selected,
        "zero_goal_selected_routes": control_selected,
        "restoration_credit_paths": len(learner.goal_route_restoration_sum),
        "goal_selection_paths": len(learner.goal_selection_utility),
        "positive_goal_selection_paths": sum(
            utility > learner.minimum_goal_selection_utility
            for utility in learner.goal_selection_utility.values()
        ),
        "goal_selection_evidence": sum(learner.goal_selection_evidence.values()),
        "goal_route_switches": learner.goal_route_switches,
        "goal_route_context_calls": learner.goal_route_context_calls,
        "goal_route_proposals": learner.goal_route_proposals,
        "goal_route_denials": learner.goal_route_denials,
        "goal_permission_paths": len(learner.goal_permission_restoration_sum),
        "positive_goal_permission_paths": sum(
            restoration > 0.0
            for restoration in learner.goal_permission_restoration_sum.values()
        ),
        "goal_permission_evidence": sum(
            learner.goal_permission_restoration_evidence.values()
        ),
        "embodied_bytes": learner.embodied_bytes,
        "transient_bytes": learner.goal_transient_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=int, default=16)
    parser.add_argument("--training-cycles", type=int, default=20)
    parser.add_argument("--delay-steps", type=int, default=8)
    parser.add_argument("--goal-gain", type=float, default=4.0)
    parser.add_argument("--goal-permission-trials", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = run_probe(
        modules=args.modules,
        training_cycles=args.training_cycles,
        delay_steps=args.delay_steps,
        goal_gain=args.goal_gain,
        goal_permission_trials=args.goal_permission_trials,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
