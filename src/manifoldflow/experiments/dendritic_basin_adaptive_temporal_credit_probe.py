"""Let local dendritic trace timescales compete under overlapping causes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.delayed_credit import (
    CompetingEligibilityTraceLearner,
)
from manifoldflow.dendritic_basin.tasks.delayed_credit import (
    OverlappingCausalStream,
    OverlappingDelayedCausalTask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receivers", type=int, default=8)
    parser.add_argument("--causal-delay", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--eval-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _bits(probability: np.ndarray) -> float:
    return float(np.mean(-np.log2(np.maximum(probability, 1e-300))))


def _train(
    learner: CompetingEligibilityTraceLearner, stream: OverlappingCausalStream
) -> None:
    for index, outcome in enumerate(stream.outcomes):
        learner.observe_sources(stream.basal_states[index])
        if outcome >= 0:
            learner.observe_consequence(
                int(stream.apical_receivers[index]), int(outcome), learn=True
            )


def _evaluate(
    learner: CompetingEligibilityTraceLearner, stream: OverlappingCausalStream
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_count = len(stream.outcomes) - stream.causal_delay
    selected_probability = np.empty(valid_count, dtype=np.float64)
    selected_prediction = np.empty(valid_count, dtype=np.uint8)
    channel_probability = np.empty(
        (valid_count, len(learner.trace_decays)), dtype=np.float64
    )
    cursor = 0
    for index, outcome in enumerate(stream.outcomes):
        learner.observe_sources(stream.basal_states[index])
        if outcome < 0:
            continue
        receiver = int(stream.apical_receivers[index])
        outcome = int(outcome)
        for channel in range(len(learner.trace_decays)):
            probability_one = learner.probability_for_channel(receiver, channel)
            channel_probability[cursor, channel] = (
                probability_one if outcome else 1.0 - probability_one
            )
        step = learner.observe_consequence(receiver, outcome, learn=False)
        selected_probability[cursor] = step.observed_probability
        selected_prediction[cursor] = step.predicted_outcome
        cursor += 1
    return selected_probability, selected_prediction, channel_probability


def run_probe(
    *,
    receiver_count: int = 8,
    causal_delay: int = 8,
    train_steps: int = 50_000,
    eval_steps: int = 20_000,
    seed: int = 0,
) -> dict[str, object]:
    task = OverlappingDelayedCausalTask(
        receiver_count, causal_delay=causal_delay, seed=seed
    )
    train = task.sample(train_steps)
    evaluation = task.sample(eval_steps)
    learner = CompetingEligibilityTraceLearner(receiver_count)
    _train(learner, train)
    learner.reset_traces()
    probability, prediction, channel_probability = _evaluate(learner, evaluation)
    valid_outcomes = evaluation.outcomes[causal_delay:]
    per_decay = [
        {
            "decay": float(decay),
            "eval_bits_per_outcome": _bits(channel_probability[:, channel]),
        }
        for channel, decay in enumerate(learner.trace_decays)
    ]
    selected_channels = np.asarray(
        [learner.selected_channel(receiver) for receiver in range(receiver_count)]
    )
    return {
        "task": "overlapping_causes_adaptive_local_trace_bank",
        "receiver_count": receiver_count,
        "causal_delay": causal_delay,
        "train_steps": train_steps,
        "eval_steps": eval_steps,
        "selected_eval_bits_per_outcome": _bits(probability),
        "selected_eval_accuracy": float(np.mean(prediction == valid_outcomes)),
        "per_decay": per_decay,
        "selected_decays": [
            float(learner.trace_decays[channel]) for channel in selected_channels
        ],
        "local_updates": learner.local_updates,
        "embodied_bytes": learner.embodied_bytes,
        "peak_transient_bytes": learner.transient_bytes,
    }


def main() -> None:
    args = parse_args()
    report = run_probe(
        receiver_count=args.receivers,
        causal_delay=args.causal_delay,
        train_steps=args.train_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
