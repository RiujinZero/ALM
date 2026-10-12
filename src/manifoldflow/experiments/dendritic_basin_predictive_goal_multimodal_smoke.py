"""Fast synthetic ablation of persistent, cross-modal predictive goals.

The task deliberately separates three things that ordinary next-event prediction can
confound:

* the recent context is identical before every scored consequence;
* a useful cue occurs in another modality ``horizon`` events earlier; and
* visually identical pseudo-cues occur in between, so merely averaging history is
  not a lossless goal register.

The learners receive integer events only.  They do not receive modality IDs, target
positions, episode goals, or cue/consequence pairings.  Their updates are counts or
Hebbian prototype updates on events that actually occurred.  Modality names are
retained only by the generator so the benchmark can verify the cross-modal setup.

This is a mechanism probe, not a language-quality claim.  In particular, the
lagged-successor condition tests whether a self-supervised forecast can be stored
causally across a long interval and used to rank an otherwise ambiguous outcome.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CUE_MODALITY = 0
CONTEXT_MODALITY = 1
CONSEQUENCE_MODALITY = 2
MODALITY_NAMES = ("cue_sensor", "intervening_context", "consequence_actuator")


@dataclass(frozen=True)
class Vocabulary:
    """Disjoint event ranges used only to construct and score the task."""

    cue_symbols: tuple[int, ...]
    distractor_symbols: tuple[int, ...]
    query_symbol: int
    consequence_symbols: tuple[int, ...]
    size: int


@dataclass(frozen=True)
class MultimodalEpisode:
    """An observable event stream plus audit-only modality annotations."""

    events: np.ndarray
    modalities: np.ndarray


def make_vocabulary(*, goals: int = 4, distractors: int = 12) -> Vocabulary:
    """Create a compact vocabulary with disjoint cue and consequence alphabets."""
    if goals < 2:
        raise ValueError("goals must be at least two")
    if distractors < 2:
        raise ValueError("distractors must be at least two")
    cue_symbols = tuple(range(goals))
    distractor_symbols = tuple(range(goals, goals + distractors))
    query_symbol = goals + distractors
    consequence_symbols = tuple(range(query_symbol + 1, query_symbol + 1 + goals))
    return Vocabulary(
        cue_symbols=cue_symbols,
        distractor_symbols=distractor_symbols,
        query_symbol=query_symbol,
        consequence_symbols=consequence_symbols,
        size=query_symbol + 1 + goals,
    )


def generate_multimodal_episodes(
    n_episodes: int,
    *,
    horizon: int,
    vocabulary: Vocabulary,
    rng: np.random.Generator,
    local_window: int = 4,
    pseudo_cue_probability: float = 0.25,
) -> list[MultimodalEpisode]:
    """Generate balanced delayed-consequence episodes without exposing goal labels.

    The consequence is exactly ``horizon`` events after the informative cue.  Goals
    are balanced before their order is randomized, making chance accuracy and local
    conditional entropy exact when ``n_episodes`` is a multiple of the goal count.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if local_window <= 0:
        raise ValueError("local_window must be positive")
    if horizon <= local_window:
        raise ValueError("horizon must exceed local_window")
    if not 0.0 <= pseudo_cue_probability <= 1.0:
        raise ValueError("pseudo_cue_probability must be in [0, 1]")

    goal_count = len(vocabulary.cue_symbols)
    latent_order = np.resize(np.arange(goal_count, dtype=np.int64), n_episodes)
    rng.shuffle(latent_order)
    episodes: list[MultimodalEpisode] = []
    distractors = np.asarray(vocabulary.distractor_symbols, dtype=np.int64)
    cues = np.asarray(vocabulary.cue_symbols, dtype=np.int64)

    for latent in latent_order:
        events = np.empty(horizon + 1, dtype=np.int64)
        modalities = np.full(horizon + 1, CONTEXT_MODALITY, dtype=np.int8)
        events[0] = vocabulary.cue_symbols[int(latent)]
        modalities[0] = CUE_MODALITY

        filler_end = horizon - local_window
        for index in range(1, filler_end):
            if rng.random() < pseudo_cue_probability:
                events[index] = int(rng.choice(cues))
                modalities[index] = CUE_MODALITY
            else:
                # A sticky, locally plausible nuisance process.  Its dynamics and
                # symbols are independent of the latent episode goal.
                if index > 1 and rng.random() < 0.72:
                    previous = int(events[index - 1])
                    if previous in vocabulary.distractor_symbols:
                        events[index] = previous
                        continue
                events[index] = int(rng.choice(distractors))

        events[filler_end:horizon] = vocabulary.query_symbol
        events[horizon] = vocabulary.consequence_symbols[int(latent)]
        modalities[horizon] = CONSEQUENCE_MODALITY
        episodes.append(MultimodalEpisode(events=events, modalities=modalities))
    return episodes


def stratified_shuffle_consequences(
    episodes: Sequence[MultimodalEpisode],
    *,
    consequence_symbols: Sequence[int],
    rng: np.random.Generator,
) -> list[MultimodalEpisode]:
    """Destroy cue/consequence dependence while retaining near-exact marginals.

    Assignment is balanced separately for every initial observable symbol.  This is
    a lower-variance shuffled control than a single unconstrained permutation and
    uses no latent goal annotations.
    """
    symbols = np.asarray(tuple(consequence_symbols), dtype=np.int64)
    if not len(symbols):
        raise ValueError("consequence_symbols must not be empty")
    grouped: dict[int, list[int]] = {}
    for index, episode in enumerate(episodes):
        grouped.setdefault(int(episode.events[0]), []).append(index)

    replacements = np.empty(len(episodes), dtype=np.int64)
    for indices in grouped.values():
        balanced = np.resize(symbols, len(indices)).copy()
        rng.shuffle(balanced)
        for episode_index, replacement in zip(indices, balanced, strict=True):
            replacements[episode_index] = replacement

    shuffled: list[MultimodalEpisode] = []
    for episode, replacement in zip(episodes, replacements, strict=True):
        events = episode.events.copy()
        events[-1] = replacement
        shuffled.append(
            MultimodalEpisode(events=events, modalities=episode.modalities.copy())
        )
    return shuffled


def _restricted_probabilities(
    counts: np.ndarray,
    candidates: np.ndarray,
    *,
    smoothing: float,
) -> np.ndarray:
    selected = np.asarray(counts[candidates], dtype=np.float64) + smoothing
    return selected / float(np.sum(selected))


class ImmediateContextPredictor:
    """A conventional finite-context event predictor with local count updates."""

    def __init__(self, vocabulary_size: int, *, window: int = 4) -> None:
        self.vocabulary_size = int(vocabulary_size)
        self.window = int(window)
        self.counts: dict[tuple[int, ...], np.ndarray] = {}
        self.local_updates = 0

    def fit(self, episodes: Iterable[MultimodalEpisode]) -> None:
        for episode in episodes:
            events = episode.events
            for index in range(self.window, len(events)):
                context = tuple(
                    int(value) for value in events[index - self.window : index]
                )
                row = self.counts.setdefault(
                    context, np.zeros(self.vocabulary_size, dtype=np.float64)
                )
                row[int(events[index])] += 1.0
                self.local_updates += 1

    def probabilities(
        self, prefix: np.ndarray, candidates: Sequence[int]
    ) -> np.ndarray:
        candidate_array = np.asarray(tuple(candidates), dtype=np.int64)
        if len(prefix) < self.window:
            return np.full(len(candidate_array), 1.0 / len(candidate_array))
        context = tuple(int(value) for value in prefix[-self.window :])
        row = self.counts.get(context)
        if row is None:
            return np.full(len(candidate_array), 1.0 / len(candidate_array))
        return _restricted_probabilities(row, candidate_array, smoothing=1.0)


class SlowThemePredictor:
    """Hebbian readout from an exponentially slow event-frequency signature."""

    def __init__(
        self,
        vocabulary_size: int,
        *,
        half_life: float = 32.0,
        temperature: float = 0.02,
    ) -> None:
        if half_life <= 0.0:
            raise ValueError("half_life must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.vocabulary_size = int(vocabulary_size)
        self.decay = math.exp(math.log(0.5) / half_life)
        self.temperature = float(temperature)
        self.prototype_sums = np.zeros(
            (self.vocabulary_size, self.vocabulary_size), dtype=np.float64
        )
        self.prototype_counts = np.zeros(self.vocabulary_size, dtype=np.int64)
        self.local_updates = 0

    def _advance(self, state: np.ndarray, event: int) -> None:
        state *= self.decay
        state[event] += 1.0 - self.decay

    def _signature(self, events: np.ndarray) -> np.ndarray:
        state = np.zeros(self.vocabulary_size, dtype=np.float64)
        for event in events:
            self._advance(state, int(event))
        total = float(np.sum(state))
        return state / total if total > 0.0 else state

    def fit(self, episodes: Iterable[MultimodalEpisode]) -> None:
        for episode in episodes:
            state = np.zeros(self.vocabulary_size, dtype=np.float64)
            for event_value in episode.events:
                event = int(event_value)
                total = float(np.sum(state))
                if total > 0.0:
                    self.prototype_sums[:, event] += state / total
                    self.prototype_counts[event] += 1
                    self.local_updates += self.vocabulary_size
                self._advance(state, event)

    def probabilities(
        self, prefix: np.ndarray, candidates: Sequence[int]
    ) -> np.ndarray:
        candidate_array = np.asarray(tuple(candidates), dtype=np.int64)
        signature = self._signature(prefix)
        prototypes = np.zeros((len(candidate_array), self.vocabulary_size))
        observed = np.zeros(len(candidate_array), dtype=np.bool_)
        for row_index, candidate in enumerate(candidate_array):
            count = int(self.prototype_counts[candidate])
            if count:
                prototype = self.prototype_sums[:, candidate] / count
                total = float(np.sum(prototype))
                prototypes[row_index] = prototype / total if total else prototype
                observed[row_index] = True
        if not np.any(observed):
            return np.full(len(candidate_array), 1.0 / len(candidate_array))
        distances = np.sum((prototypes - signature[None, :]) ** 2, axis=1)
        logits = -distances / self.temperature
        logits -= float(np.max(logits))
        probability = np.exp(logits)
        return probability / float(np.sum(probability))


class LaggedSuccessorPredictor:
    """Learn and causally schedule an observation ``lag`` events downstream.

    At inference every event schedules its learned successor distribution in a
    fixed-length queue.  The queue is the persistent representation: clearing it
    after the cue but before the consequence removes the effect without changing
    any learned weights.
    """

    def __init__(self, vocabulary_size: int, *, lag: int) -> None:
        if lag <= 0:
            raise ValueError("lag must be positive")
        self.vocabulary_size = int(vocabulary_size)
        self.lag = int(lag)
        self.counts = np.zeros(
            (self.vocabulary_size, self.vocabulary_size), dtype=np.float64
        )
        self.local_updates = 0

    def fit(self, episodes: Iterable[MultimodalEpisode]) -> None:
        for episode in episodes:
            events = episode.events
            if len(events) <= self.lag:
                continue
            sources = events[: -self.lag]
            targets = events[self.lag :]
            np.add.at(self.counts, (sources, targets), 1.0)
            self.local_updates += len(sources)

    def probabilities(
        self,
        prefix: np.ndarray,
        candidates: Sequence[int],
        *,
        reset_index: int | None = None,
    ) -> np.ndarray:
        candidate_array = np.asarray(tuple(candidates), dtype=np.int64)
        uniform = np.full(len(candidate_array), 1.0 / len(candidate_array))
        pending: deque[np.ndarray | None] = deque([None] * self.lag)
        for index, event_value in enumerate(prefix):
            if reset_index is not None and index == reset_index:
                pending = deque([None] * self.lag)
            pending.popleft()
            row = self.counts[int(event_value)]
            if float(np.sum(row[candidate_array])) == 0.0:
                pending.append(uniform)
            else:
                pending.append(
                    _restricted_probabilities(row, candidate_array, smoothing=1.0)
                )
        due = pending[0]
        return uniform if due is None else due


def _conditional_entropy_bits(
    episodes: Sequence[MultimodalEpisode], *, window: int
) -> float:
    grouped: dict[tuple[int, ...], dict[int, int]] = {}
    for episode in episodes:
        context = tuple(int(value) for value in episode.events[-window - 1 : -1])
        target = int(episode.events[-1])
        row = grouped.setdefault(context, {})
        row[target] = row.get(target, 0) + 1
    total = len(episodes)
    entropy = 0.0
    for row in grouped.values():
        counts = np.asarray(tuple(row.values()), dtype=np.float64)
        probability = counts / float(np.sum(counts))
        entropy += (
            float(np.sum(counts))
            / total
            * float(-np.sum(probability * np.log2(probability)))
        )
    return entropy


def _score(
    episodes: Sequence[MultimodalEpisode],
    predictor: ImmediateContextPredictor
    | SlowThemePredictor
    | LaggedSuccessorPredictor,
    *,
    candidates: Sequence[int],
    reset_successor: bool = False,
) -> dict[str, float | int]:
    candidate_array = np.asarray(tuple(candidates), dtype=np.int64)
    candidate_to_index = {
        int(candidate): index for index, candidate in enumerate(candidate_array)
    }
    true_probability: list[float] = []
    correct = 0
    for episode in episodes:
        prefix = episode.events[:-1]
        if isinstance(predictor, LaggedSuccessorPredictor):
            reset_index = len(prefix) // 2 if reset_successor else None
            probability = predictor.probabilities(
                prefix, candidate_array, reset_index=reset_index
            )
        else:
            probability = predictor.probabilities(prefix, candidate_array)
        target_index = candidate_to_index[int(episode.events[-1])]
        true_probability.append(float(probability[target_index]))
        correct += int(int(np.argmax(probability)) == target_index)
    values = np.asarray(true_probability, dtype=np.float64)
    return {
        "episodes": len(episodes),
        "accuracy": correct / len(episodes),
        "cross_entropy_bits": float(-np.mean(np.log2(np.maximum(values, 1e-300)))),
        "mean_true_probability": float(np.mean(values)),
        "local_updates": predictor.local_updates,
    }


def _run_horizon(
    *,
    horizon: int,
    train_episodes: int,
    eval_episodes: int,
    seed: int,
    goals: int,
    distractors: int,
    local_window: int,
    pseudo_cue_probability: float,
    theme_half_life: float,
) -> dict[str, object]:
    vocabulary = make_vocabulary(goals=goals, distractors=distractors)
    train = generate_multimodal_episodes(
        train_episodes,
        horizon=horizon,
        vocabulary=vocabulary,
        rng=np.random.default_rng(seed),
        local_window=local_window,
        pseudo_cue_probability=pseudo_cue_probability,
    )
    evaluation = generate_multimodal_episodes(
        eval_episodes,
        horizon=horizon,
        vocabulary=vocabulary,
        rng=np.random.default_rng(seed + 1),
        local_window=local_window,
        pseudo_cue_probability=pseudo_cue_probability,
    )
    shuffled = stratified_shuffle_consequences(
        train,
        consequence_symbols=vocabulary.consequence_symbols,
        rng=np.random.default_rng(seed + 2),
    )

    immediate = ImmediateContextPredictor(vocabulary.size, window=local_window)
    theme = SlowThemePredictor(vocabulary.size, half_life=theme_half_life)
    successor = LaggedSuccessorPredictor(vocabulary.size, lag=horizon)
    shuffled_successor = LaggedSuccessorPredictor(vocabulary.size, lag=horizon)
    immediate.fit(train)
    theme.fit(train)
    successor.fit(train)
    shuffled_successor.fit(shuffled)

    candidates = vocabulary.consequence_symbols
    conditions = {
        "local_immediate": _score(evaluation, immediate, candidates=candidates),
        "slow_theme": _score(evaluation, theme, candidates=candidates),
        "far_future_successor": _score(evaluation, successor, candidates=candidates),
        "shuffled_successor": _score(
            evaluation, shuffled_successor, candidates=candidates
        ),
        "mid_episode_reset_successor": _score(
            evaluation,
            successor,
            candidates=candidates,
            reset_successor=True,
        ),
    }
    local_bits = float(conditions["local_immediate"]["cross_entropy_bits"])
    for metrics in conditions.values():
        metrics["gain_over_local_bits"] = local_bits - float(
            metrics["cross_entropy_bits"]
        )

    return {
        "horizon": horizon,
        "cue_to_consequence_events": horizon,
        "final_local_context_conditional_entropy_bits": _conditional_entropy_bits(
            evaluation, window=local_window
        ),
        "chance_accuracy": 1.0 / goals,
        "chance_cross_entropy_bits": math.log2(goals),
        "conditions": conditions,
    }


def run_ablation(
    *,
    horizons: Sequence[int] = (8, 32, 128, 512),
    train_episodes: int = 800,
    eval_episodes: int = 320,
    seed: int = 73,
    goals: int = 4,
    distractors: int = 12,
    local_window: int = 4,
    pseudo_cue_probability: float = 0.25,
    theme_half_life: float = 32.0,
) -> dict[str, object]:
    """Run all mechanisms and causal controls at each requested horizon."""
    if not horizons:
        raise ValueError("at least one horizon is required")
    rows = [
        _run_horizon(
            horizon=int(horizon),
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            seed=seed + 10_007 * index,
            goals=goals,
            distractors=distractors,
            local_window=local_window,
            pseudo_cue_probability=pseudo_cue_probability,
            theme_half_life=theme_half_life,
        )
        for index, horizon in enumerate(horizons)
    ]
    successor_gains = [
        float(row["conditions"]["far_future_successor"]["gain_over_local_bits"])
        for row in rows
    ]
    return {
        "schema_version": 1,
        "task": "modality_neutral_delayed_consequence",
        "claim_scope": "synthetic mechanism ablation; not a language-quality result",
        "learner_observations": {
            "event_symbols": True,
            "candidate_consequence_vocabulary": True,
            "modality_ids": False,
            "target_positions": False,
            "latent_goal_labels": False,
            "hand_labeled_success": False,
        },
        "modalities": {
            "early_cue": MODALITY_NAMES[CUE_MODALITY],
            "intervening": MODALITY_NAMES[CONTEXT_MODALITY],
            "scored_consequence": MODALITY_NAMES[CONSEQUENCE_MODALITY],
        },
        "configuration": {
            "horizons": [int(value) for value in horizons],
            "train_episodes": train_episodes,
            "eval_episodes": eval_episodes,
            "goals": goals,
            "distractors": distractors,
            "local_window": local_window,
            "pseudo_cue_probability": pseudo_cue_probability,
            "theme_half_life": theme_half_life,
            "seed": seed,
        },
        "horizons": rows,
        "aggregate": {
            "minimum_successor_gain_over_local_bits": min(successor_gains),
            "mean_successor_gain_over_local_bits": float(np.mean(successor_gains)),
            "all_horizons_successor_better_than_local": all(
                gain > 0.0 for gain in successor_gains
            ),
        },
    }


def _parse_horizons(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "horizons must be comma-separated positive integers"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=_parse_horizons, default=(8, 32, 128, 512))
    parser.add_argument("--train-episodes", type=int, default=800)
    parser.add_argument("--eval-episodes", type=int, default=320)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--goals", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=12)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--pseudo-cue-probability", type=float, default=0.25)
    parser.add_argument("--theme-half-life", type=float, default=32.0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_ablation(
        horizons=args.horizons,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        goals=args.goals,
        distractors=args.distractors,
        local_window=args.local_window,
        pseudo_cue_probability=args.pseudo_cue_probability,
        theme_half_life=args.theme_half_life,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
