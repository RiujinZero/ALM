"""Automatic temporal-relation discovery over a token/event stream
(design doc sections 14-18, 28 of `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`;
Smoke 1's "automatic structural discovery" in
`Dendritic_Basin_Learning_Design_v0.3_Language_Smoke_Findings.md`).

Rather than manually pairing e.g. "alice" with "sees", this scans all
(anchor channel active now, trace of some other channel) candidate pairs
and trains a real `RelationUnit` for each; useful relations both clear
the quality gate (are a genuine interaction, not noise) *and* improve
next-token prediction (utility, scored separately -- section 24's
distinction between activation/utility/depth).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from manifoldflow.dendritic_basin.relation_unit import RelationUnit
from manifoldflow.dendritic_basin.temporal_trace import (
    DEFAULT_TIMESCALES,
    MultiTimescaleTraces,
)


def build_stream_diary(
    token_stream: list[int],
    vocab_size: int,
    timescales: tuple[int, ...] = DEFAULT_TIMESCALES,
) -> list[tuple[int, np.ndarray, int]]:
    """One pass over `token_stream`, recording (token_id_t, trace_snapshot_t,
    token_id_{t+1}) at every position. `trace_snapshot_t` is the full
    (n_timescales, vocab_size) trace state *after* incorporating token_t --
    every candidate relation below reads directly from this diary instead
    of replaying the stream itself.
    """
    traces = MultiTimescaleTraces(vocab_size, timescales)
    diary: list[tuple[int, np.ndarray, int]] = []
    for t in range(len(token_stream) - 1):
        token_id = token_stream[t]
        x = np.zeros(vocab_size)
        x[token_id] = 1.0
        traces.update(x)
        diary.append((token_id, traces.q.copy(), token_stream[t + 1]))
    return diary


def _make_draw_batch(
    diary: list[tuple[int, np.ndarray, int]],
    anchor: int,
    trace_channel: int,
    timescale_index: int,
) -> Callable[[int, np.random.Generator], list[tuple[float, float]]]:
    positions = np.arange(len(diary))

    def draw_batch(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
        idx = rng.choice(positions, size=n, replace=True)
        samples = []
        for i in idx:
            token_id, q, _next_token = diary[int(i)]
            a = 1.0 if token_id == anchor else 0.0
            b = float(q[timescale_index, trace_channel])
            samples.append((a, b))
        return samples

    return draw_batch


def discover_candidate_relations(
    diary: list[tuple[int, np.ndarray, int]],
    anchor: int,
    vocab_size: int,
    n_timescales: int,
    seed: int,
    n_candidates: int = 2,
    acceptance_threshold: float = 0.05,
    rounds: int = 40,
    perturbations_per_round: int = 12,
    step_size: float = 0.3,
    episodes_per_candidate: int = 1,
    min_coefficient_of_variation: float = 0.1,
) -> dict[tuple[int, int], RelationUnit]:
    """Train one RelationUnit per (trace_channel, timescale_index) paired
    with the fixed `anchor` channel (trace_channel != anchor), from real
    stream positions in `diary`. Returns only the ones that clear
    `acceptance_threshold` -- a genuine-interaction gate, not a
    usefulness gate (section 24: activation/utility must not be
    conflated). Continuous, noisy trace values reach a much lower ceiling
    on J than clean balanced-binary parents do (0.10-0.16 here vs. XOR's
    0.25), and -- more importantly -- J alone does *not* separate
    genuinely predictive relations from spurious ones at this noise level
    (a same-sentence, causally-irrelevant pair can score comparably on J):
    see `relation_information_gain` for the separate score that does.
    Callers that want *useful* relations, not just structurally valid
    ones, should rank this function's output by `relation_information_gain`
    (see `discover_and_rank_relations`).

    `episodes_per_candidate` > 1 retries each candidate with fresh
    dendrites, keeping whichever accepted episode scores the highest
    predictive information gain -- not every single episode converges as
    cleanly as the best one does (the same finding as Phase 0/1's XOR and
    parity validation), so more than one attempt per candidate matters
    here too.

    `min_coefficient_of_variation` skips any (trace_channel, timescale)
    whose trace value is nearly constant (std/mean below this ratio)
    *specifically at positions where `anchor` is active* -- the only
    positions that matter for prediction. A channel whose timing relative
    to `anchor` is fixed by the stream's own structure (e.g. a token that
    always occurs a constant number of steps before every occurrence of
    `anchor`, so its trace always reaches nearly the same value exactly
    when `anchor` fires) cannot discriminate anything there by
    construction, even though its *absolute* variance can still look large
    at longer timescales (a raw-variance threshold does not catch this --
    an earlier version of this guard used one and still let it through;
    coefficient of variation, relative to the channel's own mean level at
    these positions, does). Without this guard, a dendrite can latch onto
    minute floating-point-level differences at those positions and
    produce a spuriously high information gain -- exactly the kind of
    accidental-correlation failure mode this project keeps running into
    (see the implementation journal).
    """
    rng = np.random.default_rng(seed)
    accepted: dict[tuple[int, int], RelationUnit] = {}
    uid = 0
    anchor_values = np.array([q for tok, q, _next in diary if tok == anchor])
    for trace_channel in range(vocab_size):
        if trace_channel == anchor:
            continue
        for timescale_index in range(n_timescales):
            channel_series = anchor_values[:, timescale_index, trace_channel]
            mean = float(np.mean(channel_series))
            std = float(np.std(channel_series))
            coefficient_of_variation = std / (abs(mean) + 1e-8)
            if coefficient_of_variation < min_coefficient_of_variation:
                uid += episodes_per_candidate
                continue
            draw_batch = _make_draw_batch(diary, anchor, trace_channel, timescale_index)
            key = (trace_channel, timescale_index)
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
            if best_unit is not None:
                accepted[key] = best_unit
    return accepted


def discover_and_rank_relations(
    diary: list[tuple[int, np.ndarray, int]],
    anchor: int,
    vocab_size: int,
    n_timescales: int,
    seed: int,
    min_gain: float = 0.1,
    **discover_kwargs,
) -> list[tuple[tuple[int, int], RelationUnit, float]]:
    """`discover_candidate_relations` (structural validity) followed by
    ranking the survivors by `relation_information_gain` (usefulness),
    keeping only those above `min_gain`. Returns a list of
    (key, unit, gain) sorted by descending gain -- this is the actual
    "automatic structural discovery" pipeline (section 28's "ranked using
    locally measured predictive improvement"), not J-thresholding alone.
    """
    accepted = discover_candidate_relations(
        diary, anchor, vocab_size, n_timescales, seed, **discover_kwargs
    )
    scored = [
        (key, unit, relation_information_gain(unit, key, diary, anchor, vocab_size))
        for key, unit in accepted.items()
    ]
    useful = [(key, unit, gain) for key, unit, gain in scored if gain >= min_gain]
    useful.sort(key=lambda item: item[2], reverse=True)
    return useful


def relation_information_gain(
    unit: RelationUnit,
    key: tuple[int, int],
    diary: list[tuple[int, np.ndarray, int]],
    anchor: int,
    vocab_size: int,
) -> float:
    """A simple predictive-improvement score (section 28's "ranked using
    locally measured predictive improvement"): among diary positions
    where the anchor is active, how much more peaked is the next-token
    distribution when this relation also fires (activation > 0.5) versus
    when it doesn't -- measured as the reduction in entropy (in bits).
    """
    trace_channel, timescale_index = key
    with_relation_counts = np.zeros(vocab_size)
    without_relation_counts = np.zeros(vocab_size)
    for token_id, q, next_token in diary:
        if token_id != anchor:
            continue
        a = 1.0
        b = float(q[timescale_index, trace_channel])
        activation = unit.respond(a, b)
        if activation > 0.5:
            with_relation_counts[next_token] += 1
        else:
            without_relation_counts[next_token] += 1

    def entropy(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    base_counts = with_relation_counts + without_relation_counts
    base_entropy = entropy(base_counts)
    if with_relation_counts.sum() == 0:
        return 0.0
    return base_entropy - entropy(with_relation_counts)
