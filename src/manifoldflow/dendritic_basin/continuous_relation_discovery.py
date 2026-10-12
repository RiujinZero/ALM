"""Phase 7 (design doc section 54: "Dense continuous vectors -- Test
arbitrary continuous streams. No sparse categorical assumptions.").

`relation_discovery.py` builds its diary from one-hot token indicators
and gates candidates on "anchor token active now" (`token_id == anchor`)
-- a categorical assumption. Nothing about the underlying mechanism
(`InteractionDendrite`, `RelationUnit`, `MultiTimescaleTraces`) actually
requires that: `InteractionDendrite.respond(a, b)` and
`MultiTimescaleTraces.update(x)` already accept arbitrary real values.
This module removes the one place that *did* assume categorical
structure: instead of "anchor channel equals this token, right now",
`A` is simply the current real value of a chosen `current_channel`, and
`B` is any `(trace_channel, timescale_index)` trace value -- a genuine
two-continuous-input interaction, exactly the design's "same relational
core" applied to a dense vector stream (section 57).

The usefulness score also generalizes: `relation_discovery`'s
`relation_information_gain` measures reduction in next-*token* entropy
(discrete). Here, `continuous_information_gain` measures reduction in
variance of the next continuous *value* of a chosen `target_channel` --
the same "be more certain about what comes next when this relation
fires" idea, just with variance standing in for entropy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from manifoldflow.dendritic_basin.relation_unit import RelationUnit
from manifoldflow.dendritic_basin.temporal_trace import (
    DEFAULT_TIMESCALES,
    MultiTimescaleTraces,
)

ContinuousDiary = list[tuple[np.ndarray, np.ndarray, np.ndarray]]


def build_continuous_diary(
    vector_stream: np.ndarray,
    timescales: tuple[int, ...] = DEFAULT_TIMESCALES,
    boundary_channel: int | None = None,
) -> ContinuousDiary:
    """One pass over `vector_stream` (shape `(T, n_channels)`), recording
    `(x_t, trace_snapshot_t, x_{t+1})` at every position -- the continuous
    analogue of `relation_discovery.build_stream_diary`, with no one-hot
    encoding step: `x_t` is used exactly as it arrives.

    `boundary_channel`, if given, marks a dedicated episode-boundary
    channel (active/high exactly on placeholder "reset" rows). Diary
    entries whose `x_{t+1}` is itself a boundary row are skipped: a
    boundary row's value is a fixed structural placeholder, not a real
    sensory outcome, so asking a relation to "predict" it produces a
    spurious, deterministic variance reduction that doesn't reflect any
    genuine signal in the stream (found while validating Phase 7 -- see
    the implementation journal: every episode transition was injecting
    the same fixed placeholder value into the population used to score
    `continuous_information_gain`, and a relation that happened to fire
    near episode boundaries could "explain" that structural artifact
    instead of the oscillator's actual dynamics).
    """
    n_channels = vector_stream.shape[1]
    traces = MultiTimescaleTraces(n_channels, timescales)
    diary: ContinuousDiary = []
    for t in range(len(vector_stream) - 1):
        x = vector_stream[t]
        traces.update(x)
        x_next = vector_stream[t + 1]
        if boundary_channel is not None and x_next[boundary_channel] > 0.5:
            continue
        diary.append((x.copy(), traces.q.copy(), x_next.copy()))
    return diary


def _make_continuous_draw_batch(
    diary: ContinuousDiary,
    current_channel: int,
    trace_channel: int,
    timescale_index: int,
) -> Callable[[int, np.random.Generator], list[tuple[float, float]]]:
    positions = np.arange(len(diary))

    def draw_batch(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
        idx = rng.choice(positions, size=n, replace=True)
        samples = []
        for i in idx:
            x, q, _x_next = diary[int(i)]
            a = float(x[current_channel])
            b = float(q[timescale_index, trace_channel])
            samples.append((a, b))
        return samples

    return draw_batch


def continuous_information_gain(
    unit: RelationUnit,
    current_channel: int,
    key: tuple[int, int],
    diary: ContinuousDiary,
    target_channel: int,
    baseline_channel: int | None = None,
    anchor_active_threshold: float | None = None,
) -> float:
    """Variance reduction (continuous analogue of
    `relation_discovery.relation_information_gain`'s entropy reduction):
    across every position in the diary, how much less variable is the
    next value of `target_channel` when this relation fires (activation
    > 0.5) versus the unconditional spread.

    `anchor_active_threshold`, if given, restricts *both* populations
    (`all_values` and `with_values`) to diary positions where
    `x[current_channel] > anchor_active_threshold` -- the continuous
    counterpart of `relation_discovery.relation_information_gain`'s
    `token_id != anchor: continue` filter. Without this, on a stream
    where `current_channel` is only meaningfully "active" at a small
    fraction of positions (e.g. Phase 10's ambiguous audio tone, active
    at 1 position in 5 per episode), `target_next` at every *other*
    position dilutes the population with values the relation was never
    trying to predict in the first place -- a background of mostly-
    constant, unrelated values that any dendrite can trivially "explain"
    by matching the background's own dominant value, without discovering
    anything about the one position that actually poses the ambiguity.
    Found while validating Phase 10: every discovered relation converged
    to predicting the same (wrong-half-the-time) background constant,
    each scoring real but shallow gain from the 4-in-5 irrelevant
    positions per episode, and blending them never recovered the other
    class at all. This is the same lesson `assembly_discovery.py`'s
    anchor-conditioned sampling already learned for discrete assemblies
    (see the implementation journal's Phase 3 entry) -- it turns out to
    apply here too, and `continuous_information_gain` was the one place
    in this codebase that hadn't yet needed it (Phase 7's oscillator task
    has no "irrelevant" positions, so the gap was invisible until a task
    with a genuinely sparse region of interest came along).

    `baseline_channel`, if given, scores gain on the *residual*
    `target_next - x[baseline_channel]` instead of the raw target value
    (typically `baseline_channel == current_channel`, i.e. a persistence
    baseline: "next value equals current value"). Without this, a
    relation whose dendrite leans mostly on its `A` input (the current
    value) -- which the acceptance gate does not forbid, it only forbids
    leaning *entirely* on one parent -- can score a large apparent gain
    that is really just the stream's own trivial autocorrelation, already
    captured by persistence, dressed up as if the `B` trace had
    discovered something new (found while validating Phase 7 on the
    continuous oscillator task: every top-ranked relation reduced raw
    target variance on both train *and* held-out data -- so it wasn't
    overfitting -- yet blending them into a persistence baseline made
    held-out MSE slightly worse, because the "new" information was mostly
    a redundant, noisier restatement of what persistence already
    captured. This is the design's own "spurious deterministic
    confidence" failure mode, section 53, just via redundancy with a
    trivial baseline rather than pure noise). Scoring the residual
    credits a relation only for variance persistence does *not* already
    explain.
    """
    trace_channel, timescale_index = key
    with_values: list[float] = []
    all_values: list[float] = []
    for x, q, x_next in diary:
        if anchor_active_threshold is not None and float(x[current_channel]) <= (
            anchor_active_threshold
        ):
            continue
        a = float(x[current_channel])
        b = float(q[timescale_index, trace_channel])
        activation = unit.respond(a, b)
        target_next = float(x_next[target_channel])
        if baseline_channel is not None:
            target_next -= float(x[baseline_channel])
        all_values.append(target_next)
        if activation > 0.5:
            with_values.append(target_next)

    if not with_values:
        return 0.0
    base_var = float(np.var(all_values))
    with_var = float(np.var(with_values))
    return base_var - with_var


def discover_and_rank_continuous_relations(
    diary: ContinuousDiary,
    current_channel: int,
    target_channel: int,
    n_channels: int,
    n_timescales: int,
    seed: int,
    min_gain: float = 0.01,
    min_std: float = 0.05,
    baseline_channel: int | None = None,
    anchor_active_threshold: float | None = None,
    acceptance_threshold: float = 0.05,
    n_candidates: int = 2,
    rounds: int = 40,
    perturbations_per_round: int = 12,
    step_size: float = 0.3,
    episodes_per_candidate: int = 1,
) -> list[tuple[tuple[int, int], RelationUnit, float]]:
    """Scan every `(trace_channel, timescale_index)` pair as the `B` input
    against the fixed `current_channel` as `A`, train a `RelationUnit` on
    each (same growing-replay-buffer / retry-episodes machinery as
    `relation_discovery.discover_candidate_relations`), keep only units
    that clear the structural gate (`is_accepted`), then rank by
    `continuous_information_gain` and keep those at or above `min_gain`.

    `min_std` skips any `(trace_channel, timescale_index)` whose trace is
    nearly constant across the whole diary -- the continuous analogue of
    `relation_discovery`'s coefficient-of-variation guard. An *absolute*
    std floor is used here rather than a coefficient of variation because
    continuous channels (e.g. `sin`/`cos`) naturally oscillate around a
    mean near zero, where std-relative-to-mean blows up spuriously even
    for channels that vary plenty in absolute terms.

    `baseline_channel` and `anchor_active_threshold` are forwarded to
    `continuous_information_gain` -- see there for why scoring against a
    baseline residual, and restricting scoring to positions where
    `current_channel` is actually "active", each matter.
    """
    rng = np.random.default_rng(seed)
    uid = 0
    scored: list[tuple[tuple[int, int], RelationUnit, float]] = []

    for trace_channel in range(n_channels):
        for timescale_index in range(n_timescales):
            channel_series = np.array(
                [q[timescale_index, trace_channel] for _x, q, _x_next in diary]
            )
            if float(np.std(channel_series)) < min_std:
                uid += episodes_per_candidate
                continue

            draw_batch = _make_continuous_draw_batch(
                diary, current_channel, trace_channel, timescale_index
            )
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
                gain = continuous_information_gain(
                    unit,
                    current_channel,
                    key,
                    diary,
                    target_channel,
                    baseline_channel,
                    anchor_active_threshold,
                )
                if gain > best_gain:
                    best_gain = gain
                    best_unit = unit
            if best_unit is not None and best_gain >= min_gain:
                scored.append((key, best_unit, best_gain))

    scored.sort(key=lambda item: item[2], reverse=True)
    return scored
