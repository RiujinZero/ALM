"""Tests for continuous (dense-vector) relation discovery (design doc
section 54, "Dense continuous vectors -- Test arbitrary continuous
streams. No sparse categorical assumptions.")."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.continuous_relation_discovery import (
    build_continuous_diary,
    continuous_information_gain,
    discover_and_rank_continuous_relations,
)


def _make_synthetic_channels_stream(n: int, rng: np.random.Generator) -> np.ndarray:
    """3 channels: channel 0 is the "current" signal; channel 1's trace at
    timescale index 1 is *literally* what channel 0 becomes two steps
    later (a strong, discoverable lagged relationship); channel 2 is pure
    noise, uncorrelated with anything.
    """
    channel_1 = rng.uniform(-1.0, 1.0, size=n)
    channel_0 = np.zeros(n)
    channel_0[2:] = channel_1[:-2]
    channel_0[:2] = rng.uniform(-1.0, 1.0, size=2)
    channel_2 = rng.uniform(-1.0, 1.0, size=n)
    return np.stack([channel_0, channel_1, channel_2], axis=1)


class _AlwaysOffUnit:
    def respond(self, a: float, b: float) -> float:
        return 0.0


class _FiresWhenBHighUnit:
    def respond(self, a: float, b: float) -> float:
        return 1.0 if b > 0.0 else 0.0


def test_continuous_information_gain_is_zero_for_a_relation_that_never_fires() -> None:
    rng = np.random.default_rng(0)
    stream = _make_synthetic_channels_stream(500, rng)
    diary = build_continuous_diary(stream, timescales=(1, 2, 4))

    gain = continuous_information_gain(
        _AlwaysOffUnit(), current_channel=0, key=(1, 0), diary=diary, target_channel=0
    )
    assert gain == 0.0


def test_continuous_information_gain_is_positive_for_an_informative_relation() -> None:
    rng = np.random.default_rng(0)
    stream = _make_synthetic_channels_stream(500, rng)
    diary = build_continuous_diary(stream, timescales=(1, 2, 4))

    gain = continuous_information_gain(
        _FiresWhenBHighUnit(),
        current_channel=0,
        key=(1, 0),
        diary=diary,
        target_channel=0,
    )
    assert gain > 0.0


def test_discover_and_rank_continuous_relations_finds_the_lagged_channel() -> None:
    """Channel 1's near-term trace genuinely predicts channel 0's next
    value (by construction); channel 2 is pure noise and must not
    outrank it.
    """
    rng = np.random.default_rng(0)
    stream = _make_synthetic_channels_stream(2000, rng)
    diary = build_continuous_diary(stream, timescales=(1, 2, 4, 8))

    ranked = discover_and_rank_continuous_relations(
        diary,
        current_channel=0,
        target_channel=0,
        n_channels=3,
        n_timescales=4,
        seed=1,
        min_gain=0.02,
        n_candidates=3,
        rounds=60,
        perturbations_per_round=12,
        episodes_per_candidate=3,
    )

    assert len(ranked) > 0
    top_channel = ranked[0][0][0]
    assert top_channel == 1


def test_continuous_information_gain_nets_out_a_baseline_channel() -> None:
    """A relation whose dendrite mostly leans on its `A` input (the
    current value) rather than genuinely using its `B` trace looks
    informative when a stream has strong trivial autocorrelation
    (`target_next ~= current`, e.g. a slowly-varying signal) -- but that
    "information" is just persistence in disguise, and must net out to
    ~zero once scored as a residual against that same baseline (see
    continuous_relation_discovery.py's docstring for the real Phase 7 bug
    this guards against: raw gain looked large and reproduced on
    held-out data, yet blending those relations made held-out prediction
    slightly *worse* than persistence alone).
    """
    rng = np.random.default_rng(0)
    n = 2000
    current = rng.uniform(-1.0, 1.0, size=n)
    target_next = current + rng.normal(0.0, 0.01, size=n)  # persistence nearly perfect
    noise_channel = rng.uniform(-1.0, 1.0, size=n)  # unrelated to current/target_next

    diary = [
        (
            np.array([current[t], noise_channel[t]]),
            np.zeros((1, 2)),
            np.array([target_next[t], noise_channel[t]]),
        )
        for t in range(n)
    ]

    class _FiresOnCurrentUnit:
        """Leans entirely on `a` (current value), ignores `b` -- exactly
        the "dominated by A" failure mode, not a genuine use of the trace.
        """

        def respond(self, a: float, b: float) -> float:
            return 1.0 if a > 0.0 else 0.0

    raw_gain = continuous_information_gain(
        _FiresOnCurrentUnit(),
        current_channel=0,
        key=(1, 0),
        diary=diary,
        target_channel=0,
    )
    residual_gain = continuous_information_gain(
        _FiresOnCurrentUnit(),
        current_channel=0,
        key=(1, 0),
        diary=diary,
        target_channel=0,
        baseline_channel=0,
    )
    assert raw_gain > 0.05  # looks informative if you don't net out persistence...
    assert abs(residual_gain) < 0.001  # ...but adds ~nothing once you do


def test_anchor_active_threshold_filters_out_irrelevant_background_positions() -> None:
    """A stream where `current_channel` is "active" (> 0.5) at only a
    small fraction of positions, and the target is genuinely predictable
    from `B` *only there* -- everywhere else `target_next` is a fixed,
    unrelated constant that has nothing to do with `B`. Without
    `anchor_active_threshold`, that background constant dominates the
    population and washes out the real signal; with it, the real signal
    survives (see continuous_relation_discovery.py's docstring for the
    real Phase 10 bug this guards against: every discovered relation
    converged on the background value and never recovered the other
    class at all).
    """
    rng = np.random.default_rng(0)
    n = 3000
    current = np.zeros(n)
    b_channel = rng.uniform(-1.0, 1.0, size=n)
    target_next = np.full(n, 5.0)  # irrelevant background constant everywhere
    active_positions = rng.choice(n, size=n // 10, replace=False)
    current[active_positions] = 1.0
    # at active positions, the target is resolved by b_channel's sign
    target_next[active_positions] = np.where(b_channel[active_positions] > 0, 1.0, -1.0)

    diary = [
        (
            np.array([current[t], b_channel[t]]),
            np.array([[0.0, b_channel[t]]]),  # q[timescale=0, channel=1] = b_channel
            np.array([target_next[t], b_channel[t]]),
        )
        for t in range(n)
    ]

    class _FiresWhenBPositiveUnit:
        def respond(self, a: float, b: float) -> float:
            return 1.0 if b > 0.0 else 0.0

    anchor_conditioned_gain = continuous_information_gain(
        _FiresWhenBPositiveUnit(),
        current_channel=0,
        key=(1, 0),
        diary=diary,
        target_channel=0,
        anchor_active_threshold=0.5,
    )
    manually_filtered_diary = [entry for entry in diary if entry[0][0] > 0.5]
    manually_filtered_gain = continuous_information_gain(
        _FiresWhenBPositiveUnit(),
        current_channel=0,
        key=(1, 0),
        diary=manually_filtered_diary,
        target_channel=0,
    )
    # the parameter must behave exactly like pre-filtering the diary --
    # and on this constructed active-only population, b's sign perfectly
    # resolves the target, so gain should be at its maximum (base_var=1,
    # with_var=0 since b>0 always implies target==1.0 exactly).
    assert abs(anchor_conditioned_gain - manually_filtered_gain) < 1e-9
    assert anchor_conditioned_gain > 0.9


def test_discover_and_rank_continuous_relations_skips_a_constant_trace() -> None:
    """A channel that never varies can't discriminate anything -- the
    min_std guard must exclude it rather than let a dendrite latch onto
    floating-point noise.
    """
    rng = np.random.default_rng(0)
    stream = _make_synthetic_channels_stream(500, rng)
    stream = np.concatenate(
        [stream, np.full((500, 1), 3.0)], axis=1
    )  # constant channel 3

    diary = build_continuous_diary(stream, timescales=(1, 2, 4))
    ranked = discover_and_rank_continuous_relations(
        diary,
        current_channel=0,
        target_channel=0,
        n_channels=4,
        n_timescales=3,
        seed=2,
        min_gain=0.02,
        rounds=30,
        perturbations_per_round=12,
    )

    assert not any(key[0] == 3 for key, _unit, _gain in ranked)
