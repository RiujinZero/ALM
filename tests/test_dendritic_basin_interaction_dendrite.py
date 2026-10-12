"""Tests for the v0.2 generic interaction dendrite (design doc
`Dendritic_Basin_Learning_Design_v0.2_Interaction_Dendrites.md`):
the RunningJointStats formula, the dendrite response, and local
perturbation/selection never using a label or a gradient of J.
"""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.interaction_dendrite import (
    InteractionDendrite,
    RunningJointStats,
)


def test_running_joint_stats_matches_numpy_on_a_batch() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=200)
    b = rng.normal(0, 1, size=200)
    d = 0.3 * a - 0.7 * b + rng.normal(0, 0.1, size=200)

    stats = RunningJointStats()
    for ai, bi, di in zip(a, b, d, strict=True):
        stats.update(float(ai), float(bi), float(di))

    # RunningJointStats is internally consistent using the population
    # convention (divide by n) throughout; numpy's np.var defaults to
    # ddof=0 (population) but np.cov defaults to ddof=1 (sample) -- pass
    # ddof=0 explicitly so both sides use the same convention.
    np.testing.assert_allclose(stats.var_a(), np.var(a), atol=1e-8)
    np.testing.assert_allclose(stats.var_b(), np.var(b), atol=1e-8)
    np.testing.assert_allclose(stats.var_d(), np.var(d), atol=1e-8)
    np.testing.assert_allclose(stats.cov_d_a(), np.cov(d, a, ddof=0)[0, 1], atol=1e-8)
    np.testing.assert_allclose(stats.cov_d_b(), np.cov(d, b, ddof=0)[0, 1], atol=1e-8)


def test_objective_formula_matches_direct_computation() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=500)
    b = rng.normal(0, 1, size=500)
    d = np.exp(-((a - b) ** 2))

    stats = RunningJointStats()
    for ai, bi, di in zip(a, b, d, strict=True):
        stats.update(float(ai), float(bi), float(di))

    expected = (
        np.var(d)
        - np.cov(d, a, ddof=0)[0, 1] ** 2 / np.var(a)
        - np.cov(d, b, ddof=0)[0, 1] ** 2 / np.var(b)
    )
    np.testing.assert_allclose(stats.objective(), expected, atol=1e-6)


def test_empty_stats_objective_is_zero() -> None:
    assert RunningJointStats().objective() == 0.0


def test_dendrite_response_matches_the_bell_formula() -> None:
    dendrite = InteractionDendrite(seed=0)
    a, b = 0.7, -0.3
    drive = dendrite.w_a * a + dendrite.w_b * b
    expected = np.exp(-((drive - dendrite.mu) ** 2) / (2 * dendrite.sigma**2))
    np.testing.assert_allclose(dendrite.respond(a, b), expected)


def test_response_is_maximal_when_drive_equals_mu() -> None:
    dendrite = InteractionDendrite(seed=1)
    # Solve for `a` such that w_a*a + w_b*0 == mu (assuming w_a != 0).
    a_at_mu = dendrite.mu / dendrite.w_a
    on_target = dendrite.respond(a_at_mu, 0.0)
    off_target = dendrite.respond(a_at_mu + 5.0, 0.0)
    assert on_target > off_target
    np.testing.assert_allclose(on_target, 1.0, atol=1e-9)


def test_observe_updates_lifetime_statistics() -> None:
    dendrite = InteractionDendrite(seed=2)
    assert dendrite.stats.n == 0
    dendrite.observe(0.5, 0.5)
    dendrite.observe(-0.5, 0.5)
    assert dendrite.stats.n == 2


def test_evaluate_on_does_not_touch_lifetime_statistics() -> None:
    dendrite = InteractionDendrite(seed=3)
    dendrite.observe(1.0, 1.0)
    n_before = dendrite.stats.n
    dendrite.evaluate_on([(0.1, 0.2), (0.3, 0.4)])
    assert dendrite.stats.n == n_before


def test_perturbed_only_differs_in_parameters_not_stats() -> None:
    dendrite = InteractionDendrite(seed=4)
    dendrite.observe(1.0, 1.0)
    rng = np.random.default_rng(0)
    candidate = dendrite.perturbed(step_size=0.1, rng=rng)

    assert candidate.stats.n == 0  # fresh stats, not carried over
    assert (candidate.w_a, candidate.w_b, candidate.mu) != (
        dendrite.w_a,
        dendrite.w_b,
        dendrite.mu,
    )


def test_perturb_and_select_never_decreases_j_on_the_scoring_batch() -> None:
    """Local search only keeps a perturbation if it scores higher on the
    exact batch it was evaluated against -- so J on that batch is
    monotonically non-decreasing across repeated calls with fresh draws
    is not guaranteed (batches differ), but a single call must not accept
    a strictly worse candidate on its own batch.
    """
    rng = np.random.default_rng(5)
    dendrite = InteractionDendrite(seed=6)
    samples = [(float(rng.normal()), float(rng.normal())) for _ in range(20)]
    j_before = dendrite.evaluate_on(samples)

    j_after = dendrite.perturb_and_select(samples, step_size=0.2, rng=rng)

    assert j_after >= j_before


def test_perturb_and_select_can_improve_toward_a_known_relation() -> None:
    """Sanity check the whole local-search loop converges upward over many
    rounds on a genuinely interactive (XOR-like) target -- not asserting
    XOR discovery here (that's relation_unit's job), just that repeated
    perturb_and_select calls trend toward higher J.
    """
    rng = np.random.default_rng(7)
    dendrite = InteractionDendrite(seed=8)

    def draw_batch(n):
        a = rng.integers(0, 2, size=n).astype(float)
        b = rng.integers(0, 2, size=n).astype(float)
        return list(zip(a.tolist(), b.tolist(), strict=True))

    j_start = dendrite.evaluate_on(draw_batch(200))
    for _ in range(200):
        dendrite.perturb_and_select(draw_batch(12), step_size=0.3, rng=rng)
    j_end = dendrite.evaluate_on(draw_batch(200))

    assert j_end >= j_start
