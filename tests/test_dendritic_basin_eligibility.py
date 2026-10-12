"""Design doc section 15: forward-only eligibility traces on dendritic
synapses, and section 16's modulated prototype update off that trace."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.dendrites import DendriticConfig, DendriticPopulation


def test_eligibility_matches_geometric_decay_formula() -> None:
    config = DendriticConfig(n_branches=2, input_dim=4, trace_decay=0.6, seed=0)
    pop = DendriticPopulation(config)
    pop.grow_branch(np.array([1.0, 0.0, 1.0, 0.0]))
    pop.grow_branch(np.array([0.0, 1.0, 0.0, 1.0]))

    zs = [
        np.array([1.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 1.0]),
    ]
    expected = np.zeros_like(pop.eligibility)
    for z in zs:
        activity = pop.activate(z)
        pop.update_eligibility(z, activity)
        expected = 0.6 * expected + activity[:, None] * z[None, :] * pop.mask
        np.testing.assert_allclose(pop.eligibility, expected)


def test_eligibility_is_zero_outside_the_branch_mask() -> None:
    config = DendriticConfig(n_branches=2, input_dim=6, seed=1)
    pop = DendriticPopulation(config)
    pop.grow_branch(np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))

    z = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    activity = pop.activate(z)
    pop.update_eligibility(z, activity)

    unmasked = pop.mask == 0
    assert np.all(pop.eligibility[unmasked] == 0)


def test_prototype_update_scales_with_modulator_and_eligibility() -> None:
    config = DendriticConfig(n_branches=1, input_dim=4, prototype_lr=1.0, seed=2)
    pop = DendriticPopulation(config)
    pop.grow_branch(np.array([1.0, 0.0, 1.0, 0.0]))
    pop.eligibility[:] = np.array([[0.5, 0.0, 0.5, 0.0]])
    prototype_before = pop.prototype.copy()

    pop.learn(modulator=0.4)

    delta = 1.0 * 0.4 * np.array([[0.5, 0.0, 0.5, 0.0]]) * pop.mask
    expected = prototype_before + delta
    np.clip(expected, 0.0, 1.0, out=expected)
    np.testing.assert_allclose(pop.prototype, expected)


def test_usefulness_accumulates_modulator_weighted_activity() -> None:
    config = DendriticConfig(n_branches=1, input_dim=4, usefulness_decay=0.5, seed=3)
    pop = DendriticPopulation(config)
    pop.grow_branch(np.array([1.0, 0.0, 1.0, 0.0]))
    z = np.array([1.0, 0.0, 1.0, 0.0])
    activity = pop.activate(z)

    pop.learn(modulator=1.0)

    np.testing.assert_allclose(pop.usefulness, 0.5 * 0.0 + 1.0 * activity)
