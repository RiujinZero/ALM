"""Design doc sections 8-9, 23: structural plasticity -- growing a new
branch on (surprise AND poor representation), wiring it to the currently
active literals, and never manually pre-wiring task-specific conjunctions.
"""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.dendrites import DendriticConfig, DendriticPopulation


def test_grow_branch_wires_to_currently_active_literals() -> None:
    config = DendriticConfig(n_branches=4, input_dim=4, synapses_per_branch=2, seed=0)
    pop = DendriticPopulation(config)
    z = np.array([1.0, 0.0, 0.0, 1.0])  # literals 0 and 3 are "active"

    grown = pop.grow_branch(z)

    assert grown is not None
    assert pop.active[grown]
    assert set(np.flatnonzero(pop.mask[grown])) == {0, 3}
    assert pop.prototype[grown].tolist() == z.tolist()


def test_grow_branch_returns_none_when_all_branches_are_used() -> None:
    config = DendriticConfig(n_branches=1, input_dim=4, synapses_per_branch=2, seed=1)
    pop = DendriticPopulation(config)
    z = np.array([1.0, 0.0, 1.0, 0.0])
    first = pop.grow_branch(z)
    second = pop.grow_branch(z)

    assert first is not None
    assert second is None


def test_adapt_structure_grows_only_on_surprise_and_poor_representation() -> None:
    config = DendriticConfig(
        n_branches=4,
        input_dim=4,
        synapses_per_branch=2,
        surprise_threshold=0.3,
        match_threshold=0.5,
        seed=2,
    )
    pop = DendriticPopulation(config)
    z = np.array([1.0, 0.0, 1.0, 0.0])
    activity_no_match = np.zeros(4)  # every branch reads zero: "poorly represented"

    # Not surprising -> no growth even though poorly represented.
    pop.adapt_structure(z, activity_no_match, modulator=0.1)
    assert pop.n_active_branches() == 0

    # Surprising AND poorly represented -> grows.
    pop.adapt_structure(z, activity_no_match, modulator=0.9)
    assert pop.n_active_branches() == 1


def test_adapt_structure_does_not_grow_when_well_represented() -> None:
    config = DendriticConfig(n_branches=4, input_dim=4, match_threshold=0.5, seed=3)
    pop = DendriticPopulation(config)
    z = np.array([1.0, 0.0, 1.0, 0.0])
    activity_good_match = np.array([0.9, 0.0, 0.0, 0.0])  # already well represented

    pop.adapt_structure(z, activity_good_match, modulator=0.9)  # surprising, matched

    assert pop.n_active_branches() == 0


def test_structural_plasticity_disabled_pre_wires_all_branches_up_front() -> None:
    """Ablation B (section 25): branches are randomly wired at init and
    adapt_structure must never grow (or change) anything afterward.
    """
    config = DendriticConfig(
        n_branches=6,
        input_dim=4,
        synapses_per_branch=2,
        structural_plasticity=False,
        seed=4,
    )
    pop = DendriticPopulation(config)

    assert pop.n_active_branches() == 6
    mask_before = pop.mask.copy()

    z = np.array([1.0, 0.0, 1.0, 0.0])
    result = pop.adapt_structure(z, activity=np.zeros(6), modulator=0.9)

    assert result is None
    assert np.array_equal(pop.mask, mask_before)


def test_activation_is_zero_for_ungrown_branches() -> None:
    """An ungrown (all-zero-mask) branch must never look like a match."""
    config = DendriticConfig(n_branches=3, input_dim=4, seed=5)
    pop = DendriticPopulation(config)
    z = np.array([1.0, 0.0, 1.0, 0.0])

    activity = pop.activate(z)

    assert np.array_equal(activity, np.zeros(3))
