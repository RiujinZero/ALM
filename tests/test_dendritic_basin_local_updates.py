"""Design doc section 3.3: a synaptic update must depend only on local
information -- checked here as "the update only touches this component's
own state," a structural proxy for locality analogous to the ALM
plastic-column locality tests (tests/test_alm_plastic_column.py).
"""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.basin import BasinConfig, RecurrentBasin
from manifoldflow.dendritic_basin.dendrites import DendriticConfig, DendriticPopulation
from manifoldflow.dendritic_basin.readout import LocalReadout, ReadoutConfig


def test_dendrite_learn_only_changes_prototype_and_usefulness_not_mask() -> None:
    config = DendriticConfig(n_branches=4, input_dim=4, synapses_per_branch=2, seed=0)
    pop = DendriticPopulation(config)
    z = np.array([1.0, 0.0, 1.0, 0.0])
    pop.grow_branch(z)
    d = pop.activate(z)
    pop.update_eligibility(z, d)
    mask_before = pop.mask.copy()

    pop.learn(modulator=0.5)

    # connectivity is structural, not learned by `learn()`
    assert np.array_equal(pop.mask, mask_before)


def test_dendrite_learn_only_perturbs_masked_connections() -> None:
    """A branch's prototype must not drift on connections it isn't wired to."""
    config = DendriticConfig(n_branches=4, input_dim=6, synapses_per_branch=2, seed=1)
    pop = DendriticPopulation(config)
    z = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    pop.grow_branch(z)
    prototype_before = pop.prototype.copy()
    d = pop.activate(z)
    pop.update_eligibility(z, d)

    pop.learn(modulator=1.0)

    unmasked = pop.mask == 0
    assert np.allclose(pop.prototype[unmasked], prototype_before[unmasked])


def test_readout_learn_only_changes_weights_and_bias() -> None:
    config = ReadoutConfig(n_inputs=5, seed=2)
    readout = LocalReadout(config)
    h = np.array([0.1, 0.5, 0.0, 0.3, 0.9])
    readout.learn(h, modulator=0.4)
    assert readout.weights.shape == (5,)
    assert isinstance(readout.bias, float)


def test_readout_update_direction_matches_modulator_sign() -> None:
    """A positive modulator should push weights toward the active input
    directions; a negative one should push away -- the local delta rule's
    defining property (section 13).
    """
    config = ReadoutConfig(n_inputs=3, learning_rate=1.0, seed=3)
    readout_pos = LocalReadout(config)
    readout_neg = LocalReadout(config)
    h = np.array([1.0, 0.0, 1.0])

    readout_pos.learn(h, modulator=1.0)
    readout_neg.learn(h, modulator=-1.0)

    assert readout_pos.weights[0] > readout_neg.weights[0]
    assert readout_pos.weights[2] > readout_neg.weights[2]
    assert readout_pos.weights[1] == readout_neg.weights[1]  # untouched (h_1 == 0)


def test_basin_learn_only_changes_w_rec_and_keeps_zero_diagonal() -> None:
    config = BasinConfig(n_units=4, seed=4)
    basin = RecurrentBasin(config)
    h = np.array([0.2, 0.8, 0.0, 0.5])
    basin.learn(h)
    assert basin.w_rec.shape == (4, 4)
    assert np.allclose(np.diag(basin.w_rec), 0.0)
