"""Focused tests for surprise-created predictive ledger assemblies."""

from __future__ import annotations

import ast
import inspect

import numpy as np

from manifoldflow.dendritic_basin import selective_predictive_ledger as module


def test_prototype_birth_matching_and_frozen_inference() -> None:
    bank = module.SurprisePrototypeBank(
        module.SurprisePrototypeBankConfig(
            input_dim=2,
            capacity=3,
            birth_distance=1.0,
            match_distance=2.0,
            prototype_learning_rate=0.1,
        )
    )
    assert bank.observe(np.asarray([0.0, 0.0]), learn=True) == 0
    assert bank.observe(np.asarray([0.1, 0.0]), learn=True) == 0
    assert bank.observe(np.asarray([2.0, 0.0]), learn=True) == 1
    before = bank.fingerprint()
    assert bank.observe(np.asarray([0.0, 0.0]), learn=False) == 0
    assert bank.fingerprint() == before


def test_empty_ledger_exactly_preserves_prior() -> None:
    readout = module.PredictiveLedgerReadout(
        module.PredictiveLedgerReadoutConfig(assembly_capacity=3, mode_capacity=4)
    )
    prior = np.asarray([0.6, 0.3, 0.1])
    predicted = readout.predict((), (0, 2), prior)
    np.testing.assert_allclose(predicted, prior, atol=1e-15)


def test_local_update_improves_target_and_assigns_causal_utility() -> None:
    readout = module.PredictiveLedgerReadout(
        module.PredictiveLedgerReadoutConfig(
            assembly_capacity=3,
            mode_capacity=4,
            learning_rate=0.2,
            utility_decay=0.0,
        )
    )
    prior = np.asarray([0.7, 0.2, 0.1])
    target = np.asarray([0.0, 0.0, 1.0])
    before = readout.predict((1,), (0, 2), prior)
    readout.update((1,), (0, 2), prior, target)
    after = readout.predict((1,), (0, 2), prior)
    assert after[2] > before[2]
    # First prediction was identical to the prior, so it receives no
    # retroactive utility for the update it is only now learning.
    assert abs(readout.utility[1]) < 1e-12
    readout.update((1,), (0, 2), prior, target)
    assert readout.utility[1] > 0.0
    assert readout.retained(minimum_fires=2, minimum_utility_bits=0.0) == (1,)


def test_unrelated_active_assembly_is_not_updated() -> None:
    readout = module.PredictiveLedgerReadout(
        module.PredictiveLedgerReadoutConfig(
            assembly_capacity=3, mode_capacity=2, learning_rate=0.2
        )
    )
    before = readout.fingerprint()
    readout.update(
        (0,),
        (0,),
        np.asarray([0.5, 0.5]),
        np.asarray([0.0, 1.0]),
    )
    assert readout.fingerprint() != before
    assert readout.fires.tolist() == [1, 0, 0]


def test_module_has_no_backprop_framework_imports() -> None:
    source = inspect.getsource(module)
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint({"torch", "tensorflow", "jax", "autograd"})
    assert ".backward(" not in source
