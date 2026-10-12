"""Tests for the persistent continuous register (design doc section 54,
"Visual world -- ... persistence, occlusion, identity")."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.persistent_continuous_register import (
    PersistentContinuousRegister,
)


def test_register_starts_at_zero() -> None:
    register = PersistentContinuousRegister(n_channels=3)
    assert np.array_equal(register.snapshot(), np.zeros(3))


def test_register_updates_when_visible() -> None:
    register = PersistentContinuousRegister(n_channels=2)
    register.update(np.array([1.5, -2.0]), visible=True)
    assert np.array_equal(register.snapshot(), np.array([1.5, -2.0]))


def test_register_holds_last_value_while_not_visible() -> None:
    """The core property distinguishing this from a decaying trace: the
    stored value must not drift at all across an arbitrarily long span
    of `visible=False` updates.
    """
    register = PersistentContinuousRegister(n_channels=1)
    register.update(np.array([7.0]), visible=True)
    for _ in range(1000):
        register.update(np.array([0.0]), visible=False)
    assert register.snapshot()[0] == 7.0


def test_register_resumes_updating_once_visible_again() -> None:
    register = PersistentContinuousRegister(n_channels=1)
    register.update(np.array([7.0]), visible=True)
    register.update(np.array([0.0]), visible=False)
    register.update(np.array([3.0]), visible=True)
    assert register.snapshot()[0] == 3.0


def test_reset_clears_the_held_value() -> None:
    register = PersistentContinuousRegister(n_channels=1)
    register.update(np.array([7.0]), visible=True)
    register.reset()
    assert register.snapshot()[0] == 0.0
