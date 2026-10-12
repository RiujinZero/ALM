"""Bootstrapped, consequence-sensitive teaching signal for eligible pathways.

Every prior calibrated pathway in this package (``risk_permission.py``,
``TemporalRelationalEligibility.confirm``) teaches eligible connections from
a single-step observed utility: how much better or worse was the very next
byte because this pathway was active. That is a valid ``delta_t`` but it is
not the general template. It cannot express "this made the immediate byte
slightly worse but left the organism in a more predictable position," since
it never consults what happens *after* the very next step.

This module adds the missing bootstrap term:

    delta_t = u_{t+1} + gamma * V(x_{t+1}) - V(x_t)

``u_{t+1}`` is the immediate observed outcome (the same signed bits-of-
surprise-reduction utility already used elsewhere), ``V`` is a local,
table-based value estimate over a hashable context key (no backprop, no
function approximation beyond per-key running averages, matching every
other local learner in this package), and ``gamma`` discounts the estimated
future. Setting ``gamma = 0`` collapses this exactly to the existing
one-step utility signal, which is verified as a regression identity.

The resulting ``delta_t`` is meant to be handed to
``TemporalRelationalEligibility.confirm`` (or an equivalent
``Delta w = eta * e_ij * delta_t`` rule) in place of a raw one-step utility.
"""

from __future__ import annotations

from collections.abc import Hashable

ContextKey = Hashable


class LocalValueEstimate:
    """A per-context running-average value table, updated by TD(0)."""

    def __init__(
        self, *, learning_rate: float = 0.1, initial_value: float = 0.0
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        self.learning_rate = float(learning_rate)
        self.initial_value = float(initial_value)
        self.values: dict[ContextKey, float] = {}
        self.visits: dict[ContextKey, int] = {}

    def value(self, key: ContextKey) -> float:
        return self.values.get(key, self.initial_value)

    def update(
        self,
        key: ContextKey,
        *,
        outcome: float,
        next_key: ContextKey | None,
        gamma: float,
    ) -> float:
        """Apply one TD(0) step at ``key`` and return the resulting ``delta_t``."""
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        current = self.value(key)
        bootstrap = 0.0 if next_key is None else self.value(next_key)
        delta = float(outcome) + gamma * bootstrap - current
        self.values[key] = current + self.learning_rate * delta
        self.visits[key] = self.visits.get(key, 0) + 1
        return delta

    @property
    def embodied_bytes(self) -> int:
        import pickle

        return len(pickle.dumps(self.values, protocol=pickle.HIGHEST_PROTOCOL))


def td_error(
    outcome: float,
    value_next: float,
    value_current: float,
    *,
    gamma: float,
) -> float:
    """Stateless form of the boxed equation for callers that keep their own ``V``."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    return float(outcome) + gamma * float(value_next) - float(value_current)
