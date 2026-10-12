from collections import deque
from types import SimpleNamespace

import pytest

from manifoldflow.experiments.dendritic_basin_phase102_phase_aligned_chain import (
    _advance_generated_state,
)


class _Expression:
    def __init__(self) -> None:
        self.observed = []

    def observe(self, byte_id: int, *, external: bool) -> None:
        self.observed.append((byte_id, external))


class _Learner:
    def __init__(self) -> None:
        self.observed = []

    def _observe_probability_only(self, byte_id: int, *, learn: bool) -> None:
        self.observed.append((byte_id, learn))

    def _active_route_keys(self) -> tuple[tuple[int, int], ...]:
        return ((3, 7),)

    def _temporal_disagreement(self, routes: object) -> float:
        assert routes == ((3, 7),)
        return 0.25


class _Clock:
    def __init__(self) -> None:
        self.advances = 0

    def advance(self) -> None:
        self.advances += 1


def test_generated_advance_preserves_provenance_without_durable_learning() -> None:
    expression = _Expression()
    learner = _Learner()
    field = _Clock()
    access = _Clock()
    arm = SimpleNamespace(
        recalled_expression=expression,
        learner=learner,
        previous_routes={(2, 4)},
        previous_surprise=1.0,
        last_boundary_signal=0.0,
        raw_window=bytearray(b"0123456789abcdef"),
        route_history=deque(maxlen=4),
        field=field,
        relational_access=access,
        return_adaptation_half_life=2.0,
        return_adaptation={(9, 1): 1.0},
    )

    _advance_generated_state(arm, ord("x"), 0.25)

    assert expression.observed == [(ord("x"), False)]
    assert learner.observed == [(ord("x"), False)]
    assert arm.raw_window == bytearray(b"123456789abcdefx")
    assert arm.route_history[-1] == ((7, 3),)
    assert field.advances == 1
    assert access.advances == 1
    assert arm.return_adaptation[(9, 1)] == pytest.approx(2**-0.5)
    assert arm.previous_routes == {(3, 7)}
    assert arm.previous_surprise == pytest.approx(2.0)
    assert arm.last_boundary_signal == pytest.approx(2.25)
