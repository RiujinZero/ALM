from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.recalled_expression import (
    RecalledCorticalExpression,
)


def test_silent_return_row_can_mature_only_from_external_observation() -> None:
    item = (7, 2)
    baseline = np.full(256, 1.0 / 256.0)
    expression = RecalledCorticalExpression(minimum_evidence=2)
    expression.activate((item,))

    silent = expression.predict(baseline, maximum_gain=0.5)
    expression.observe(ord("x"), external=False)
    assert silent.gain == 0.0
    assert expression.evidence == {}

    expression.reset_episode()
    expression.activate((item,))
    expression.predict(baseline, maximum_gain=0.5)
    expression.observe(ord("x"), external=True)

    assert expression.evidence[(7, 2, 256)] == 1
    assert expression.weights[(7, 2, 256)][ord("x")] > 0.0


def test_recalled_identity_learns_an_ordered_external_trace() -> None:
    path = RecalledCorticalExpression(minimum_evidence=1)
    red = (11, 4)
    blue = (12, 4)
    path.learn_sequence(red, b"red", repetitions=24)
    path.learn_sequence(blue, b"blue", repetitions=24)
    baseline = np.full(256, 1.0 / 256.0)

    path.activate((red,))
    generated = bytearray()
    for _ in range(3):
        signal = path.predict(baseline, maximum_gain=1.0)
        byte_id = int(np.argmax(signal.distribution))
        generated.append(byte_id)
        path.observe(byte_id, external=False)

    assert bytes(generated) == b"red"


def test_wrong_identity_preserves_receptor_but_changes_content() -> None:
    path = RecalledCorticalExpression(minimum_evidence=1)
    path.learn_sequence((11, 8), b"red", repetitions=24)
    path.learn_sequence((12, 8), b"blue", repetitions=24)
    baseline = np.full(256, 1.0 / 256.0)

    path.activate(((12, 8),))
    first = path.predict(baseline, maximum_gain=1.0)

    assert int(np.argmax(first.distribution)) == ord("b")
    assert first.distribution[ord("b")] > first.distribution[ord("r")]


def test_recalled_identity_can_resume_from_a_learned_prefix_phase() -> None:
    path = RecalledCorticalExpression(minimum_evidence=1)
    item = (11, 4)
    path.learn_sequence(item, b" red", repetitions=24)
    baseline = np.full(256, 1.0 / 256.0)

    path.activate((item,), initial_phases=(ord(" "),))
    signal = path.predict(baseline, maximum_gain=1.0)

    assert int(np.argmax(signal.distribution)) == ord("r")


def test_order_sensitive_phase_cells_terminate_after_repeated_bytes() -> None:
    path = RecalledCorticalExpression(
        minimum_evidence=1, order_sensitive_phases=True
    )
    item = (11, 4)
    path.learn_sequence(item, b"ball", repetitions=24)
    baseline = np.full(256, 1.0 / 256.0)

    path.activate((item,))
    generated = bytearray()
    for _ in range(4):
        signal = path.predict(baseline, maximum_gain=1.0)
        byte_id = int(np.argmax(signal.distribution))
        generated.append(byte_id)
        path.observe(byte_id, external=False)
    terminal = path.predict(baseline, maximum_gain=1.0)

    assert bytes(generated) == b"ball"
    assert terminal.active_rows == 0
    assert np.array_equal(terminal.distribution, baseline)


def test_order_sensitive_phase_cells_resume_after_a_prefix() -> None:
    path = RecalledCorticalExpression(
        minimum_evidence=1, order_sensitive_phases=True
    )
    item = (11, 4)
    path.learn_sequence(item, b"the ball", repetitions=24)
    phase = path.phase_after_prefix(item, b"the ")
    baseline = np.full(256, 1.0 / 256.0)

    assert phase is not None
    path.activate((item,), initial_phases=(phase,))
    signal = path.predict(baseline, maximum_gain=1.0)

    assert int(np.argmax(signal.distribution)) == ord("b")


def test_generated_activity_advances_but_cannot_self_train() -> None:
    path = RecalledCorticalExpression(minimum_evidence=1)
    item = (7, 2)
    path.activate((item,))
    baseline = np.full(256, 1.0 / 256.0)
    path.predict(baseline, maximum_gain=1.0)
    path.observe(ord("x"), external=False)

    assert path.weights == {}
    assert path.evidence == {}
    assert path.active[item][1] == ord("x")


def test_same_identity_on_unlearned_receptor_is_silent() -> None:
    path = RecalledCorticalExpression(minimum_evidence=1)
    path.learn_sequence((3, 1), b"cat", repetitions=16)
    baseline = np.linspace(1.0, 2.0, 256)
    baseline /= baseline.sum()

    path.activate(((3, 2),))
    signal = path.predict(baseline, maximum_gain=1.0)

    assert signal.active_rows == 0
    assert np.array_equal(signal.distribution, baseline)


def test_pre_normalization_cancellation_preserves_discriminating_current() -> None:
    path = RecalledCorticalExpression(minimum_evidence=1)
    target = (1, 4)
    distractors = tuple((assembly, 4) for assembly in range(2, 10))
    path.learn_sequence(target, b"r", repetitions=24)
    for item in distractors:
        path.learn_sequence(item, b"b", repetitions=24)
    active = (target,) + distractors
    strengths = (1.0,) + tuple(0.1 for _ in distractors)
    baseline = np.full(256, 1.0 / 256.0)

    path.activate(active, strengths)
    original = path.predict(
        baseline, maximum_gain=0.5, shared_cancellation="none"
    )
    path.activate(active, strengths)
    residual = path.predict(
        baseline, maximum_gain=0.5, shared_cancellation="pre"
    )

    original_margin = (
        original.distribution[ord("r")] - original.distribution[ord("b")]
    )
    residual_margin = (
        residual.distribution[ord("r")] - residual.distribution[ord("b")]
    )
    assert residual_margin > original_margin
    assert residual.distribution[ord("r")] > residual.distribution[ord("b")]
    assert np.isclose(residual.distribution.sum(), 1.0)
