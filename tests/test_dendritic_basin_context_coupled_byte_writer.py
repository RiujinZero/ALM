from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.context_coupled_byte_writer import (
    ContextCoupledByteWriter,
    ContextCoupledEpisode,
    likelihood_ratio_correction,
)


def test_matched_allocation_and_wide_additive_active_count() -> None:
    writers = {
        kind: ContextCoupledByteWriter(
            kind, alphabet_size=7, table_bits=8, local_orders=(1, 2, 4, 8)
        )
        for kind in ("local", "additive", "additive_wide", "coupled")
    }
    state = (11, 22, 33)
    prefix = (1, 2, 3)

    assert {writer.parameter_count for writer in writers.values()} == {256 * 7}
    assert {writer.allocated_bytes for writer in writers.values()} == {256 * 7 * 4}
    assert len(writers["local"].raw_feature_ids(prefix, state)) == 5
    assert len(writers["additive"].raw_feature_ids(prefix, state)) == 8
    assert len(writers["additive_wide"].raw_feature_ids(prefix, state)) == 23
    assert len(writers["coupled"].raw_feature_ids(prefix, state)) == 23
    assert writers["additive_wide"].raw_active_feature_count(3) == 23
    assert writers["coupled"].raw_active_feature_count(3) == 23


def test_local_features_see_only_configured_emitted_suffix() -> None:
    writer = ContextCoupledByteWriter("local", alphabet_size=16, table_bits=10)

    first = writer.local_feature_ids((12, 13, 0, 1, 2, 3, 4, 5, 6, 7))
    second = writer.local_feature_ids((14, 15, 0, 1, 2, 3, 4, 5, 6, 7))

    np.testing.assert_array_equal(first, second)


def test_collision_multiplicity_drives_prediction_and_one_local_update() -> None:
    writer = ContextCoupledByteWriter(
        "coupled", alphabet_size=2, table_bits=1, local_orders=(1,)
    )
    raw = writer.raw_feature_ids((0,), (101, 202))
    indexes, multiplicities = writer.features((0,), (101, 202))

    assert len(raw) == writer.raw_active_feature_count(2) == 8
    assert len(indexes) <= 2
    assert int(multiplicities.sum()) == len(raw)
    assert np.max(multiplicities) > 1

    writer.fit_episode((1,), (101, 202), prefix=(0,))
    weights = writer.snapshot().weights
    modified = np.flatnonzero(np.any(weights != 0.0, axis=1))
    np.testing.assert_array_equal(modified, indexes)
    np.testing.assert_allclose(
        writer.scores((0,), (101, 202)), multiplicities @ weights[indexes]
    )


def test_coupling_learns_xor_that_all_additive_controls_cannot_express() -> None:
    writers = {
        kind: ContextCoupledByteWriter(
            kind,
            alphabet_size=2,
            table_bits=16,
            local_orders=(1,),
            learning_rate=0.55,
        )
        for kind in ("local", "additive", "additive_wide", "coupled")
    }
    cases = [
        (local, state, int(local ^ (state == 20)))
        for local in (0, 1)
        for state in (10, 20)
    ]
    for _ in range(100):
        for local, state, target in cases:
            for writer in writers.values():
                writer.fit_episode((target,), (state,), prefix=(local,))

    correct: dict[str, int] = {}
    for kind, writer in writers.items():
        correct[kind] = sum(
            writer.generate((state,), max_symbols=1, prefix=(local,))[0] == target
            for local, state, target in cases
        )

    assert correct["coupled"] == 4
    assert correct["local"] < 4
    assert correct["additive"] < 4
    assert correct["additive_wide"] < 4

    coupled = writers["coupled"]
    frozen_correct = sum(
        coupled.generate((state,), max_symbols=1, prefix=(local,), freeze_joint=True)[0]
        == target
        for local, state, target in cases
    )
    assert frozen_correct < 4


def test_teacher_forcing_preserves_episode_boundaries_and_validates_first() -> None:
    writer = ContextCoupledByteWriter(
        "local", alphabet_size=3, table_bits=16, local_orders=(1,)
    )
    after_one, _ = writer.features((1,), ())
    at_boundary, _ = writer.features((), ())
    exclusive_after_one = np.setdiff1d(after_one, at_boundary)
    assert len(exclusive_after_one) == 1

    writer.fit_episodes(
        (
            ContextCoupledEpisode((1,), ()),
            ContextCoupledEpisode((2,), ()),
        )
    )
    assert writer.episodes_seen == 2
    assert writer.symbols_seen == 2
    assert not np.any(writer.snapshot().weights[exclusive_after_one])

    before = writer.fingerprint()
    with pytest.raises(ValueError, match="outside the output alphabet"):
        writer.fit_episode((1, 99), ())
    assert writer.fingerprint() == before
    assert writer.episodes_seen == 2
    assert writer.symbols_seen == 2


def test_generation_is_autoregressive_deterministic_and_inference_frozen() -> None:
    writer = ContextCoupledByteWriter(
        "coupled", alphabet_size=4, table_bits=14, learning_rate=0.55
    )
    for _ in range(40):
        writer.fit_episode((1, 2, 3), (777,))
    before = writer.fingerprint()

    first = writer.generate((777,), max_symbols=12, stop_symbol=3)
    second = writer.generate((777,), max_symbols=12, stop_symbol=3)
    probabilities = writer.distribution((1, 2), (777,))

    assert first == second == (1, 2, 3)
    assert np.isclose(probabilities.sum(), 1.0)
    assert writer.fingerprint() == before


def test_snapshot_round_trip_is_exact_and_detached() -> None:
    writer = ContextCoupledByteWriter(
        "coupled", alphabet_size=5, table_bits=8, local_orders=(1, 3)
    )
    writer.fit_episode((2, 4), (-7, 19), prefix=(1,))
    snapshot = writer.snapshot()
    restored = ContextCoupledByteWriter.from_snapshot(snapshot)

    assert not snapshot.weights.flags.writeable
    assert restored.kind == writer.kind
    assert restored.episodes_seen == writer.episodes_seen
    assert restored.symbols_seen == writer.symbols_seen
    assert restored.fingerprint() == writer.fingerprint()
    np.testing.assert_array_equal(
        restored.scores((1, 2), (-7, 19)), writer.scores((1, 2), (-7, 19))
    )


def test_full_byte_alphabet_accepts_bytes_and_emits_bytes() -> None:
    writer = ContextCoupledByteWriter("local", alphabet_size=256, table_bits=5)
    for _ in range(4):
        writer.fit_episode(b"\xff", ())

    assert writer.generate_bytes((), max_bytes=1) == b"\xff"


def test_beta_zero_is_exact_and_nonzero_uses_matched_route_ratio() -> None:
    base = np.asarray((0.5, 0.5), dtype=np.float32)

    bypass = likelihood_ratio_correction(
        base,
        (float("nan"), 1.0),
        (float("nan"), 1.0),
        beta=0.0,
    )
    corrected = likelihood_ratio_correction(
        base,
        (0.8, 0.2),
        (0.2, 0.8),
        beta=1.0,
    )

    assert bypass.dtype == base.dtype
    assert np.array_equal(bypass, base)
    assert corrected[1] > corrected[0]
    assert np.isclose(corrected.sum(), 1.0)


@pytest.mark.parametrize("kind", ("local", "additive", "additive_wide", "coupled"))
def test_zero_residual_is_base_identity_for_every_feature_kind(kind: str) -> None:
    writer = ContextCoupledByteWriter(
        kind,
        alphabet_size=3,
        table_bits=8,
        local_orders=(1, 2),
    )
    state = () if kind == "local" else (101, 202)
    base = np.asarray((0.25, 0.0, 0.75), dtype=np.float64)

    anchored = writer.base_anchored_distribution(base, (2, 1), state)

    np.testing.assert_allclose(anchored, base, rtol=0.0, atol=1e-15)
    assert anchored[1] == 0.0
    assert np.isclose(anchored.sum(), 1.0)


def test_one_base_anchored_step_has_exact_feature_normalized_update() -> None:
    writer = ContextCoupledByteWriter(
        "local",
        alphabet_size=2,
        table_bits=8,
        learning_rate=0.4,
        local_orders=(1,),
    )
    indexes, multiplicities = writer.features((0,), ())
    base = np.asarray((0.75, 0.25), dtype=np.float64)
    expected_error = np.asarray((-0.75, 0.75))
    expected_update = (
        (writer.learning_rate / float(multiplicities @ multiplicities))
        * multiplicities[:, None]
        * expected_error[None, :]
    )

    diagnostics = writer.fit_base_anchored_episode(
        (1,),
        (base,),
        (),
        prefix=(0,),
    )
    weights = writer.snapshot().weights

    np.testing.assert_allclose(
        weights[indexes],
        expected_update,
        rtol=0.0,
        atol=2e-8,
    )
    assert diagnostics.symbols == diagnostics.symbol_updates == 1
    assert diagnostics.raw_feature_activations == writer.raw_active_feature_count(0)
    assert diagnostics.unique_feature_row_updates == len(indexes)
    assert diagnostics.scalar_weight_slots_touched == len(indexes) * 2
    assert diagnostics.base_nll_nats == pytest.approx(-np.log(0.25))
    assert diagnostics.anchored_nll_nats == pytest.approx(-np.log(0.25))
    assert diagnostics.residual_improvement_nats == pytest.approx(0.0)
    assert diagnostics.maximum_absolute_update == pytest.approx(
        float(np.max(np.abs(weights[indexes])))
    )
    assert diagnostics.minimum_target_base_probability == 0.25
    assert writer.episodes_seen == 1
    assert writer.symbols_seen == 1


def test_base_anchored_episode_advances_only_the_observed_causal_suffix() -> None:
    config = {
        "alphabet_size": 3,
        "table_bits": 12,
        "learning_rate": 0.3,
        "local_orders": (1, 2),
    }
    batched = ContextCoupledByteWriter("coupled", **config)
    split = ContextCoupledByteWriter("coupled", **config)
    bases = (
        np.asarray((0.2, 0.5, 0.3)),
        np.asarray((0.4, 0.1, 0.5)),
    )

    batched.fit_base_anchored_episode(
        (1, 2),
        bases,
        (77,),
        prefix=(0,),
    )
    split.fit_base_anchored_episode((1,), bases[:1], (77,), prefix=(0,))
    split.fit_base_anchored_episode((2,), bases[1:], (77,), prefix=(0, 1))

    np.testing.assert_array_equal(
        batched.snapshot().weights,
        split.snapshot().weights,
    )
    assert batched.episodes_seen == 1
    assert split.episodes_seen == 2
    assert batched.symbols_seen == split.symbols_seen == 2


def test_base_anchored_diagnostics_match_wide_and_coupled_active_budgets() -> None:
    writers = {
        kind: ContextCoupledByteWriter(
            kind,
            alphabet_size=4,
            table_bits=16,
            local_orders=(1, 2, 4),
        )
        for kind in ("additive_wide", "coupled")
    }
    state = (11, 22, 33)
    base = (0.1, 0.2, 0.3, 0.4)
    diagnostics = {
        kind: writer.fit_base_anchored_episode(
            (3,),
            (base,),
            state,
            prefix=(0, 1),
        )
        for kind, writer in writers.items()
    }

    assert {writer.parameter_count for writer in writers.values()} == {65536 * 4}
    assert {writer.raw_active_feature_count(3) for writer in writers.values()} == {19}
    assert {row.raw_feature_activations for row in diagnostics.values()} == {19}
    for row in diagnostics.values():
        assert row.unique_feature_row_updates <= row.raw_feature_activations
        assert row.scalar_weight_slots_touched == row.unique_feature_row_updates * 4


def test_zero_gain_is_bit_exact_and_does_not_inspect_writer_route() -> None:
    writer = ContextCoupledByteWriter("coupled", alphabet_size=3, table_bits=8)
    base = np.asarray((0.25, 0.0, 0.75), dtype=np.float32)
    before = writer.fingerprint()

    bypass = writer.base_anchored_distribution(
        base,
        prefix=(999,),
        persistent_feature_ids=(),
        gain=0.0,
        residual_clip=2.0,
    )

    assert bypass.dtype == base.dtype
    assert np.array_equal(bypass, base)
    assert writer.fingerprint() == before


def test_base_anchored_support_and_snapshot_round_trip() -> None:
    writer = ContextCoupledByteWriter(
        "coupled",
        alphabet_size=3,
        table_bits=10,
        local_orders=(1,),
    )
    for _ in range(3):
        writer.fit_base_anchored_episode(
            (2,),
            ((0.2, 0.0, 0.8),),
            (17,),
            prefix=(1,),
        )
    restored = ContextCoupledByteWriter.from_snapshot(writer.snapshot())

    original = writer.base_anchored_distribution(
        (0.2, 0.0, 0.8),
        (1,),
        (17,),
        gain=1.5,
        residual_clip=0.25,
    )
    round_trip = restored.base_anchored_distribution(
        (0.2, 0.0, 0.8),
        (1,),
        (17,),
        gain=1.5,
        residual_clip=0.25,
    )
    residual = 1.5 * writer.scores((1,), (17,)).astype(np.float64)
    residual = np.clip(residual, -0.25, 0.25)
    expected = np.asarray((0.2, 0.0, 0.8)) * np.exp(residual)
    expected /= expected.sum()

    np.testing.assert_array_equal(round_trip, original)
    np.testing.assert_allclose(original, expected, rtol=0.0, atol=1e-15)
    assert original[1] == 0.0
    assert original[2] > 0.8


@pytest.mark.parametrize(
    "base, message",
    (
        ((0.5, 0.5), "shape"),
        ((0.5, float("nan"), 0.5), "finite"),
        ((0.5, -0.1, 0.6), "non-negative"),
        ((0.2, 0.2, 0.2), "sum to one"),
    ),
)
def test_base_anchored_inference_rejects_invalid_base_inputs(
    base: tuple[float, ...], message: str
) -> None:
    writer = ContextCoupledByteWriter("local", alphabet_size=3, table_bits=5)
    with pytest.raises(ValueError, match=message):
        writer.base_anchored_distribution(base, (), ())


def test_base_anchored_training_validates_all_rows_before_updating() -> None:
    writer = ContextCoupledByteWriter("local", alphabet_size=3, table_bits=8)
    before = writer.fingerprint()

    with pytest.raises(ValueError, match="positive base support"):
        writer.fit_base_anchored_episode(
            (1, 2),
            ((0.2, 0.8, 0.0), (0.5, 0.5, 0.0)),
            (),
        )
    assert writer.fingerprint() == before
    assert writer.episodes_seen == writer.symbols_seen == 0

    with pytest.raises(ValueError, match="exactly 2 rows"):
        writer.fit_base_anchored_episode(
            (1, 2),
            ((0.2, 0.8, 0.0),),
            (),
        )
    assert writer.fingerprint() == before


def test_base_anchored_training_rolls_back_a_late_numerical_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = ContextCoupledByteWriter("local", alphabet_size=3, table_bits=8)
    before = writer.fingerprint()
    calls = 0

    def fail_on_second_softmax(
        base: np.ndarray,
        residual: np.ndarray,
    ) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError("forced late numerical failure")
        weighted = base * np.exp(residual - np.max(residual))
        return weighted / weighted.sum()

    monkeypatch.setattr(
        "manifoldflow.dendritic_basin.context_coupled_byte_writer."
        "_base_anchored_softmax",
        fail_on_second_softmax,
    )

    with pytest.raises(FloatingPointError, match="forced late numerical failure"):
        writer.fit_base_anchored_episode(
            (1, 2),
            ((0.2, 0.5, 0.3), (0.4, 0.1, 0.5)),
            (),
            prefix=(0,),
        )

    assert calls == 2
    assert writer.fingerprint() == before
    assert writer.episodes_seen == writer.symbols_seen == 0


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"gain": -1.0}, "gain"),
        ({"gain": float("nan")}, "gain"),
        ({"residual_clip": 0.0}, "residual_clip"),
        ({"residual_clip": float("inf")}, "residual_clip"),
    ),
)
def test_base_anchored_inference_rejects_invalid_controls(
    kwargs: dict[str, float], message: str
) -> None:
    writer = ContextCoupledByteWriter("local", alphabet_size=2, table_bits=5)
    with pytest.raises(ValueError, match=message):
        writer.base_anchored_distribution((0.5, 0.5), (), (), **kwargs)
