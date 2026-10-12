from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from manifoldflow.dendritic_basin.completion_potential import (
    CompletionLeafError,
    ExactCompletionPotential,
    OffTriePrefixError,
    PrefixStatus,
    ZeroAcceptedMassError,
)


def _distribution(
    alphabet_size: int, probabilities: dict[int, float]
) -> np.ndarray:
    result = np.zeros(alphabet_size, dtype=np.float64)
    for symbol, probability in probabilities.items():
        result[symbol] = probability
    assert np.isclose(result.sum(), 1.0)
    return result


def _enumeration_fixture() -> tuple[
    tuple[bytes, ...], dict[bytes, np.ndarray]
]:
    # Base accepted path probabilities are .10, .20, and .03 respectively.
    completions = (bytes((0, 1)), bytes((0, 2)), bytes((1, 3, 4)))
    distributions = {
        b"": _distribution(10, {0: 0.4, 1: 0.3, 9: 0.3}),
        bytes((0,)): _distribution(10, {1: 0.25, 2: 0.5, 8: 0.25}),
        bytes((1,)): _distribution(10, {3: 0.5, 7: 0.5}),
        bytes((1, 3)): _distribution(10, {4: 0.2, 6: 0.8}),
    }
    return completions, distributions


def _path_probability(
    potential: ExactCompletionPotential, sequence: bytes, *, guided: bool
) -> float:
    probability = 1.0
    for index, byte in enumerate(sequence):
        prefix = sequence[:index]
        distribution = (
            potential.distribution(prefix)
            if guided
            else potential.base_distribution(prefix)
        )
        probability *= float(distribution[byte])
    return probability


def test_exact_masses_normalization_and_telescoping_match_enumeration() -> None:
    completions, distributions = _enumeration_fixture()
    potential = ExactCompletionPotential.from_prefix_distributions(
        completions, distributions, alphabet_size=10
    )

    assert potential.accepted_mass == pytest.approx(0.33)
    assert potential.accepted_log_mass == pytest.approx(math.log(0.33))
    assert potential.mass(bytes((0,))) == pytest.approx(0.75)
    assert potential.mass(bytes((1,))) == pytest.approx(0.1)
    assert potential.mass(bytes((1, 3))) == pytest.approx(0.2)
    for leaf in completions:
        assert potential.log_mass(leaf) == 0.0
        assert potential.mass(leaf) == 1.0

    for prefix in potential.internal_prefixes:
        guided = potential.distribution(prefix)
        assert guided.sum() == pytest.approx(1.0)
        assert np.all(guided >= 0.0)
        base = potential.base_distribution(prefix)
        successor_h = np.exp(potential.successor_log_masses(prefix))
        numerator = base * successor_h
        np.testing.assert_allclose(
            guided,
            numerator / potential.mass(prefix),
            rtol=1e-14,
            atol=1e-15,
        )

    for sequence in completions:
        expected = _path_probability(potential, sequence, guided=False) / 0.33
        assert _path_probability(potential, sequence, guided=True) == pytest.approx(
            expected
        )
    assert sum(
        _path_probability(potential, sequence, guided=True)
        for sequence in completions
    ) == pytest.approx(1.0)


def test_exact_transform_preserves_relative_probability_of_valid_paths() -> None:
    completions, distributions = _enumeration_fixture()
    potential = ExactCompletionPotential.from_prefix_distributions(
        completions, distributions, alphabet_size=10
    )
    first, second = completions[:2]

    base_ratio = _path_probability(
        potential, first, guided=False
    ) / _path_probability(potential, second, guided=False)
    guided_ratio = _path_probability(
        potential, first, guided=True
    ) / _path_probability(potential, second, guided=True)

    assert guided_ratio == pytest.approx(base_ratio)
    assert guided_ratio == pytest.approx(0.5)


def test_compile_is_canonical_and_callback_is_called_once_per_internal_prefix() -> None:
    completions, distributions = _enumeration_fixture()
    calls: list[bytes] = []

    def callback(prefix: bytes) -> np.ndarray:
        calls.append(prefix)
        return distributions[prefix]

    compiled = ExactCompletionPotential.compile(
        reversed(completions), callback, alphabet_size=10
    )
    imported = ExactCompletionPotential.from_prefix_distributions(
        completions, dict(reversed(tuple(distributions.items()))), alphabet_size=10
    )

    assert calls == list(compiled.internal_prefixes)
    assert compiled.accepted_sequences == tuple(sorted(completions))
    assert compiled.fingerprint() == imported.fingerprint()
    assert len(compiled.fingerprint()) == 64
    assert compiled.diagnostics.accepted_sequence_count == 3
    assert compiled.diagnostics.internal_prefix_count == 4
    assert compiled.diagnostics.leaf_count == 3
    assert compiled.diagnostics.maximum_depth == 3
    assert compiled.diagnostics.zero_mass_internal_prefix_count == 0
    assert compiled.diagnostics.root_mass == pytest.approx(0.33)


@pytest.mark.parametrize(
    ("sequences", "message"),
    (
        ((), "must be non-empty"),
        ((b"",), "must be non-empty"),
        ((b"a", b"a"), "duplicate"),
        ((b"a", b"ab"), "prefix-free"),
        ((b"ab", b"a"), "prefix-free"),
    ),
)
def test_accepted_sequences_must_be_nonempty_unique_and_prefix_free(
    sequences: Sequence[bytes], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ExactCompletionPotential.compile(
            sequences, lambda _prefix: _distribution(256, {0: 1.0})
        )


def test_byte_and_distribution_validation_is_eager() -> None:
    with pytest.raises(TypeError, match="iterable of byte sequences"):
        ExactCompletionPotential.compile(
            b"single", lambda _prefix: _distribution(256, {0: 1.0})
        )
    with pytest.raises(ValueError, match="outside the alphabet"):
        ExactCompletionPotential.compile(
            (bytes((2,)),),
            lambda _prefix: _distribution(2, {0: 1.0}),
            alphabet_size=2,
        )
    with pytest.raises(ValueError, match="sum to one"):
        ExactCompletionPotential.compile(
            (b"a",), lambda _prefix: np.zeros(256)
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        ExactCompletionPotential.compile(
            (b"a",), lambda _prefix: np.full(256, np.nan)
        )


def test_import_requires_exactly_all_internal_prefix_distributions() -> None:
    completions, distributions = _enumeration_fixture()
    missing = dict(distributions)
    del missing[bytes((1, 3))]
    extra = dict(distributions)
    extra[b"off"] = _distribution(10, {0: 1.0})

    with pytest.raises(ValueError, match="missing"):
        ExactCompletionPotential.from_prefix_distributions(
            completions, missing, alphabet_size=10
        )
    with pytest.raises(ValueError, match="extra"):
        ExactCompletionPotential.from_prefix_distributions(
            completions, extra, alphabet_size=10
        )


def test_gain_zero_is_exact_base_bypass_and_zero_support_stays_zero() -> None:
    completions = (bytes((0,)), bytes((1,)))
    base = _distribution(4, {1: 0.6, 3: 0.4})
    potential = ExactCompletionPotential.from_prefix_distributions(
        completions, {b"": base}, alphabet_size=4
    )

    bypass = potential.distribution(b"", gain=0.0)
    exact = potential.distribution(b"")

    np.testing.assert_array_equal(bypass, base)
    np.testing.assert_array_equal(exact, np.asarray((0.0, 1.0, 0.0, 0.0)))
    assert exact[0] == 0.0  # Accepted, but absent from native support.
    assert exact[2] == 0.0
    assert exact[3] == 0.0  # Native support, but not an accepted successor.


def test_false_positive_floor_leaks_only_where_base_has_support() -> None:
    base = _distribution(4, {0: 0.8, 1: 0.2})
    potential = ExactCompletionPotential.from_prefix_distributions(
        (bytes((0,)),), {b"": base}, alphabet_size=4
    )

    exact = potential.distribution(b"")
    stressed = potential.distribution(b"", false_positive_floor=0.1)
    bypass = potential.distribution(
        b"", gain=0.0, false_positive_floor=0.1
    )

    np.testing.assert_array_equal(exact, np.asarray((1.0, 0.0, 0.0, 0.0)))
    np.testing.assert_allclose(stressed, np.asarray((0.8, 0.02, 0.0, 0.0)) / 0.82)
    np.testing.assert_array_equal(bypass, base)
    assert stressed[1] > 0.0
    assert stressed[2] == stressed[3] == 0.0


def test_positive_floor_continues_off_trie_without_fabricated_support() -> None:
    potential = ExactCompletionPotential.from_prefix_distributions(
        (bytes((0,)),),
        {b"": _distribution(4, {0: 0.5, 1: 0.5})},
        alphabet_size=4,
    )
    fallback = _distribution(4, {1: 0.25, 3: 0.75})

    with pytest.raises(OffTriePrefixError, match="outside"):
        potential.distribution(bytes((1,)), off_trie_base_distribution=fallback)
    with pytest.raises(OffTriePrefixError, match="requires"):
        potential.distribution(bytes((1,)), false_positive_floor=1e-3)

    continued = potential.distribution(
        bytes((1,)),
        false_positive_floor=1e-3,
        off_trie_base_distribution=fallback,
    )
    np.testing.assert_array_equal(continued, fallback)
    assert continued[0] == continued[2] == 0.0


def test_zero_accepted_mass_and_floor_recovery_use_only_native_support() -> None:
    base = _distribution(3, {2: 1.0})
    potential = ExactCompletionPotential.from_prefix_distributions(
        (bytes((0,)),), {b"": base}, alphabet_size=3
    )

    assert potential.accepted_mass == 0.0
    assert potential.accepted_log_mass == -math.inf
    assert potential.diagnostics.zero_mass_internal_prefix_count == 1
    with pytest.raises(ZeroAcceptedMassError, match="zero"):
        potential.distribution(b"")

    stressed = potential.distribution(b"", false_positive_floor=1e-3)
    np.testing.assert_array_equal(stressed, base)
    assert stressed[0] == 0.0


def test_status_distinguishes_internal_leaf_off_trie_and_invalid_prefix() -> None:
    potential = ExactCompletionPotential.from_prefix_distributions(
        (b"ab",),
        {
            b"": _distribution(256, {ord("a"): 1.0}),
            b"a": _distribution(256, {ord("b"): 1.0}),
        },
    )

    assert potential.status(b"") is PrefixStatus.INTERNAL
    assert potential.status(bytearray(b"a")) is PrefixStatus.INTERNAL
    assert potential.status(memoryview(b"ab")) is PrefixStatus.LEAF
    assert potential.status(b"ax") is PrefixStatus.OFF_TRIE
    with pytest.raises(CompletionLeafError, match="leaf"):
        potential.distribution(b"ab")
    with pytest.raises(CompletionLeafError, match="leaf"):
        potential.base_distribution(b"ab")
    with pytest.raises(OffTriePrefixError, match="outside"):
        potential.log_mass(b"ax")
    with pytest.raises(TypeError, match="byte sequence"):
        potential.status("not bytes")  # type: ignore[arg-type]


def test_byte_at_a_time_sampling_reaches_only_accepted_leaves() -> None:
    completions, distributions = _enumeration_fixture()
    potential = ExactCompletionPotential.from_prefix_distributions(
        completions, distributions, alphabet_size=10
    )
    rng = np.random.default_rng(260_914)
    counts = {sequence: 0 for sequence in completions}

    for _ in range(8_000):
        prefix = b""
        while potential.status(prefix) is PrefixStatus.INTERNAL:
            next_byte = int(rng.choice(10, p=potential.distribution(prefix)))
            prefix += bytes((next_byte,))
        assert potential.status(prefix) is PrefixStatus.LEAF
        counts[prefix] += 1

    observed = np.asarray([counts[sequence] / 8_000 for sequence in completions])
    expected = np.asarray((0.10, 0.20, 0.03)) / 0.33
    np.testing.assert_allclose(observed, expected, atol=0.018, rtol=0.0)


def test_log_space_backward_pass_survives_probability_underflow() -> None:
    depth = 500
    sequence = bytes((0,)) * depth
    base = _distribution(2, {0: 0.01, 1: 0.99})
    potential = ExactCompletionPotential.compile(
        (sequence,), lambda _prefix: base, alphabet_size=2
    )

    assert math.isfinite(potential.accepted_log_mass)
    assert potential.accepted_log_mass == pytest.approx(depth * math.log(0.01))
    assert potential.accepted_mass == 0.0  # Probability space really underflows.
    np.testing.assert_array_equal(potential.distribution(b""), (1.0, 0.0))


@pytest.mark.parametrize("gain", (-1.0, math.inf, math.nan))
def test_invalid_gain_is_rejected(gain: float) -> None:
    potential = ExactCompletionPotential.from_prefix_distributions(
        (bytes((0,)),), {b"": _distribution(2, {0: 1.0})}, alphabet_size=2
    )
    with pytest.raises(ValueError, match="gain"):
        potential.distribution(b"", gain=gain)


@pytest.mark.parametrize("floor", (-0.1, 1.1, math.inf, math.nan))
def test_invalid_false_positive_floor_is_rejected(floor: float) -> None:
    potential = ExactCompletionPotential.from_prefix_distributions(
        (bytes((0,)),), {b"": _distribution(2, {0: 1.0})}, alphabet_size=2
    )
    with pytest.raises(ValueError, match="false_positive_floor"):
        potential.distribution(b"", false_positive_floor=floor)
