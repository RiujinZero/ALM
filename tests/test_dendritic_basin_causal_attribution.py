from __future__ import annotations

from manifoldflow.dendritic_basin.causal_attribution import SelfCausalAttribution


def test_large_effect_favors_self_when_self_scale_is_larger() -> None:
    attribution = SelfCausalAttribution(self_prior_scale=1.0, other_prior_scale=0.05)

    posterior = attribution.posterior_self(d=0.8)

    assert posterior > 0.5


def test_small_effect_favors_other_when_other_scale_is_small() -> None:
    attribution = SelfCausalAttribution(self_prior_scale=1.0, other_prior_scale=0.05)

    posterior = attribution.posterior_self(d=0.01)

    assert posterior < 0.5


def test_zero_evidence_reduces_to_the_prior_ratio_of_densities() -> None:
    attribution = SelfCausalAttribution(
        prior_self=0.5, self_prior_scale=1.0, other_prior_scale=1.0
    )
    # Equal scales and equal prior at d=0 must give exactly 0.5.
    assert abs(attribution.posterior_self(d=0.0) - 0.5) < 1e-12


def test_confirmed_evidence_refines_the_learned_scale_and_shifts_posterior() -> None:
    attribution = SelfCausalAttribution(self_prior_scale=1.0, other_prior_scale=1.0)
    before = attribution.posterior_self(d=0.05)

    for _ in range(50):
        attribution.confirm(0.01, cause="other")

    after = attribution.posterior_self(d=0.05)

    # A tightly learned "other" scale (RMS ~0.01) makes d=0.05 sit five
    # scales out under "other" -- far less likely there than under the
    # still-broad "self" scale, so the posterior should shift toward self.
    assert after > before


def test_prior_self_shifts_the_posterior_monotonically() -> None:
    attribution = SelfCausalAttribution(self_prior_scale=1.0, other_prior_scale=1.0)

    low_prior = attribution.posterior_self(d=0.3, prior_self=0.1)
    high_prior = attribution.posterior_self(d=0.3, prior_self=0.9)

    assert low_prior < high_prior


def test_rejects_invalid_construction_and_confirmation() -> None:
    try:
        SelfCausalAttribution(prior_self=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for prior outside (0, 1)")

    attribution = SelfCausalAttribution()
    try:
        attribution.confirm(0.1, cause="neither")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown cause")
