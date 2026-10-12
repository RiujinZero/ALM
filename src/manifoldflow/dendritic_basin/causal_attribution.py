"""Bayesian estimate of "did my own recalled activity cause this outcome?"

Every gating rule tried so far for holding an internally emitted assembly as
persistent apical context (the successor-held rule, the relation-conditioned
ablation) asked only "is there support for this content" or "is this the
globally strongest relation." Neither is an estimate of authorship: whether
the *observed effect on generation* is better explained by this organism's
own recalled activity than by everything else that could have produced it
(ordinary cortical drift, coincidence, an unrelated but active relation).

This module adds that missing quantity as a small, local, streaming Bayesian
update over two candidate causes, ``self`` and ``other``:

    P(C=self | d) = p(d | C=self) P(C=self)
                    -------------------------------------------
                    sum_c p(d | C=c) P(C=c)

``d`` is a non-negative evidence scalar (here: the total-variation distance
between generation with the candidate internal contribution included and
generation with it withheld -- the same quantity already computed by
``dendritic_basin_shared_cortical_assembly_smoke.py`` and the autonomous
internal-evidence closure experiments). Each candidate cause has its own
half-normal likelihood over ``d``, parameterized by a scale learned online
from confirmed cases (a case is "confirmed" only when a later externally
observed byte validates or refutes the candidate's contribution -- exactly
the project-wide invariant that recalled/generated activity may never
confirm itself).

This is a generic two-hypothesis causal-inference account of attribution,
not a claim about the neural implementation of a sense of agency.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi, sqrt


@dataclass
class _ScaleEstimate:
    """Running half-normal scale (RMS of confirmed evidence) for one cause."""

    sum_squares: float = 0.0
    count: int = 0
    prior_scale: float = 1.0

    def update(self, magnitude: float) -> None:
        self.sum_squares += float(magnitude) ** 2
        self.count += 1

    @property
    def scale(self) -> float:
        if self.count == 0:
            return self.prior_scale
        return sqrt(self.sum_squares / self.count) or self.prior_scale


def _half_normal_log_density(d: float, scale: float) -> float:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if d < 0.0:
        raise ValueError("evidence must be non-negative")
    return (
        0.5 * log(2.0 / pi)
        - log(scale)
        - (d * d) / (2.0 * scale * scale)
    )


class SelfCausalAttribution:
    """Streaming two-cause Bayesian attribution over an effect-size scalar."""

    def __init__(
        self,
        *,
        prior_self: float = 0.5,
        self_prior_scale: float = 1.0,
        other_prior_scale: float = 0.1,
    ) -> None:
        if not 0.0 < prior_self < 1.0:
            raise ValueError("prior_self must be in (0, 1)")
        if self_prior_scale <= 0.0 or other_prior_scale <= 0.0:
            raise ValueError("prior scales must be positive")
        self.prior_self = float(prior_self)
        self.self_scale = _ScaleEstimate(prior_scale=self_prior_scale)
        self.other_scale = _ScaleEstimate(prior_scale=other_prior_scale)
        self.observations = 0
        self.confirmed_self = 0
        self.confirmed_other = 0

    def posterior_self(self, d: float, *, prior_self: float | None = None) -> float:
        """Return ``P(C=self | d)`` under the current learned scales."""
        prior = self.prior_self if prior_self is None else float(prior_self)
        if not 0.0 < prior < 1.0:
            raise ValueError("prior must be in (0, 1)")
        d = max(0.0, float(d))
        log_self = log(prior) + _half_normal_log_density(d, self.self_scale.scale)
        log_other = log(1.0 - prior) + _half_normal_log_density(
            d, self.other_scale.scale
        )
        # log-sum-exp for numerical stability
        peak = max(log_self, log_other)
        total = exp(log_self - peak) + exp(log_other - peak)
        return exp(log_self - peak) / total

    def confirm(self, d: float, *, cause: str) -> None:
        """Refine a cause's learned scale from one externally confirmed case.

        Only call this once a real external consequence has validated (or
        refuted) which cause actually produced ``d``; internal or generated
        activity may never supply its own confirmation.
        """
        if cause not in {"self", "other"}:
            raise ValueError("cause must be 'self' or 'other'")
        d = max(0.0, float(d))
        self.observations += 1
        if cause == "self":
            self.self_scale.update(d)
            self.confirmed_self += 1
        else:
            self.other_scale.update(d)
            self.confirmed_other += 1

    @property
    def embodied_bytes(self) -> int:
        import pickle

        return len(
            pickle.dumps(
                (self.self_scale, self.other_scale, self.prior_self),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )
