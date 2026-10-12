"""Risk-sensitive local permission for long-range cortical interventions."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass
class PermissionStatistics:
    observations: int = 0
    mean_utility: float = 0.0
    utility_m2: float = 0.0
    catastrophic_events: int = 0

    def update(self, utility_bits: float, *, catastrophic_threshold: float) -> None:
        self.observations += 1
        delta = utility_bits - self.mean_utility
        self.mean_utility += delta / self.observations
        self.utility_m2 += delta * (utility_bits - self.mean_utility)
        if utility_bits < -catastrophic_threshold:
            self.catastrophic_events += 1

    @property
    def variance(self) -> float:
        return (
            self.utility_m2 / (self.observations - 1) if self.observations > 1 else 0.0
        )

    @property
    def catastrophic_rate(self) -> float:
        return self.catastrophic_events / max(1, self.observations)


@dataclass(frozen=True)
class PermissionDecision:
    permitted: bool
    observations: int
    mean_utility: float
    standard_deviation: float
    lower_confidence_bound: float
    catastrophic_rate: float
    conservative_value: float


class RiskSensitivePermissionBank:
    """Let experienced low-risk intervention classes leave shadow probation."""

    def __init__(
        self,
        *,
        minimum_evidence: int = 16,
        confidence_width: float = 2.0,
        catastrophic_threshold: float = 0.1,
        tail_risk_penalty: float = 0.5,
        maximum_catastrophic_rate: float = 0.05,
    ) -> None:
        if minimum_evidence < 1:
            raise ValueError("minimum_evidence must be positive")
        if confidence_width < 0.0 or catastrophic_threshold <= 0.0:
            raise ValueError("confidence and catastrophe scales are invalid")
        if tail_risk_penalty < 0.0:
            raise ValueError("tail risk penalty must be non-negative")
        if not 0.0 <= maximum_catastrophic_rate <= 1.0:
            raise ValueError("maximum catastrophic rate must be in [0, 1]")
        self.minimum_evidence = minimum_evidence
        self.confidence_width = confidence_width
        self.catastrophic_threshold = catastrophic_threshold
        self.tail_risk_penalty = tail_risk_penalty
        self.maximum_catastrophic_rate = maximum_catastrophic_rate
        self.statistics: dict[tuple[int, ...], PermissionStatistics] = {}

    def update(self, key: tuple[int, ...], utility_bits: float) -> None:
        self.statistics.setdefault(key, PermissionStatistics()).update(
            utility_bits,
            catastrophic_threshold=self.catastrophic_threshold,
        )

    def decide(self, key: tuple[int, ...]) -> PermissionDecision:
        stats = self.statistics.get(key, PermissionStatistics())
        deviation = sqrt(max(0.0, stats.variance))
        uncertainty = deviation / sqrt(max(1, stats.observations))
        lower_bound = stats.mean_utility - self.confidence_width * uncertainty
        conservative = lower_bound - self.tail_risk_penalty * stats.catastrophic_rate
        permitted = (
            stats.observations >= self.minimum_evidence
            and conservative > 0.0
            and stats.catastrophic_rate <= self.maximum_catastrophic_rate
        )
        return PermissionDecision(
            permitted,
            stats.observations,
            stats.mean_utility,
            deviation,
            lower_bound,
            stats.catastrophic_rate,
            conservative,
        )
