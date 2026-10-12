from __future__ import annotations

from manifoldflow.dendritic_basin.risk_permission import RiskSensitivePermissionBank


def test_positive_low_variance_class_leaves_probation() -> None:
    bank = RiskSensitivePermissionBank(minimum_evidence=4, confidence_width=1.0)
    key = (1, 2, 3)
    for utility in (0.2, 0.2, 0.2, 0.2):
        bank.update(key, utility)

    decision = bank.decide(key)

    assert decision.permitted
    assert decision.lower_confidence_bound == 0.2


def test_helpful_majority_with_catastrophic_tail_stays_shadowed() -> None:
    bank = RiskSensitivePermissionBank(
        minimum_evidence=4,
        confidence_width=0.0,
        catastrophic_threshold=0.1,
        maximum_catastrophic_rate=0.2,
    )
    key = (4, 5, 6)
    for utility in (0.1, 0.1, 0.1, -1.0):
        bank.update(key, utility)

    decision = bank.decide(key)

    assert not decision.permitted
    assert decision.catastrophic_rate == 0.25


def test_unobserved_class_is_never_permitted() -> None:
    bank = RiskSensitivePermissionBank()

    decision = bank.decide((9,))

    assert not decision.permitted
    assert decision.observations == 0
