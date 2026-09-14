from dataclasses import replace

import pytest

from atsf.live_authorization import LiveAuthorization, LiveAuthorizationDecision
from atsf.live_execution_intent import build_execution_intent, verify_execution_intent
from atsf.live_risk_gateway import issue_live_risk_certificate


def certificate():
    authorization = LiveAuthorization(
        strategy_id="strategy-1",
        decision=LiveAuthorizationDecision.AUTHORIZED,
        execution_authority=True,
        capital_fraction=0.05,
        reasons=(),
    )
    return issue_live_risk_certificate(
        authorization, issued_at=100.0, expires_at=200.0, nonce="nonce-1"
    )


def test_intent_is_deterministic_and_tamper_evident():
    first = build_execution_intent(
        certificate(), intent_id="intent-1", symbol="SPY", side="buy",
        quantity_fraction=0.02, now=150.0
    )
    second = build_execution_intent(
        certificate(), intent_id="intent-1", symbol="SPY", side="BUY",
        quantity_fraction=0.02, now=150.0
    )
    assert first == second
    assert verify_execution_intent(first)
    assert not verify_execution_intent(replace(first, quantity_fraction=0.03))


def test_intent_rejects_invalid_side():
    with pytest.raises(ValueError, match="side must be"):
        build_execution_intent(
            certificate(), intent_id="intent-1", symbol="SPY", side="HOLD",
            quantity_fraction=0.02, now=150.0
        )


def test_intent_respects_risk_gateway():
    with pytest.raises(ValueError, match="requested capital exceeds certificate allocation"):
        build_execution_intent(
            certificate(), intent_id="intent-1", symbol="SPY", side="BUY",
            quantity_fraction=0.06, now=150.0
        )


def test_intent_respects_kill_switch():
    with pytest.raises(ValueError, match="kill switch"):
        build_execution_intent(
            certificate(), intent_id="intent-1", symbol="SPY", side="BUY",
            quantity_fraction=0.02, now=150.0, kill_switch_engaged=True
        )
