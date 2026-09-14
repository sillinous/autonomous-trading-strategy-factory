from dataclasses import replace

import pytest

from atsf.live_authorization import (
    LiveAuthorization,
    LiveAuthorizationDecision,
)
from atsf.live_risk_gateway import (
    LiveRiskPolicy,
    issue_live_risk_certificate,
    validate_live_risk_certificate,
)


def authorization() -> LiveAuthorization:
    return LiveAuthorization(
        strategy_id="strategy-1",
        decision=LiveAuthorizationDecision.AUTHORIZED,
        execution_authority=True,
        capital_fraction=0.05,
        reasons=(),
    )


def certificate():
    return issue_live_risk_certificate(
        authorization(),
        issued_at=100.0,
        expires_at=200.0,
        nonce="nonce-1",
    )


def test_certificate_is_deterministic_and_bounded() -> None:
    first = certificate()
    second = certificate()
    assert first == second
    assert first.capital_fraction == 0.05
    assert first.max_loss_fraction == 0.02
    assert first.fingerprint


def test_unauthorized_cannot_receive_certificate() -> None:
    rejected = replace(authorization(), decision=LiveAuthorizationDecision.REJECT)
    with pytest.raises(ValueError, match="authorization is required"):
        issue_live_risk_certificate(
            rejected, issued_at=100.0, expires_at=200.0, nonce="nonce-1"
        )


def test_gateway_accepts_valid_certificate() -> None:
    allowed, reasons = validate_live_risk_certificate(
        certificate(),
        strategy_id="strategy-1",
        now=150.0,
        requested_capital_fraction=0.04,
    )
    assert allowed is True
    assert reasons == ()


def test_gateway_rejects_expired_certificate() -> None:
    allowed, reasons = validate_live_risk_certificate(
        certificate(), strategy_id="strategy-1", now=200.0, requested_capital_fraction=0.01
    )
    assert allowed is False
    assert "expired or not yet valid" in reasons[0]


def test_gateway_rejects_kill_switch() -> None:
    allowed, reasons = validate_live_risk_certificate(
        certificate(),
        strategy_id="strategy-1",
        now=150.0,
        requested_capital_fraction=0.01,
        kill_switch_engaged=True,
    )
    assert allowed is False
    assert "kill switch" in reasons[0]


def test_gateway_rejects_strategy_mismatch() -> None:
    allowed, reasons = validate_live_risk_certificate(
        certificate(), strategy_id="strategy-2", now=150.0, requested_capital_fraction=0.01
    )
    assert allowed is False
    assert "identity mismatch" in reasons


def test_gateway_rejects_tampered_certificate() -> None:
    tampered = replace(certificate(), capital_fraction=0.09)
    allowed, reasons = validate_live_risk_certificate(
        tampered,
        strategy_id="strategy-1",
        now=150.0,
        requested_capital_fraction=0.01,
    )
    assert allowed is False
    assert "fingerprint mismatch" in reasons


def test_gateway_rejects_excess_allocation() -> None:
    allowed, reasons = validate_live_risk_certificate(
        certificate(), strategy_id="strategy-1", now=150.0, requested_capital_fraction=0.06
    )
    assert allowed is False
    assert "exceeds certificate allocation" in reasons


def test_gateway_rejects_invalid_current_policy() -> None:
    with pytest.raises(ValueError):
        LiveRiskPolicy(max_loss_fraction=0.11)
