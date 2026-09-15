from dataclasses import replace

import pytest

from atsf.live_authorization import LiveAuthorization, LiveAuthorizationDecision
from atsf.live_execution_intent import build_execution_intent
from atsf.live_path_audit import audit_live_path
from atsf.live_risk_gateway import issue_live_risk_certificate
from atsf.monitoring import DegradationReport
from atsf.paper_qualification import PaperQualification, PaperQualificationDecision


def _context():
    qualification = PaperQualification(
        strategy_id="strategy-1",
        decision=PaperQualificationDecision.ELIGIBLE_FOR_LIVE_REVIEW,
        execution_authority=False,
        report=DegradationReport(observations=60, total_return=0.10),
        reasons=(),
    )
    authorization = LiveAuthorization(
        strategy_id="strategy-1",
        decision=LiveAuthorizationDecision.AUTHORIZED,
        execution_authority=True,
        capital_fraction=0.05,
        reasons=(),
    )
    certificate = issue_live_risk_certificate(
        authorization,
        issued_at=100.0,
        expires_at=200.0,
        nonce="nonce-1",
    )
    intent = build_execution_intent(
        certificate,
        intent_id="intent-1",
        symbol="SPY",
        side="BUY",
        quantity_fraction=0.02,
        now=150.0,
    )
    return qualification, authorization, certificate, intent


def test_complete_live_path_passes_but_audit_never_grants_authority():
    result = audit_live_path(*_context())
    assert result.passed is True
    assert result.execution_authority is False
    assert result.failures == ()
    assert len(result.fingerprint) == 64


def test_audit_rejects_identity_mismatch():
    qualification, authorization, certificate, intent = _context()
    mismatched = replace(authorization, strategy_id="other-strategy")
    result = audit_live_path(qualification, mismatched, certificate, intent)
    assert result.passed is False
    assert "authorization strategy identity mismatch" in result.failures
    assert result.execution_authority is False


def test_audit_rejects_tampered_certificate():
    qualification, authorization, certificate, intent = _context()
    tampered = replace(certificate, capital_fraction=0.09)
    result = audit_live_path(qualification, authorization, tampered, intent)
    assert result.passed is False
    assert "risk certificate fingerprint mismatch" in result.failures


def test_audit_rejects_tampered_intent():
    qualification, authorization, certificate, intent = _context()
    tampered = replace(intent, quantity_fraction=0.04)
    result = audit_live_path(qualification, authorization, certificate, tampered)
    assert result.passed is False
    assert "execution intent fingerprint mismatch" in result.failures


def test_audit_rejects_non_authorized_qualification():
    _, authorization, certificate, intent = _context()
    rejected = PaperQualification(
        strategy_id="strategy-1",
        decision=PaperQualificationDecision.REJECT,
        execution_authority=False,
        report=DegradationReport(observations=60),
        reasons=("failed",),
    )
    result = audit_live_path(rejected, authorization, certificate, intent)
    assert result.passed is False
    assert "paper qualification is not eligible for live review" in result.failures


def test_audit_fingerprint_is_deterministic():
    context = _context()
    first = audit_live_path(*context)
    second = audit_live_path(*context)
    assert first == second
