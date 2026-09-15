from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .live_authorization import LiveAuthorization, LiveAuthorizationDecision
from .live_execution_intent import LiveExecutionIntent, verify_execution_intent
from .live_risk_gateway import LiveRiskCertificate, _fingerprint
from .paper_qualification import PaperQualification, PaperQualificationDecision


@dataclass(frozen=True)
class LivePathAudit:
    """Fail-closed structural audit of the research-to-live control path.

    This is evidence only. It can never grant execution authority.
    """

    strategy_id: str
    passed: bool
    execution_authority: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    fingerprint: str


def _audit_fingerprint(strategy_id: str, checks: tuple[str, ...], failures: tuple[str, ...]) -> str:
    payload = {"checks": checks, "failures": failures, "strategy_id": strategy_id}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def audit_live_path(
    qualification: PaperQualification,
    authorization: LiveAuthorization,
    certificate: LiveRiskCertificate,
    intent: LiveExecutionIntent,
) -> LivePathAudit:
    """Verify identity, lifecycle-boundary, certificate, and intent integrity.

    The audit intentionally returns ``execution_authority=False`` even when every
    check passes. Only ``authorize_live_strategy`` may produce live authority.
    """
    checks: list[str] = []
    failures: list[str] = []

    def check(name: str, condition: bool, failure: str) -> None:
        if condition:
            checks.append(name)
        else:
            failures.append(failure)

    strategy_id = qualification.strategy_id
    check("strategy_identity", bool(strategy_id.strip()), "strategy_id is required")
    check("paper_qualification", qualification.decision is PaperQualificationDecision.ELIGIBLE_FOR_LIVE_REVIEW,
          "paper qualification is not eligible for live review")
    check("qualification_no_authority", qualification.execution_authority is False,
          "qualification layer must not possess execution authority")
    check("authorization_identity", authorization.strategy_id == strategy_id,
          "authorization strategy identity mismatch")
    check("authorization_decision", authorization.decision is LiveAuthorizationDecision.AUTHORIZED,
          "live authorization is not authorized")
    check("authorization_authority", authorization.execution_authority is True,
          "authorized decision lacks execution authority")
    check("certificate_identity", certificate.strategy_id == strategy_id,
          "risk certificate strategy identity mismatch")
    expected_certificate = _fingerprint(
        certificate.strategy_id,
        certificate.capital_fraction,
        certificate.max_loss_fraction,
        certificate.issued_at,
        certificate.expires_at,
        certificate.nonce,
    )
    check("certificate_integrity", certificate.fingerprint == expected_certificate,
          "risk certificate fingerprint mismatch")
    check("intent_identity", intent.strategy_id == strategy_id,
          "execution intent strategy identity mismatch")
    check("intent_certificate_binding", intent.certificate_fingerprint == certificate.fingerprint,
          "execution intent is not bound to the risk certificate")
    check("intent_integrity", verify_execution_intent(intent),
          "execution intent fingerprint mismatch")

    passed = not failures
    check_names = tuple(checks)
    failure_names = tuple(dict.fromkeys(failures))
    return LivePathAudit(
        strategy_id=strategy_id,
        passed=passed,
        execution_authority=False,
        checks=check_names,
        failures=failure_names,
        fingerprint=_audit_fingerprint(strategy_id, check_names, failure_names),
    )
