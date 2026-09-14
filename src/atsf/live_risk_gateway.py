from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite

from .live_authorization import LiveAuthorization, LiveAuthorizationDecision


@dataclass(frozen=True)
class LiveRiskPolicy:
    """Independent runtime limits applied after human authorization."""

    max_capital_fraction: float = 0.10
    max_loss_fraction: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.max_capital_fraction <= 1.0:
            raise ValueError("max_capital_fraction must be greater than 0 and at most 1")
        if not 0.0 < self.max_loss_fraction <= self.max_capital_fraction:
            raise ValueError("max_loss_fraction must be positive and within capital limit")


@dataclass(frozen=True)
class LiveRiskCertificate:
    """Tamper-evident, bounded authority presented to a future live gateway."""

    strategy_id: str
    capital_fraction: float
    max_loss_fraction: float
    issued_at: float
    expires_at: float
    nonce: str
    fingerprint: str


def _fingerprint(
    strategy_id: str,
    capital_fraction: float,
    max_loss_fraction: float,
    issued_at: float,
    expires_at: float,
    nonce: str,
) -> str:
    payload = {
        "capital_fraction": capital_fraction,
        "expires_at": expires_at,
        "issued_at": issued_at,
        "max_loss_fraction": max_loss_fraction,
        "nonce": nonce,
        "strategy_id": strategy_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def issue_live_risk_certificate(
    authorization: LiveAuthorization,
    *,
    issued_at: float,
    expires_at: float,
    nonce: str,
    policy: LiveRiskPolicy | None = None,
) -> LiveRiskCertificate:
    """Convert bounded authorization into a short-lived, tamper-evident certificate."""
    policy = policy or LiveRiskPolicy()
    if authorization.decision is not LiveAuthorizationDecision.AUTHORIZED:
        raise ValueError("live authorization is required")
    if not authorization.execution_authority:
        raise ValueError("live authorization has no execution authority")
    if not authorization.strategy_id.strip():
        raise ValueError("strategy_id is required")
    if not nonce.strip():
        raise ValueError("nonce is required")
    if not isfinite(issued_at) or not isfinite(expires_at) or expires_at <= issued_at:
        raise ValueError("certificate timestamps must be finite and expire after issuance")
    if not isfinite(authorization.capital_fraction) or authorization.capital_fraction <= 0:
        raise ValueError("authorized capital fraction must be finite and positive")
    if authorization.capital_fraction > policy.max_capital_fraction:
        raise ValueError("authorized capital exceeds live risk policy")
    certificate = LiveRiskCertificate(
        strategy_id=authorization.strategy_id,
        capital_fraction=authorization.capital_fraction,
        max_loss_fraction=policy.max_loss_fraction,
        issued_at=issued_at,
        expires_at=expires_at,
        nonce=nonce,
        fingerprint="",
    )
    return LiveRiskCertificate(
        **{**certificate.__dict__, "fingerprint": _fingerprint(**{k: getattr(certificate, k) for k in (
            "strategy_id", "capital_fraction", "max_loss_fraction", "issued_at", "expires_at", "nonce"
        )})}
    )


def validate_live_risk_certificate(
    certificate: LiveRiskCertificate,
    *,
    strategy_id: str,
    now: float,
    requested_capital_fraction: float,
    kill_switch_engaged: bool = False,
    policy: LiveRiskPolicy | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Perform independent pre-trade checks; any failed check blocks execution."""
    policy = policy or LiveRiskPolicy()
    reasons: list[str] = []
    if kill_switch_engaged:
        reasons.append("live kill switch is engaged")
    if not strategy_id.strip() or certificate.strategy_id != strategy_id:
        reasons.append("strategy identity mismatch")
    if not isfinite(now) or now < certificate.issued_at or now >= certificate.expires_at:
        reasons.append("live risk certificate is expired or not yet valid")
    expected = _fingerprint(
        certificate.strategy_id,
        certificate.capital_fraction,
        certificate.max_loss_fraction,
        certificate.issued_at,
        certificate.expires_at,
        certificate.nonce,
    )
    if certificate.fingerprint != expected:
        reasons.append("live risk certificate fingerprint mismatch")
    if not isfinite(requested_capital_fraction) or requested_capital_fraction <= 0:
        reasons.append("requested capital fraction must be finite and positive")
    elif requested_capital_fraction > certificate.capital_fraction:
        reasons.append("requested capital exceeds certificate allocation")
    if certificate.max_loss_fraction > policy.max_loss_fraction:
        reasons.append("certificate loss limit exceeds current risk policy")
    return not reasons, tuple(dict.fromkeys(reasons))
