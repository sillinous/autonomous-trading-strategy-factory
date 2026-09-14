from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite

from .live_risk_gateway import LiveRiskCertificate, validate_live_risk_certificate


@dataclass(frozen=True)
class LiveExecutionIntent:
    """Immutable order intent; it is not a broker submission."""

    intent_id: str
    strategy_id: str
    symbol: str
    side: str
    quantity_fraction: float
    created_at: float
    certificate_fingerprint: str
    fingerprint: str


def _intent_fingerprint(
    intent_id: str,
    strategy_id: str,
    symbol: str,
    side: str,
    quantity_fraction: float,
    created_at: float,
    certificate_fingerprint: str,
) -> str:
    payload = {
        "certificate_fingerprint": certificate_fingerprint,
        "created_at": created_at,
        "intent_id": intent_id,
        "quantity_fraction": quantity_fraction,
        "side": side,
        "strategy_id": strategy_id,
        "symbol": symbol,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def build_execution_intent(
    certificate: LiveRiskCertificate,
    *,
    intent_id: str,
    symbol: str,
    side: str,
    quantity_fraction: float,
    now: float,
    kill_switch_engaged: bool = False,
) -> LiveExecutionIntent:
    """Create an intent only after the independent risk gateway admits the request."""
    if not intent_id.strip() or not symbol.strip():
        raise ValueError("intent_id and symbol are required")
    normalized_side = side.upper()
    if normalized_side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if not isfinite(quantity_fraction) or quantity_fraction <= 0:
        raise ValueError("quantity_fraction must be finite and positive")
    allowed, reasons = validate_live_risk_certificate(
        certificate,
        strategy_id=certificate.strategy_id,
        now=now,
        requested_capital_fraction=quantity_fraction,
        kill_switch_engaged=kill_switch_engaged,
    )
    if not allowed:
        raise ValueError("live execution intent rejected: " + "; ".join(reasons))
    intent = LiveExecutionIntent(
        intent_id=intent_id,
        strategy_id=certificate.strategy_id,
        symbol=symbol,
        side=normalized_side,
        quantity_fraction=quantity_fraction,
        created_at=now,
        certificate_fingerprint=certificate.fingerprint,
        fingerprint="",
    )
    return replace(
        intent,
        fingerprint=_intent_fingerprint(
            intent.intent_id,
            intent.strategy_id,
            intent.symbol,
            intent.side,
            intent.quantity_fraction,
            intent.created_at,
            intent.certificate_fingerprint,
        ),
    )


def verify_execution_intent(intent: LiveExecutionIntent) -> bool:
    """Verify the intent has not been modified since creation."""
    return intent.fingerprint == _intent_fingerprint(
        intent.intent_id,
        intent.strategy_id,
        intent.symbol,
        intent.side,
        intent.quantity_fraction,
        intent.created_at,
        intent.certificate_fingerprint,
    )
