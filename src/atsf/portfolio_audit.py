from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from numbers import Real


@dataclass(frozen=True)
class PortfolioAuditEvent:
    sequence: int
    strategy_id: str
    action: str
    timestamp: str
    quantity: float
    price: float
    fee: float


def audit_event_id(event: PortfolioAuditEvent) -> str:
    """Return a stable identifier for a validated immutable execution event."""
    if isinstance(event.sequence, bool) or not isinstance(event.sequence, int) or event.sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if (
        not isinstance(event.strategy_id, str)
        or not event.strategy_id.strip()
        or not isinstance(event.action, str)
        or event.action not in {"buy", "sell"}
        or not isinstance(event.timestamp, str)
        or not event.timestamp.strip()
    ):
        raise ValueError("strategy_id, supported action, and timestamp are required")
    try:
        timestamp = datetime.fromisoformat(event.timestamp)
    except ValueError as exc:
        raise ValueError("timestamp must be a valid ISO 8601 timestamp") from exc
    if timestamp.tzinfo is not None:
        raise ValueError("timestamp must be timezone-naive")
    values = (event.quantity, event.price, event.fee)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("quantity, price, and fee must be numeric")
    if not all(isfinite(value) for value in values):
        raise ValueError("quantity, price, and fee must be finite")
    if event.quantity < 0 or event.price <= 0 or event.fee < 0:
        raise ValueError("quantity must be non-negative, price positive, and fee non-negative")
    payload = {
        "sequence": event.sequence,
        "strategy_id": event.strategy_id,
        "action": event.action,
        "timestamp": event.timestamp,
        "quantity": event.quantity,
        "price": event.price,
        "fee": event.fee,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:16]
