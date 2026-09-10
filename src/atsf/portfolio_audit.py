from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite


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
    """Return a stable identifier for an immutable execution event."""
    if event.sequence < 0:
        raise ValueError("sequence must be non-negative")
    if not event.strategy_id or not event.action or not event.timestamp:
        raise ValueError("strategy_id, action, and timestamp are required")
    if not all(isfinite(value) for value in (event.quantity, event.price, event.fee)):
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
