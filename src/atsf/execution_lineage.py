from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .portfolio_audit import PortfolioAuditEvent, audit_event_id


@dataclass(frozen=True)
class FillLineage:
    event_id: str
    decision_id: str
    reason: str


def decision_id(strategy_id: str, timestamp: pd.Timestamp, action: str, reason: str) -> str:
    """Return a deterministic identifier for the decision that authorized a fill."""
    if not strategy_id or not action or not reason:
        raise ValueError("strategy_id, action, and reason are required")
    payload = {
        "strategy_id": strategy_id,
        "timestamp": pd.Timestamp(timestamp).isoformat(),
        "action": action,
        "reason": reason,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:16]


def build_fill_lineage(
    events: tuple[PortfolioAuditEvent, ...],
    signals: dict[str, tuple[pd.Series, pd.Series]],
    *,
    halted: bool,
    liquidation_timestamp: pd.Timestamp | None = None,
) -> tuple[FillLineage, ...]:
    """Connect every persisted fill to its deterministic authorizing decision."""
    result: list[FillLineage] = []
    halt_timestamp = pd.Timestamp(liquidation_timestamp) if liquidation_timestamp is not None else None
    for event in sorted(events, key=lambda item: item.sequence):
        timestamp = pd.Timestamp(event.timestamp)
        if event.strategy_id not in signals:
            raise ValueError(f"missing signals for strategy: {event.strategy_id}")
        entry, exit_ = signals[event.strategy_id]
        is_entry = bool(entry.get(timestamp, False))
        is_exit = bool(exit_.get(timestamp, False))
        if event.action == "buy" and is_entry:
            reason = "entry_signal"
        elif event.action == "sell" and is_exit:
            reason = "exit_signal"
        elif event.action == "sell" and halted and halt_timestamp == timestamp:
            reason = "risk_halt"
        elif event.action == "sell" and timestamp == entry.index[-1]:
            reason = "end_of_sample"
        else:
            raise ValueError(
                f"fill at {event.timestamp} for {event.strategy_id} has no deterministic authorizing decision"
            )
        result.append(
            FillLineage(
                audit_event_id(event),
                decision_id(event.strategy_id, timestamp, event.action, reason),
                reason,
            )
        )
    return tuple(result)


def verify_fill_lineage(
    events: tuple[PortfolioAuditEvent, ...],
    lineage: tuple[FillLineage, ...],
    signals: dict[str, tuple[pd.Series, pd.Series]],
    *,
    halted: bool,
    liquidation_timestamp: pd.Timestamp | None = None,
) -> bool:
    """Verify that persisted lineage exactly matches deterministic strategy signals."""
    expected = build_fill_lineage(
        events,
        signals,
        halted=halted,
        liquidation_timestamp=liquidation_timestamp,
    )
    return expected == lineage
