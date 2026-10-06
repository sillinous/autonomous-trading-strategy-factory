from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .execution import RISK_EXIT_REASONS
from .portfolio_audit import PortfolioAuditEvent, audit_event_id
from .signals import position_state


@dataclass(frozen=True)
class FillLineage:
    event_id: str
    decision_id: str
    reason: str


def decision_id(strategy_id: str, timestamp: pd.Timestamp, action: str, reason: str) -> str:
    if not strategy_id or not action or not reason:
        raise ValueError("strategy_id, action, and reason are required")
    payload = {"strategy_id": strategy_id, "timestamp": pd.Timestamp(timestamp).isoformat(), "action": action, "reason": reason}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:16]


def build_fill_lineage(events: tuple[PortfolioAuditEvent, ...], signals: dict[str, tuple[pd.Series, pd.Series]], *, halted: bool, liquidation_timestamp: pd.Timestamp | None = None, risk_exits: dict[str, dict[pd.Timestamp, str]] | None = None) -> tuple[FillLineage, ...]:
    """Attribute every fill to the deterministic decision that authorized it.

    Fills follow the next-bar execution contract in :mod:`atsf.execution`: a signal
    fill at bar ``t`` is authorized by the position state decided at bar ``t - 1``.
    Sleeve stop-loss and drawdown exits must be declared in ``risk_exits``; anything
    without an authorizing decision fails closed.
    """
    result: list[FillLineage] = []
    risk_exits = {key: {pd.Timestamp(ts): reason for ts, reason in value.items()} for key, value in (risk_exits or {}).items()}
    if any(reason not in RISK_EXIT_REASONS for value in risk_exits.values() for reason in value.values()):
        raise ValueError("risk_exits may only declare stop_loss or max_drawdown exits")
    halt_timestamp = pd.Timestamp(liquidation_timestamp) if liquidation_timestamp is not None else None
    if halted and halt_timestamp is None and any(event.action == "sell" for event in events):
        raise ValueError("halted liquidation requires a deterministic liquidation timestamp")
    states: dict[str, pd.Series] = {}
    for event in sorted(events, key=lambda item: item.sequence):
        timestamp = pd.Timestamp(event.timestamp)
        if event.strategy_id not in signals:
            raise ValueError(f"missing signals for strategy: {event.strategy_id}")
        if event.strategy_id not in states:
            states[event.strategy_id] = position_state(*signals[event.strategy_id])
        state = states[event.strategy_id]
        if timestamp not in state.index:
            raise ValueError(f"fill at {event.timestamp} for {event.strategy_id} has no deterministic authorizing decision")
        location = state.index.get_loc(timestamp)
        decided_active = bool(state.iloc[location - 1]) if location > 0 else None
        declared = risk_exits.get(event.strategy_id, {}).get(timestamp)
        if event.action == "sell" and halted and timestamp == halt_timestamp:
            reason = "risk_halt"
        elif event.action == "sell" and declared is not None:
            reason = declared
        elif event.action == "buy" and decided_active is True:
            reason = "entry_signal"
        elif event.action == "sell" and decided_active is False:
            reason = "exit_signal"
        elif event.action == "sell" and not halted and location == len(state.index) - 1:
            reason = "end_of_sample"
        else:
            raise ValueError(f"fill at {event.timestamp} for {event.strategy_id} has no deterministic authorizing decision")
        result.append(FillLineage(audit_event_id(event), decision_id(event.strategy_id, timestamp, event.action, reason), reason))
    return tuple(result)


def verify_fill_lineage(events: tuple[PortfolioAuditEvent, ...], lineage: tuple[FillLineage, ...], signals: dict[str, tuple[pd.Series, pd.Series]], *, halted: bool, liquidation_timestamp: pd.Timestamp | None = None, risk_exits: dict[str, dict[pd.Timestamp, str]] | None = None) -> bool:
    return build_fill_lineage(events, signals, halted=halted, liquidation_timestamp=liquidation_timestamp, risk_exits=risk_exits) == lineage
