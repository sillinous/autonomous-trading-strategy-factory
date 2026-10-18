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
    if not strategy_id or not action or not reason:
        raise ValueError("strategy_id, action, and reason are required")
    payload = {"strategy_id": strategy_id, "timestamp": pd.Timestamp(timestamp).isoformat(), "action": action, "reason": reason}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:16]


def build_fill_lineage(events: tuple[PortfolioAuditEvent, ...], signals: dict[str, tuple[pd.Series, pd.Series]], *, halted: bool, liquidation_timestamp: pd.Timestamp | None = None) -> tuple[FillLineage, ...]:
    if not isinstance(halted, bool):
        raise ValueError("halted must be boolean")
    for strategy_id, pair in signals.items():
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("signal strategy IDs must be non-empty strings")
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"signals for {strategy_id} must be an entry/exit series pair")
        entry, exit_ = pair
        if not isinstance(entry, pd.Series) or not isinstance(exit_, pd.Series):
            raise ValueError(f"signals for {strategy_id} must be pandas Series")
        for signal in (entry, exit_):
            index = signal.index
            if not isinstance(index, pd.DatetimeIndex):
                raise ValueError(f"signal indexes for {strategy_id} must be DatetimeIndex")
            if index.isna().any() or index.tz is not None:
                raise ValueError(f"signal indexes for {strategy_id} must have valid timezone-naive timestamps")
            if not index.is_monotonic_increasing or index.has_duplicates:
                raise ValueError(f"signal indexes for {strategy_id} must be sorted and unique")
            if not pd.api.types.is_bool_dtype(signal.dtype) or signal.isna().any():
                raise ValueError(f"signals for {strategy_id} must contain non-missing boolean values")
        if not entry.index.equals(exit_.index):
            raise ValueError(f"entry and exit signal indexes must match for {strategy_id}")
    sequences = [event.sequence for event in events]
    if len(sequences) != len(set(sequences)):
        raise ValueError("event sequence values must be unique")
    result: list[FillLineage] = []
    halt_timestamp = pd.Timestamp(liquidation_timestamp) if liquidation_timestamp is not None else None
    if halted and halt_timestamp is None and any(event.action == "sell" for event in events):
        raise ValueError("halted liquidation requires a deterministic liquidation timestamp")
    for event in sorted(events, key=lambda item: item.sequence):
        if event.action not in {"buy", "sell"}:
            raise ValueError(f"unsupported execution action: {event.action}")
        try:
            timestamp = pd.Timestamp(event.timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid fill timestamp: {event.timestamp}") from exc
        if pd.isna(timestamp) or timestamp.tz is not None:
            raise ValueError(f"fill timestamp must be valid and timezone-naive: {event.timestamp}")
        if event.strategy_id not in signals:
            raise ValueError(f"missing signals for strategy: {event.strategy_id}")
        entry, exit_ = signals[event.strategy_id]
        if timestamp not in entry.index:
            raise ValueError(f"fill timestamp is absent from signals for strategy: {event.strategy_id}")
        # Decisions are observed at the prior close; a fill cannot be
        # authorized by a signal generated on its own fill timestamp.
        prior_timestamps = entry.index[entry.index < timestamp]
        decision_timestamp = prior_timestamps[-1] if len(prior_timestamps) else None
        is_entry = decision_timestamp is not None and bool(entry.get(decision_timestamp, False))
        is_exit = decision_timestamp is not None and bool(exit_.get(decision_timestamp, False))
        if halted and event.action == "sell":
            if halt_timestamp != timestamp:
                raise ValueError(f"fill at {event.timestamp} for {event.strategy_id} has no deterministic authorizing decision")
            reason = "risk_halt"
        elif event.action == "buy" and is_entry:
            reason = "entry_signal"
        elif event.action == "sell" and is_exit:
            reason = "exit_signal"
        elif event.action == "sell" and not halted and timestamp == entry.index[-1]:
            reason = "end_of_sample"
        else:
            raise ValueError(f"fill at {event.timestamp} for {event.strategy_id} has no deterministic authorizing decision")
        result.append(FillLineage(audit_event_id(event), decision_id(event.strategy_id, timestamp, event.action, reason), reason))
    return tuple(result)


def verify_fill_lineage(events: tuple[PortfolioAuditEvent, ...], lineage: tuple[FillLineage, ...], signals: dict[str, tuple[pd.Series, pd.Series]], *, halted: bool, liquidation_timestamp: pd.Timestamp | None = None) -> bool:
    return build_fill_lineage(events, signals, halted=halted, liquidation_timestamp=liquidation_timestamp) == lineage
