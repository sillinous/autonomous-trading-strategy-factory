from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from numbers import Real

from .portfolio_audit import PortfolioAuditEvent, audit_event_id


@dataclass(frozen=True)
class LedgerState:
    cash: float
    positions: dict[str, float]


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and isfinite(value)


def validate_execution_ledger(
    events: tuple[PortfolioAuditEvent, ...],
    *,
    initial_cash_by_strategy: dict[str, float],
    commission_bps: float,
    initial_cash: float,
    final_equity: float,
    tolerance: float = 1e-8,
) -> LedgerState:
    """Replay validated fills as accounting state and reject impossible transitions."""
    if not isinstance(events, tuple):
        raise ValueError("events must be a tuple")
    if not isinstance(initial_cash_by_strategy, dict) or not initial_cash_by_strategy:
        raise ValueError("initial cash by strategy cannot be empty")
    if any(not isinstance(key, str) or not key.strip() for key in initial_cash_by_strategy):
        raise ValueError("strategy IDs must be non-empty strings")
    if not _finite_number(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if not _finite_number(final_equity) or final_equity <= 0:
        raise ValueError("final_equity must be positive and finite")
    if not _finite_number(commission_bps) or commission_bps < 0:
        raise ValueError("commission_bps must be finite and non-negative")
    if not _finite_number(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if any(not _finite_number(value) or value <= 0 for value in initial_cash_by_strategy.values()):
        raise ValueError("strategy cash allocations must be positive and finite")
    if abs(sum(initial_cash_by_strategy.values()) - initial_cash) > tolerance:
        raise ValueError("strategy cash allocations do not reconcile to initial cash")

    cash = dict(initial_cash_by_strategy)
    positions = {strategy_id: 0.0 for strategy_id in initial_cash_by_strategy}
    previous_timestamp: datetime | None = None
    previous_sequence: int | None = None
    seen_sequences: set[int] = set()
    fee_mismatches: list[PortfolioAuditEvent] = []
    for event in events:
        if not isinstance(event, PortfolioAuditEvent):
            raise ValueError("events must contain PortfolioAuditEvent records")
        # Reuse the canonical event validator so replay and event hashing agree on
        # which event shapes are considered valid.
        audit_event_id(event)
        if event.sequence in seen_sequences:
            raise ValueError("audit event sequence values must be unique")
        if previous_sequence is not None and event.sequence <= previous_sequence:
            raise ValueError("audit event sequence values must be strictly increasing")
        seen_sequences.add(event.sequence)
        previous_sequence = event.sequence
        timestamp = datetime.fromisoformat(event.timestamp)
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError("audit events must be chronological")
        previous_timestamp = timestamp
        if event.strategy_id not in cash:
            raise ValueError("audit event references a strategy outside the execution")
        if event.quantity <= 0 or event.price <= 0 or event.fee < 0:
            raise ValueError("audit fill quantity, price, and fee are invalid")
        notional = event.quantity * event.price
        if notional <= 0 or not isfinite(notional):
            raise ValueError("audit fill notional is invalid")
        expected_fee = notional * commission_bps / 10_000.0
        if abs(event.fee - expected_fee) > tolerance * max(1.0, expected_fee):
            fee_mismatches.append(event)
        if event.action == "sell":
            if event.quantity > positions[event.strategy_id] + tolerance:
                raise ValueError("audit ledger contains a sell exceeding the strategy position")
            cash[event.strategy_id] += notional - event.fee
            positions[event.strategy_id] = max(0.0, positions[event.strategy_id] - event.quantity)
        else:
            required = notional + event.fee
            if required > cash[event.strategy_id] + tolerance:
                raise ValueError("audit ledger contains a buy exceeding available strategy cash")
            cash[event.strategy_id] -= required
            positions[event.strategy_id] += event.quantity

    if fee_mismatches:
        raise ValueError("audit fill fee does not match execution configuration")
    if any(abs(position) > tolerance for position in positions.values()):
        raise ValueError("audit ledger does not end flat")
    ending_cash = sum(cash.values())
    if abs(ending_cash - final_equity) > tolerance * max(1.0, abs(final_equity)):
        raise ValueError("audit ledger cash does not reconcile to final equity")
    return LedgerState(ending_cash, positions)
