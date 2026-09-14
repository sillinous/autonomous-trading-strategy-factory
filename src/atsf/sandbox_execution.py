from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING

from .live_execution_intent import LiveExecutionIntent, verify_execution_intent
from .live_execution_intent_store import LiveExecutionIntentStore
from .live_risk_gateway import LiveRiskCertificate, validate_live_risk_certificate

if TYPE_CHECKING:
    from .sandbox_execution_journal import SandboxExecutionJournal


class ExecutionMode(str, Enum):
    SANDBOX = "sandbox"
    LIVE = "live"


class ExecutionState(str, Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    ADMITTED = "ADMITTED"
    CONSUMED = "CONSUMED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    bid: float
    ask: float
    timestamp: float

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        for name, value in (("bid", self.bid), ("ask", self.ask), ("timestamp", self.timestamp)):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            raise ValueError("market bid/ask must be positive with ask >= bid")


@dataclass(frozen=True)
class SandboxFill:
    intent_id: str
    strategy_id: str
    symbol: str
    side: str
    quantity_fraction: float
    price: float
    market_timestamp: float
    filled_at: float


@dataclass(frozen=True)
class SandboxExecutionResult:
    mode: ExecutionMode
    state: ExecutionState
    intent_id: str
    fill: SandboxFill | None
    reasons: tuple[str, ...] = ()
    execution_authority: bool = False


class SandboxExecutionAdapter:
    """Deterministic local adapter; it has no broker or network capability."""

    def __init__(self, store: LiveExecutionIntentStore, journal: SandboxExecutionJournal | None = None) -> None:
        self._store = store
        self._journal = journal

    def _reject(self, intent_id: str, reason: str, *, now: float) -> SandboxExecutionResult:
        if self._journal is not None:
            latest = self._journal.latest(intent_id)
            if latest is not None and latest.state not in {
                ExecutionState.FILLED, ExecutionState.REJECTED,
                ExecutionState.CANCELLED, ExecutionState.EXPIRED,
            }:
                self._journal.append_in_transaction(
                    intent_id, ExecutionState.REJECTED, timestamp=now, detail=reason
                )
        return SandboxExecutionResult(
            mode=ExecutionMode.SANDBOX,
            state=ExecutionState.REJECTED,
            intent_id=intent_id,
            fill=None,
            reasons=(reason,),
        )

    def _execute_in_transaction(
        self,
        intent: LiveExecutionIntent,
        certificate: LiveRiskCertificate,
        market: MarketSnapshot,
        *,
        now: float,
        kill_switch_engaged: bool,
    ) -> SandboxExecutionResult:
        if self._journal is not None and self._journal.latest(intent.intent_id) is None:
            self._journal.append_in_transaction(intent.intent_id, ExecutionState.CREATED, timestamp=now)
        if market.symbol != intent.symbol:
            return self._reject(intent.intent_id, "market symbol does not match intent", now=now)
        if not verify_execution_intent(intent):
            return self._reject(intent.intent_id, "execution intent fingerprint is invalid", now=now)
        allowed, reasons = validate_live_risk_certificate(
            certificate,
            strategy_id=intent.strategy_id,
            now=now,
            requested_capital_fraction=intent.quantity_fraction,
            kill_switch_engaged=kill_switch_engaged,
        )
        if not allowed:
            return self._reject(intent.intent_id, reasons[0], now=now)
        record = self._store.get(intent.intent_id)
        if record is None:
            return self._reject(intent.intent_id, "execution intent has not been journaled", now=now)
        if record.status != "CREATED":
            return SandboxExecutionResult(
                mode=ExecutionMode.SANDBOX,
                state=ExecutionState.REJECTED,
                intent_id=intent.intent_id,
                fill=None,
                reasons=(f"already {record.status.lower()}",),
            )
        if self._journal is not None:
            latest = self._journal.latest(intent.intent_id)
            if latest is not None and latest.state is ExecutionState.CREATED:
                self._journal.append_in_transaction(intent.intent_id, ExecutionState.VALIDATED, timestamp=now)
                self._journal.append_in_transaction(intent.intent_id, ExecutionState.ADMITTED, timestamp=now)
        consumed = self._store.consume_in_transaction(
            intent, certificate, now=now, kill_switch_engaged=kill_switch_engaged
        )
        if consumed.status != "CONSUMED":
            return self._reject(intent.intent_id, "execution intent consumption failed", now=now)
        if self._journal is not None:
            self._journal.append_in_transaction(intent.intent_id, ExecutionState.CONSUMED, timestamp=now)
        price = market.ask if intent.side == "BUY" else market.bid
        fill = SandboxFill(
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity_fraction=intent.quantity_fraction,
            price=price,
            market_timestamp=market.timestamp,
            filled_at=now,
        )
        if self._journal is not None:
            self._journal.append_in_transaction(
                intent.intent_id, ExecutionState.FILLED, timestamp=now, detail=f"price={price}"
            )
        return SandboxExecutionResult(
            mode=ExecutionMode.SANDBOX,
            state=ExecutionState.FILLED,
            intent_id=intent.intent_id,
            fill=fill,
        )

    def execute(
        self,
        intent: LiveExecutionIntent,
        certificate: LiveRiskCertificate,
        market: MarketSnapshot,
        *,
        now: float,
        kill_switch_engaged: bool = False,
        mode: ExecutionMode = ExecutionMode.SANDBOX,
    ) -> SandboxExecutionResult:
        if mode is not ExecutionMode.SANDBOX:
            return SandboxExecutionResult(
                mode=mode, state=ExecutionState.REJECTED, intent_id=intent.intent_id,
                fill=None, reasons=("live execution mode is disabled",),
            )
        if not isfinite(now):
            raise ValueError("now must be finite")
        with self._store.transaction():
            return self._execute_in_transaction(
                intent, certificate, market, now=now, kill_switch_engaged=kill_switch_engaged
            )
