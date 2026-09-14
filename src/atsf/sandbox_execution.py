from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .live_execution_intent import LiveExecutionIntent, verify_execution_intent
from .live_execution_intent_store import LiveExecutionIntentStore
from .live_risk_gateway import LiveRiskCertificate, validate_live_risk_certificate


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

    def __init__(self, store: LiveExecutionIntentStore) -> None:
        self._store = store

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
                mode=mode,
                state=ExecutionState.REJECTED,
                intent_id=intent.intent_id,
                fill=None,
                reasons=("live execution mode is disabled",),
            )
        if not isfinite(now):
            raise ValueError("now must be finite")
        if market.symbol != intent.symbol:
            return SandboxExecutionResult(
                mode=mode, state=ExecutionState.REJECTED, intent_id=intent.intent_id,
                fill=None, reasons=("market symbol does not match intent",)
            )
        if not verify_execution_intent(intent):
            return SandboxExecutionResult(
                mode=mode, state=ExecutionState.REJECTED, intent_id=intent.intent_id,
                fill=None, reasons=("execution intent fingerprint is invalid",)
            )
        allowed, reasons = validate_live_risk_certificate(
            certificate,
            strategy_id=intent.strategy_id,
            now=now,
            requested_capital_fraction=intent.quantity_fraction,
            kill_switch_engaged=kill_switch_engaged,
        )
        if not allowed:
            return SandboxExecutionResult(
                mode=mode, state=ExecutionState.REJECTED, intent_id=intent.intent_id,
                fill=None, reasons=tuple(reasons)
            )
        record = self._store.get(intent.intent_id)
        if record is None:
            return SandboxExecutionResult(
                mode=mode, state=ExecutionState.REJECTED, intent_id=intent.intent_id,
                fill=None, reasons=("execution intent has not been journaled",)
            )
        if record.status != "CREATED":
            return SandboxExecutionResult(
                mode=mode, state=ExecutionState.REJECTED, intent_id=intent.intent_id,
                fill=None, reasons=(f"execution intent is already {record.status.lower()}",)
            )
        consumed = self._store.consume(
            intent, certificate, now=now, kill_switch_engaged=kill_switch_engaged
        )
        if consumed.status != "CONSUMED":
            return SandboxExecutionResult(
                mode=mode, state=ExecutionState.REJECTED, intent_id=intent.intent_id,
                fill=None, reasons=("execution intent consumption failed",)
            )
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
        return SandboxExecutionResult(
            mode=mode,
            state=ExecutionState.FILLED,
            intent_id=intent.intent_id,
            fill=fill,
        )
