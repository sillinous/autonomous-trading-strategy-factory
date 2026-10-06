from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from .paper import PaperBroker, PaperConfig, PaperFill
from .paper_risk import PaperRiskController
from .promotion import PromotionDecision
from .execution import ExecutionEvent, ExecutionPolicy, SleeveEngine
from .signals import position_state, strategy_signals
from .strategy import StrategySpec


@dataclass(frozen=True)
class PortfolioPaperSnapshot:
    timestamp: pd.Timestamp
    equity: float
    cash_reserve: float


@dataclass(frozen=True)
class PortfolioPaperResult:
    snapshots: tuple[PortfolioPaperSnapshot, ...]
    fills: tuple[tuple[str, PaperFill], ...]
    final_equity: float
    halted: bool
    halt_reason: str | None
    liquidation_timestamp: pd.Timestamp | None = None
    events: tuple[tuple[str, ExecutionEvent], ...] = ()

    @property
    def risk_exits(self) -> dict[str, dict[pd.Timestamp, str]]:
        """Sleeve-level stop/kill exits, keyed by strategy and fill timestamp."""
        exits: dict[str, dict[pd.Timestamp, str]] = {}
        for strategy_id, event in self.events:
            if event.reason in {"stop_loss", "max_drawdown"}:
                exits.setdefault(strategy_id, {})[pd.Timestamp(event.timestamp)] = event.reason
        return exits


def run_paper_portfolio(data: dict[str, pd.DataFrame], strategies: dict[str, StrategySpec], weights: dict[str, float], *, decisions: dict[str, PromotionDecision], initial_cash: float = 100_000.0, commission_bps: float = 1.0, slippage_bps: float = 2.0, max_drawdown: float | None = None, policy: ExecutionPolicy | None = None) -> PortfolioPaperResult:
    """Run promotion-approved strategies as isolated sleeves on the shared execution engine.

    Each sleeve follows exactly the backtest execution contract. A portfolio drawdown
    breach observed at a bar's close flattens every sleeve on the next bar (or at the
    close when the breach is on the final bar) and blocks further entries.
    """
    if not strategies:
        raise ValueError("strategies cannot be empty")
    if set(data) != set(strategies) or set(weights) != set(strategies):
        raise ValueError("data, strategies, and weights must contain the same strategy IDs")
    if set(decisions) != set(strategies):
        raise ValueError("paper eligibility decisions are required for every strategy")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if any(not isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("weights must be finite and non-negative")
    total_weight = sum(weights.values())
    if total_weight > 1.0 + 1e-12:
        raise ValueError("portfolio weights exceed 100% gross exposure")
    if max_drawdown is not None and not 0 < max_drawdown < 1:
        raise ValueError("max_drawdown must be between 0 and 1")
    for strategy_id, decision in decisions.items():
        if not decision.eligible or decision.stage not in {"promoted", "paper", "live"}:
            raise PermissionError(f"strategy {strategy_id} is not eligible for paper execution")
    indexes = [frame.index for frame in data.values()]
    if any(not isinstance(index, pd.DatetimeIndex) for index in indexes):
        raise ValueError("all paper portfolio indexes must be DatetimeIndex")
    if any(not index.is_monotonic_increasing or index.has_duplicates for index in indexes):
        raise ValueError("all paper portfolio indexes must be sorted and unique")
    reference_index = indexes[0]
    if any(not index.equals(reference_index) for index in indexes[1:]):
        raise ValueError("all paper portfolio datasets must use the same timestamps")
    if len(reference_index) == 0:
        raise ValueError("paper portfolio data cannot be empty")
    engines: dict[str, SleeveEngine] = {}
    for strategy_id, strategy in strategies.items():
        frame = data[strategy_id]
        if frame.empty or "close" not in frame.columns:
            raise ValueError(f"data for {strategy_id} must contain a non-empty close column")
        close = pd.to_numeric(frame["close"], errors="coerce")
        if close.isna().any() or (~close.map(isfinite)).any() or (close <= 0).any():
            raise ValueError(f"data for {strategy_id} must contain finite positive close prices")
        broker = PaperBroker(PaperConfig(initial_cash=initial_cash * weights[strategy_id], commission_bps=commission_bps, slippage_bps=slippage_bps))
        entry, exit_ = strategy_signals(frame, strategy)
        engines[strategy_id] = SleeveEngine(strategy, frame, position_state(entry, exit_), broker, policy)
    risk = PaperRiskController(max_drawdown)
    reserve = initial_cash * (1.0 - total_weight)
    snapshots: list[PortfolioPaperSnapshot] = []
    halted = False
    halt_reason: str | None = None
    liquidation_timestamp: pd.Timestamp | None = None
    last = len(reference_index) - 1
    for i, timestamp in enumerate(reference_index):
        equity = reserve + sum(engine.step(i) for engine in engines.values())
        if not halted and not risk.check(equity):
            halted, halt_reason = True, risk.state.reason
            for engine in engines.values():
                engine.request_halt()
            if i == last:
                for engine in engines.values():
                    engine.liquidate(i, "risk_halt")
                equity = reserve + sum(engine.equity[i] for engine in engines.values())
            liquidation_timestamp = reference_index[min(i + 1, last)]
        snapshots.append(PortfolioPaperSnapshot(timestamp, equity, reserve))
    if not halted:
        for engine in engines.values():
            if engine.in_position:
                engine.liquidate(last, "max_drawdown" if engine.halted_at is not None else "end_of_sample")
        snapshots[-1] = PortfolioPaperSnapshot(reference_index[-1], reserve + sum(engine.equity[last] for engine in engines.values()), reserve)
    ordered = sorted(
        ((strategy_id, event) for strategy_id, engine in engines.items() for event in engine.events),
        key=lambda item: (item[1].timestamp, list(engines).index(item[0])),
    )
    fills = tuple((strategy_id, PaperFill(event.timestamp, event.action, event.quantity, event.price, event.fee)) for strategy_id, event in ordered)
    return PortfolioPaperResult(tuple(snapshots), fills, snapshots[-1].equity, halted, halt_reason, liquidation_timestamp if halted else None, tuple(ordered))
