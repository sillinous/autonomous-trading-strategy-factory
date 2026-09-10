from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from .paper import PaperBroker, PaperConfig, PaperFill
from .paper_risk import PaperRiskController
from .promotion import PromotionDecision
from .signals import strategy_signals
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


def run_paper_portfolio(
    data: dict[str, pd.DataFrame],
    strategies: dict[str, StrategySpec],
    weights: dict[str, float],
    *,
    decisions: dict[str, PromotionDecision],
    initial_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    max_drawdown: float | None = None,
) -> PortfolioPaperResult:
    """Run multiple promotion-approved strategies through isolated paper brokers."""
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
        if not decision.eligible or decision.stage not in {"paper", "live"}:
            raise PermissionError(f"strategy {strategy_id} is not eligible for paper execution")

    indexes = [frame.index for frame in data.values()]
    if any(not isinstance(index, pd.DatetimeIndex) for index in indexes):
        raise ValueError("all paper portfolio indexes must be DatetimeIndex")
    if any(not index.is_monotonic_increasing or index.has_duplicates for index in indexes):
        raise ValueError("all paper portfolio indexes must be sorted and unique")
    reference_index = indexes[0]
    if any(not index.equals(reference_index) for index in indexes[1:]):
        raise ValueError("all paper portfolio datasets must use the same timestamps")

    brokers: dict[str, PaperBroker] = {}
    signals: dict[str, tuple[pd.Series, pd.Series]] = {}
    for strategy_id, strategy in strategies.items():
        frame = data[strategy_id]
        if frame.empty or "close" not in frame.columns:
            raise ValueError(f"data for {strategy_id} must contain a non-empty close column")
        close = pd.to_numeric(frame["close"], errors="coerce")
        if close.isna().any() or (~close.map(isfinite)).any() or (close <= 0).any():
            raise ValueError(f"data for {strategy_id} must contain finite positive close prices")
        brokers[strategy_id] = PaperBroker(
            PaperConfig(
                initial_cash=initial_cash * weights[strategy_id],
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )
        )
        signals[strategy_id] = strategy_signals(frame, strategy)

    risk = PaperRiskController(max_drawdown)
    reserve = initial_cash * (1.0 - total_weight)
    snapshots: list[PortfolioPaperSnapshot] = []
    fills: list[tuple[str, PaperFill]] = []

    for timestamp in reference_index:
        equity = reserve
        for strategy_id, broker in brokers.items():
            frame = data[strategy_id]
            price = float(frame.loc[timestamp, "close"])
            entry, exit_ = signals[strategy_id]
            if bool(entry.loc[timestamp]) and broker.position == 0:
                quantity = broker.cash * strategies[strategy_id].position_sizing.max_position / price
                if quantity > 0:
                    fills.append((strategy_id, broker.execute(timestamp, "buy", quantity, price)))
            elif bool(exit_.loc[timestamp]) and broker.position > 0:
                fills.append((strategy_id, broker.execute(timestamp, "sell", broker.position, price)))
            equity += broker.mark(timestamp, price).equity

        if not risk.check(equity):
            for strategy_id, broker in brokers.items():
                if broker.position > 0:
                    price = float(data[strategy_id].loc[timestamp, "close"])
                    fills.append((strategy_id, broker.execute(timestamp, "sell", broker.position, price)))
            equity = reserve + sum(
                broker.mark(timestamp, float(data[strategy_id].loc[timestamp, "close"])).equity
                for strategy_id, broker in brokers.items()
            )
            snapshots.append(PortfolioPaperSnapshot(timestamp, equity, reserve))
            break
        snapshots.append(PortfolioPaperSnapshot(timestamp, equity, reserve))

    if not snapshots:
        raise ValueError("paper portfolio data cannot be empty")

    # Deterministically close residual positions at the final observed price so
    # the returned result represents realized portfolio capital, not open mark-to-market exposure.
    if not risk.state.halted:
        timestamp = reference_index[-1]
        for strategy_id, broker in brokers.items():
            if broker.position > 0:
                price = float(data[strategy_id].loc[timestamp, "close"])
                fills.append((strategy_id, broker.execute(timestamp, "sell", broker.position, price)))
        final_equity = reserve + sum(
            broker.mark(timestamp, float(data[strategy_id].loc[timestamp, "close"])).equity
            for strategy_id, broker in brokers.items()
        )
        if snapshots[-1].timestamp == timestamp:
            snapshots[-1] = PortfolioPaperSnapshot(timestamp, final_equity, reserve)
        else:
            snapshots.append(PortfolioPaperSnapshot(timestamp, final_equity, reserve))

    state = risk.state
    return PortfolioPaperResult(
        tuple(snapshots),
        tuple(fills),
        snapshots[-1].equity,
        state.halted,
        state.reason,
    )
