"""Deterministic long-only backtest built on the shared execution engine.

The kernel steps :class:`~atsf.execution.SleeveEngine` against a fresh paper broker, so
backtest fills are exactly the fills paper execution would produce on the same data.
See :mod:`atsf.execution` for the execution contract.

Ledger invariant: trades never overlap and idle cash earns nothing, so compounded
per-trade returns equal the equity curve's total return. Every run asserts this.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .accounting import reconcile_equity
from .execution import PERIODS_PER_YEAR, ExecutionPolicy, run_sleeve
from .execution import target_fraction as _target_fraction
from .paper import PaperBroker, PaperConfig
from .strategy import StrategySpec

TRADE_COLUMNS = ["timestamp", "action", "price", "shares", "fee", "reason", "decision_timestamp"]
ROUND_TRIP_COLUMNS = ["entry_time", "exit_time", "entry_price", "exit_price", "shares",
                      "fraction", "pnl", "return", "exit_reason"]

__all__ = ["BacktestConfig", "BacktestResult", "PERIODS_PER_YEAR", "run_long_signal_backtest",
           "target_fraction"]


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    fill: Literal["next_open", "next_close"] = "next_open"
    volatility_lookback: int = 20

    @property
    def policy(self) -> ExecutionPolicy:
        return ExecutionPolicy(self.fill, self.volatility_lookback)

    @property
    def paper(self) -> PaperConfig:
        return PaperConfig(self.initial_cash, self.commission_bps, self.slippage_bps)


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    total_return: float
    max_drawdown: float
    trade_returns: tuple[float, ...] = ()
    round_trips: pd.DataFrame | None = None
    halted_at: pd.Timestamp | None = None


def target_fraction(strategy: StrategySpec, closes: np.ndarray, decision: int,
                    config: BacktestConfig | None = None) -> float:
    return _target_fraction(strategy, closes, decision, (config or BacktestConfig()).policy)


def run_long_signal_backtest(
    data: pd.DataFrame,
    signal: pd.Series,
    strategy: StrategySpec,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Deterministic long-only kernel with next-bar execution and an auditable ledger."""
    config = config or BacktestConfig()
    if "close" not in data.columns:
        raise ValueError("data must contain a close column")
    if not data.index.equals(signal.index):
        raise ValueError("data and signal indexes must match")
    if data.empty:
        raise ValueError("data cannot be empty")
    if config.initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if config.commission_bps < 0 or config.slippage_bps < 0:
        raise ValueError("transaction costs cannot be negative")

    engine = run_sleeve(strategy, data, signal, PaperBroker(config.paper), config.policy)
    equity = pd.Series(engine.equity, index=data.index, name="equity", dtype=float)
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / config.initial_cash - 1.0
    if not reconcile_equity(equity, returns, tolerance=1e-9 * config.initial_cash).passed:
        raise RuntimeError("backtest accounting reconciliation failed")

    trips = list(engine.round_trips)
    still_open = engine.open_round_trip()
    if still_open is not None:
        trips.append(still_open)
    trade_returns = tuple(trip.trade_return for trip in trips)
    compounded = math.prod(1.0 + r for r in trade_returns)
    if not math.isclose(compounded, equity.iloc[-1] / config.initial_cash, rel_tol=1e-9):
        raise RuntimeError("trade ledger does not reconcile with the equity curve")

    trades = pd.DataFrame(
        [{"timestamp": e.timestamp, "action": e.action, "price": e.price, "shares": e.quantity,
          "fee": e.fee, "reason": e.reason, "decision_timestamp": e.decision_timestamp}
         for e in engine.events],
        columns=TRADE_COLUMNS,
    )
    round_trips = pd.DataFrame(
        [{**{k: v for k, v in asdict(t).items() if k != "trade_return"}, "return": t.trade_return}
         for t in trips],
        columns=ROUND_TRIP_COLUMNS,
    )
    return BacktestResult(
        equity=equity,
        returns=returns,
        trades=trades,
        total_return=float(equity.iloc[-1] / config.initial_cash - 1.0),
        max_drawdown=float((equity / equity.cummax() - 1.0).min()),
        trade_returns=trade_returns,
        round_trips=round_trips,
        halted_at=engine.halted_at,
    )
