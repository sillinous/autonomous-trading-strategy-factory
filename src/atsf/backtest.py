from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .strategy import StrategySpec


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    total_return: float
    max_drawdown: float


def run_long_signal_backtest(
    data: pd.DataFrame,
    signal: pd.Series,
    strategy: StrategySpec,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Minimal deterministic long-only kernel.

    `data` must contain close prices indexed by timestamp. `signal` is a boolean
    series aligned to data; positions are entered/exited at the next bar's close.
    This intentionally forms a small foundation for the richer event engine.
    """
    config = config or BacktestConfig()
    if "close" not in data.columns:
        raise ValueError("data must contain a close column")
    if not data.index.equals(signal.index):
        raise ValueError("data and signal indexes must match")
    if data.empty:
        raise ValueError("data cannot be empty")

    prices = data["close"].astype(float)
    target = signal.astype(bool).shift(1).fillna(False)
    position = target.astype(float) * strategy.position_sizing.max_position
    returns = prices.pct_change().fillna(0.0) * position

    turnover = position.diff().abs().fillna(position.abs())
    friction = turnover * (config.commission_bps + config.slippage_bps) / 10_000.0
    net_returns = returns - friction

    equity = (1.0 + net_returns).cumprod() * config.initial_cash
    drawdown = equity / equity.cummax() - 1.0

    trade_rows = []
    previous = False
    for timestamp, active in target.items():
        active = bool(active)
        if active != previous:
            trade_rows.append({"timestamp": timestamp, "action": "buy" if active else "sell"})
        previous = active

    trades = pd.DataFrame(trade_rows, columns=["timestamp", "action"])
    return BacktestResult(
        equity=equity,
        trades=trades,
        total_return=float(equity.iloc[-1] / config.initial_cash - 1.0),
        max_drawdown=float(drawdown.min()),
    )
