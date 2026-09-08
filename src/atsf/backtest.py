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
    trade_returns: tuple[float, ...] = ()


def run_long_signal_backtest(
    data: pd.DataFrame,
    signal: pd.Series,
    strategy: StrategySpec,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Deterministic long-only kernel with an auditable trade-return ledger."""
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

    trade_rows: list[dict[str, object]] = []
    trade_returns: list[float] = []
    previous = False
    entry_price: float | None = None
    for timestamp, active in target.items():
        active = bool(active)
        if active and not previous:
            entry_price = float(prices.loc[timestamp])
            trade_rows.append({"timestamp": timestamp, "action": "buy"})
        elif not active and previous:
            if entry_price is not None:
                gross = float(prices.loc[timestamp] / entry_price - 1.0)
                cost = 2.0 * (config.commission_bps + config.slippage_bps) / 10_000.0
                trade_returns.append(gross * strategy.position_sizing.max_position - cost)
            trade_rows.append({"timestamp": timestamp, "action": "sell"})
            entry_price = None
        previous = active

    trades = pd.DataFrame(trade_rows, columns=["timestamp", "action"])
    return BacktestResult(
        equity=equity,
        trades=trades,
        total_return=float(equity.iloc[-1] / config.initial_cash - 1.0),
        max_drawdown=float(drawdown.min()),
        trade_returns=tuple(trade_returns),
    )
