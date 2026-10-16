from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .accounting import reconcile_equity
from .strategy import StrategySpec


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
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
    """Run a deterministic long-only backtest, liquidating at the final close.

    Signals execute on the next bar. Any position still open at the end of the
    supplied evaluation window is liquidated at that window's final close;
    the exit incurs commission and slippage and is represented in the trade
    ledger and trade_returns. This keeps fold-level trade statistics aligned
    with the equity ledger without carrying positions across independent folds.
    """
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

    prices = data["close"].astype(float)
    target = signal.astype(bool).shift(1).fillna(False)
    position = target.astype(float) * strategy.position_sizing.max_position
    returns = prices.pct_change().fillna(0.0) * position
    turnover = position.diff().abs().fillna(position.abs())
    cost_rate = (config.commission_bps + config.slippage_bps) / 10_000.0
    friction = turnover * cost_rate
    # The final mark-to-market return is earned while holding the position;
    # liquidation friction is charged at that same final close.
    if bool(target.iloc[-1]):
        friction.iloc[-1] += float(position.iloc[-1]) * cost_rate
    net_returns = returns - friction
    equity = (1.0 + net_returns).cumprod() * config.initial_cash
    drawdown = equity / equity.cummax() - 1.0

    reconciliation = reconcile_equity(equity, net_returns)
    if not reconciliation.passed:
        raise RuntimeError("backtest accounting reconciliation failed")

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
                cost = 2.0 * cost_rate
                trade_returns.append(gross * strategy.position_sizing.max_position - cost)
            trade_rows.append({"timestamp": timestamp, "action": "sell"})
            entry_price = None
        previous = active

    if previous and entry_price is not None:
        final_timestamp = data.index[-1]
        gross = float(prices.iloc[-1] / entry_price - 1.0)
        # Trade-level returns use the same full round-trip cost convention as
        # completed trades, including the forced final-close exit.
        trade_returns.append(gross * strategy.position_sizing.max_position - 2.0 * cost_rate)
        trade_rows.append({"timestamp": final_timestamp, "action": "sell"})

    trades = pd.DataFrame(trade_rows, columns=["timestamp", "action"])
    return BacktestResult(
        equity=equity,
        returns=net_returns,
        trades=trades,
        total_return=float(equity.iloc[-1] / config.initial_cash - 1.0),
        max_drawdown=float(drawdown.min()),
        trade_returns=tuple(trade_returns),
    )
