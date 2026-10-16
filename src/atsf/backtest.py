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
    """Backtest close-only signals with fills at the next close.

    A signal observed at close[t-1] is filled at close[t]. The position filled
    at close[t] earns only the subsequent close-to-close return ending at t+1,
    never the return ending at its own fill. Position changes and costs are
    booked at their fill timestamp. Any position open at the evaluation-window
    end is liquidated at the final close; this charges exit friction but does
    not carry exposure into a later fold. Final-bar entries are immediately
    liquidated at that same close because no subsequent bar exists in-window.
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
    if not prices.map(pd.notna).all() or (prices <= 0).any():
        raise ValueError("close prices must be finite and positive")
    target = signal.astype(bool).shift(1).fillna(False)
    position = target.astype(float) * strategy.position_sizing.max_position

    # Exposure for the interval ending at t is the position held after close[t-1].
    interval_exposure = position.shift(1).fillna(0.0)
    price_returns = prices.pct_change().fillna(0.0)
    gross_returns = price_returns * interval_exposure

    turnover = position.diff().abs().fillna(position.abs())
    cost_rate = (config.commission_bps + config.slippage_bps) / 10_000.0
    friction = turnover * cost_rate
    # A position still open after the final fill is closed at the final close.
    if bool(target.iloc[-1]):
        friction.iloc[-1] += float(position.iloc[-1]) * cost_rate

    net_returns = gross_returns - friction
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
            trade_rows.append({"timestamp": timestamp, "action": "buy", "price": entry_price})
        elif not active and previous:
            if entry_price is not None:
                gross = float(prices.loc[timestamp] / entry_price - 1.0)
                trade_returns.append(
                    gross * strategy.position_sizing.max_position - 2.0 * strategy.position_sizing.max_position * cost_rate
                )
            trade_rows.append(
                {"timestamp": timestamp, "action": "sell", "price": float(prices.loc[timestamp])}
            )
            entry_price = None
        previous = active

    if previous and entry_price is not None:
        final_timestamp = data.index[-1]
        gross = float(prices.iloc[-1] / entry_price - 1.0)
        trade_returns.append(gross * strategy.position_sizing.max_position - 2.0 * cost_rate)
        trade_rows.append(
            {"timestamp": final_timestamp, "action": "sell", "price": float(prices.iloc[-1])}
        )

    trades = pd.DataFrame(trade_rows, columns=["timestamp", "action", "price"])
    return BacktestResult(
        equity=equity,
        returns=net_returns,
        trades=trades,
        total_return=float(equity.iloc[-1] / config.initial_cash - 1.0),
        max_drawdown=float(drawdown.min()),
        trade_returns=tuple(trade_returns),
    )
