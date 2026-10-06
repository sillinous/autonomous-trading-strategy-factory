"""Deterministic long-only backtest kernel with a cash-accounted, reconciled trade ledger.

Execution model
---------------
* ``signal`` is the *desired position state* decided at the close of each bar.
* A change in desired state is filled on the **next** bar: at its ``open`` when the
  data carries an ``open`` column, otherwise at its ``close``. Nothing ever fills on
  the bar that produced the decision, so the kernel cannot trade on information it
  did not yet have.
* Slippage is applied adversely to the fill price; commission is charged on notional.
* Position size is fixed at entry from information available at the decision bar.
* ``risk.stop_loss`` is checked intrabar against ``low`` (``close`` if absent); a gap
  through the stop fills at the open. After a stop, re-entry requires the signal to
  reset (go flat, then active again).
* ``risk.max_drawdown`` is a kill switch: once equity breaches it, the position is
  flattened on the next bar and the strategy stays flat for the rest of the run.

Ledger invariant
----------------
Trades never overlap and idle cash earns nothing, so the compounded per-trade returns
equal the equity curve's total return exactly. The kernel asserts this on every run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .accounting import reconcile_equity
from .strategy import StrategySpec

PERIODS_PER_YEAR: dict[str, int] = {
    "1m": 252 * 390,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "1h": 252 * 7,
    "4h": 252 * 2,
    "1d": 252,
}

TRADE_COLUMNS = ["timestamp", "action", "price", "shares", "fee", "reason"]
ROUND_TRIP_COLUMNS = [
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "shares",
    "fraction",
    "pnl",
    "return",
    "exit_reason",
]


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    fill: Literal["next_open", "next_close"] = "next_open"
    volatility_lookback: int = 20


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


def _column(data: pd.DataFrame, name: str, fallback: str = "close") -> np.ndarray:
    source = name if name in data.columns else fallback
    values = pd.to_numeric(data[source], errors="raise").astype(float).to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{source} prices must be finite and positive")
    return values


def target_fraction(strategy: StrategySpec, closes: np.ndarray, decision: int,
                    config: BacktestConfig) -> float:
    """Capital fraction for an entry decided at bar ``decision`` (uses data <= decision)."""
    sizing = strategy.position_sizing
    ceiling = min(sizing.max_position, strategy.risk.max_position, strategy.risk.max_leverage)
    if sizing.method == "fixed_fraction":
        fraction = sizing.value
    elif sizing.method == "equal_weight":
        fraction = 1.0 / len(strategy.universe)
    else:  # volatility_target
        lookback = config.volatility_lookback
        if decision < lookback:
            return 0.0
        window = np.log(closes[decision - lookback: decision + 1])
        realized = float(np.diff(window).std(ddof=1)) * math.sqrt(
            PERIODS_PER_YEAR[strategy.timeframe]
        )
        if not math.isfinite(realized) or realized <= 0:
            return 0.0
        fraction = sizing.value / realized
    return float(max(0.0, min(fraction, ceiling)))


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
    if config.volatility_lookback < 2:
        raise ValueError("volatility_lookback must be at least 2")

    index = data.index
    closes = _column(data, "close")
    opens = _column(data, "open") if config.fill == "next_open" else closes
    lows = _column(data, "low")
    desired = signal.fillna(False).astype(bool).to_numpy()
    commission = config.commission_bps / 10_000.0
    slippage = config.slippage_bps / 10_000.0
    stop = strategy.risk.stop_loss
    kill = strategy.risk.max_drawdown

    cash = float(config.initial_cash)
    shares = 0.0
    entry_cost = 0.0          # cash paid for the open position, fees included
    entry_equity = 0.0        # account equity immediately before the entry
    entry_price = 0.0         # executed (slipped) entry price
    entry_reference = 0.0     # unslipped fill reference, basis for the stop
    entry_time: pd.Timestamp | None = None
    entry_fraction = 0.0
    stopped_out = False
    halted = False
    halted_at: pd.Timestamp | None = None
    peak = cash

    equity = np.empty(len(index))
    events: list[dict[str, object]] = []
    trips: list[dict[str, object]] = []

    def sell(i: int, reference: float, reason: str) -> None:
        nonlocal cash, shares
        price = reference * (1.0 - slippage)
        proceeds = shares * price
        fee = proceeds * commission
        cash += proceeds - fee
        pnl = proceeds - fee - entry_cost
        events.append({"timestamp": index[i], "action": "sell", "price": price,
                       "shares": shares, "fee": fee, "reason": reason})
        trips.append({"entry_time": entry_time, "exit_time": index[i],
                      "entry_price": entry_price, "exit_price": price, "shares": shares,
                      "fraction": entry_fraction, "pnl": pnl,
                      "return": pnl / entry_equity, "exit_reason": reason})
        shares = 0.0

    for i in range(len(index)):
        if i > 0:
            fill = opens[i] if config.fill == "next_open" else closes[i]
            want = bool(desired[i - 1]) and not halted
            if not desired[i - 1]:
                stopped_out = False
            if shares > 0 and not want:
                sell(i, fill, "max_drawdown" if halted else "signal")
            elif shares == 0 and want and not stopped_out:
                fraction = target_fraction(strategy, closes, i - 1, config)
                if fraction > 0:
                    price = fill * (1.0 + slippage)
                    budget = cash * fraction
                    qty = budget / (price * (1.0 + commission))
                    fee = qty * price * commission
                    entry_equity, entry_fraction = cash, fraction
                    entry_cost = qty * price + fee
                    cash -= entry_cost
                    shares, entry_price, entry_reference = qty, price, fill
                    entry_time = index[i]
                    events.append({"timestamp": index[i], "action": "buy", "price": price,
                                   "shares": qty, "fee": fee, "reason": "signal"})
            if shares > 0 and stop is not None:
                stop_price = entry_reference * (1.0 - stop)
                if lows[i] <= stop_price:
                    gapped = config.fill == "next_open" and opens[i] <= stop_price \
                        and entry_time != index[i]
                    sell(i, opens[i] if gapped else stop_price, "stop_loss")
                    stopped_out = True
        equity[i] = cash + shares * closes[i]
        peak = max(peak, equity[i])
        if kill is not None and not halted and equity[i] / peak - 1.0 <= -kill:
            halted, halted_at = True, index[i]

    if shares > 0:  # mark the open position to the final close; not a fill, no cost
        mark = shares * closes[-1]
        pnl = mark - entry_cost
        trips.append({"entry_time": entry_time, "exit_time": index[-1],
                      "entry_price": entry_price, "exit_price": closes[-1], "shares": shares,
                      "fraction": entry_fraction, "pnl": pnl,
                      "return": pnl / entry_equity, "exit_reason": "open"})

    equity_series = pd.Series(equity, index=index, name="equity")
    returns = equity_series.pct_change()
    returns.iloc[0] = equity[0] / config.initial_cash - 1.0
    reconciliation = reconcile_equity(equity_series, returns,
                                      tolerance=1e-9 * config.initial_cash)
    if not reconciliation.passed:
        raise RuntimeError("backtest accounting reconciliation failed")

    trade_returns = tuple(float(trip["return"]) for trip in trips)
    compounded = float(np.prod([1.0 + r for r in trade_returns])) if trade_returns else 1.0
    if not math.isclose(compounded, equity[-1] / config.initial_cash, rel_tol=1e-9):
        raise RuntimeError("trade ledger does not reconcile with the equity curve")

    drawdown = equity_series / equity_series.cummax() - 1.0
    return BacktestResult(
        equity=equity_series,
        returns=returns,
        trades=pd.DataFrame(events, columns=TRADE_COLUMNS),
        total_return=float(equity[-1] / config.initial_cash - 1.0),
        max_drawdown=float(drawdown.min()),
        trade_returns=trade_returns,
        round_trips=pd.DataFrame(trips, columns=ROUND_TRIP_COLUMNS),
        halted_at=halted_at,
    )
