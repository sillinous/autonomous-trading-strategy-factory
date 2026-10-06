"""The single execution engine shared by backtesting and every paper-trading path.

Backtests, single-strategy paper runs, portfolio sleeves, and attribution replays all
step a :class:`SleeveEngine` against a :class:`~atsf.paper.PaperBroker`, so a strategy
cannot behave differently in research than it does in paper execution.

Execution contract
------------------
* ``desired`` is the position state decided at the close of each bar. A change is
  filled on the next bar, at its ``open`` (or ``close`` when ``fill="next_close"`` or
  no ``open`` column exists). Nothing fills on the bar that produced the decision.
* Size is fixed at entry from information available at the decision bar.
* ``risk.stop_loss`` is checked intrabar against ``low`` (``close`` if absent); a gap
  through the stop fills at the open. Re-entry after a stop requires the signal to reset.
* ``risk.max_drawdown`` is a sleeve kill switch: flatten on the next bar, then stay flat.
* :meth:`SleeveEngine.request_halt` applies an external (portfolio) halt the same way.

Fill reasons form a closed vocabulary shared with fill lineage:
``entry_signal``, ``exit_signal``, ``stop_loss``, ``max_drawdown``, ``risk_halt``,
``end_of_sample``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .paper import PaperBroker, PaperSnapshot
from .strategy import StrategySpec

PERIODS_PER_YEAR: dict[str, int] = {
    "1m": 252 * 390,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "1h": 252 * 7,
    "4h": 252 * 2,
    "1d": 252,
}

FILL_REASONS = frozenset(
    {"entry_signal", "exit_signal", "stop_loss", "max_drawdown", "risk_halt", "end_of_sample"}
)
RISK_EXIT_REASONS = frozenset({"stop_loss", "max_drawdown"})


@dataclass(frozen=True)
class ExecutionPolicy:
    fill: Literal["next_open", "next_close"] = "next_open"
    volatility_lookback: int = 20

    def __post_init__(self) -> None:
        if self.fill not in {"next_open", "next_close"}:
            raise ValueError("fill must be next_open or next_close")
        if self.volatility_lookback < 2:
            raise ValueError("volatility_lookback must be at least 2")


@dataclass(frozen=True)
class ExecutionEvent:
    """One broker fill together with the decision that authorized it."""

    timestamp: pd.Timestamp
    decision_timestamp: pd.Timestamp
    action: str
    reason: str
    quantity: float
    price: float
    fee: float


@dataclass(frozen=True)
class RoundTrip:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    fraction: float
    pnl: float
    trade_return: float
    exit_reason: str


def price_column(data: pd.DataFrame, name: str, fallback: str = "close") -> np.ndarray:
    source = name if name in data.columns else fallback
    values = pd.to_numeric(data[source], errors="raise").astype(float).to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{source} prices must be finite and positive")
    return values


def target_fraction(strategy: StrategySpec, closes: np.ndarray, decision: int,
                    policy: ExecutionPolicy | None = None) -> float:
    """Capital fraction for an entry decided at bar ``decision`` (uses data <= decision)."""
    policy = policy or ExecutionPolicy()
    sizing = strategy.position_sizing
    ceiling = min(sizing.max_position, strategy.risk.max_position, strategy.risk.max_leverage)
    if sizing.method == "fixed_fraction":
        fraction = sizing.value
    elif sizing.method == "equal_weight":
        fraction = 1.0 / len(strategy.universe)
    else:  # volatility_target
        lookback = policy.volatility_lookback
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


class SleeveEngine:
    """Bar-stepped execution of one strategy sleeve against a paper broker."""

    def __init__(self, strategy: StrategySpec, data: pd.DataFrame, desired: pd.Series,
                 broker: PaperBroker, policy: ExecutionPolicy | None = None) -> None:
        if data.empty:
            raise ValueError("data cannot be empty")
        if "close" not in data.columns:
            raise ValueError("data must contain a close column")
        if not data.index.equals(desired.index):
            raise ValueError("data and signal indexes must match")
        self.strategy = strategy
        self.broker = broker
        self.policy = policy or ExecutionPolicy()
        self.index = data.index
        self.closes = price_column(data, "close")
        self.opens = price_column(data, "open") if self.policy.fill == "next_open" else self.closes
        self.lows = price_column(data, "low")
        self.desired = desired.fillna(False).astype(bool).to_numpy()
        self.events: list[ExecutionEvent] = []
        self.round_trips: list[RoundTrip] = []
        self.equity: list[float] = []
        self.snapshots: list[PaperSnapshot] = []
        self.halted_at: pd.Timestamp | None = None
        self._external_halt = False
        self._stopped_out = False
        self._peak = broker.cash
        self._entry: tuple[pd.Timestamp, float, float, float, float, float] | None = None

    @property
    def in_position(self) -> bool:
        return self.broker.position > 0

    @property
    def halted(self) -> bool:
        return self.halted_at is not None or self._external_halt

    def request_halt(self) -> None:
        """Flatten on the next bar and block all further entries."""
        self._external_halt = True

    def _buy(self, i: int, reference: float, fraction: float) -> None:
        broker = self.broker
        config = broker.config
        unit_cost = reference * (1.0 + config.slippage_bps / 10_000.0) * (
            1.0 + config.commission_bps / 10_000.0
        )
        quantity = min(broker.cash * fraction / unit_cost, broker.max_affordable_quantity(reference))
        if quantity <= 0:
            return
        equity_before = broker.cash
        fill = broker.execute(self.index[i], "buy", quantity, reference)
        self.events.append(ExecutionEvent(fill.timestamp, self.index[i - 1], "buy", "entry_signal",
                                          fill.quantity, fill.price, fill.fee))
        self._entry = (fill.timestamp, fill.price, reference, fill.quantity * fill.price + fill.fee,
                       equity_before, fraction)

    def _sell(self, i: int, reference: float, reason: str, decision: pd.Timestamp) -> None:
        assert self._entry is not None
        entry_time, entry_price, _, entry_cost, entry_equity, fraction = self._entry
        fill = self.broker.execute(self.index[i], "sell", self.broker.position, reference)
        self.events.append(ExecutionEvent(fill.timestamp, decision, "sell", reason,
                                          fill.quantity, fill.price, fill.fee))
        pnl = fill.quantity * fill.price - fill.fee - entry_cost
        self.round_trips.append(RoundTrip(entry_time, fill.timestamp, entry_price, fill.price,
                                          fill.quantity, fraction, pnl, pnl / entry_equity, reason))
        self._entry = None

    def step(self, i: int) -> float:
        """Process bar ``i`` (fills at its open, intrabar stop, close mark); return equity."""
        if i != len(self.equity):
            raise ValueError("bars must be stepped in order")
        if i > 0:
            reference = self.opens[i] if self.policy.fill == "next_open" else self.closes[i]
            previous = self.index[i - 1]
            if not self.desired[i - 1]:
                self._stopped_out = False
            want = bool(self.desired[i - 1]) and not self.halted
            if self.in_position and not want:
                reason = ("risk_halt" if self._external_halt
                          else "max_drawdown" if self.halted_at is not None else "exit_signal")
                decision = previous if reason == "exit_signal" else (self.halted_at or previous)
                self._sell(i, reference, reason, decision)
            elif not self.in_position and want and not self._stopped_out:
                fraction = target_fraction(self.strategy, self.closes, i - 1, self.policy)
                if fraction > 0:
                    self._buy(i, reference, fraction)
            stop = self.strategy.risk.stop_loss
            if self.in_position and stop is not None:
                assert self._entry is not None
                entry_time, _, entry_reference, *_ = self._entry
                stop_price = entry_reference * (1.0 - stop)
                if self.lows[i] <= stop_price:
                    gapped = (self.policy.fill == "next_open" and entry_time != self.index[i]
                              and self.opens[i] <= stop_price)
                    self._sell(i, self.opens[i] if gapped else stop_price, "stop_loss",
                               self.index[i])
                    self._stopped_out = True
        snapshot = self.broker.mark(self.index[i], float(self.closes[i]))
        equity = snapshot.equity
        self.snapshots.append(snapshot)
        self.equity.append(equity)
        self._peak = max(self._peak, equity)
        kill = self.strategy.risk.max_drawdown
        if kill is not None and self.halted_at is None and equity / self._peak - 1.0 <= -kill:
            self.halted_at = self.index[i]
        return equity

    def liquidate(self, i: int, reason: str) -> None:
        """Close an open position at bar ``i``'s close (end of sample or final-bar halt)."""
        if reason not in {"end_of_sample", "risk_halt", "max_drawdown"}:
            raise ValueError(f"invalid liquidation reason: {reason}")
        if self.in_position:
            self._sell(i, float(self.closes[i]), reason, self.index[i])
            self.snapshots[i] = self.broker.mark(self.index[i], float(self.closes[i]))
            self.equity[i] = self.snapshots[i].equity

    def open_round_trip(self) -> RoundTrip | None:
        """Mark-to-market view of a still-open position at the final close (no costs)."""
        if self._entry is None:
            return None
        entry_time, entry_price, _, entry_cost, entry_equity, fraction = self._entry
        last = float(self.closes[len(self.equity) - 1])
        pnl = self.broker.position * last - entry_cost
        return RoundTrip(entry_time, self.index[len(self.equity) - 1], entry_price, last,
                         self.broker.position, fraction, pnl, pnl / entry_equity, "open")


def run_sleeve(strategy: StrategySpec, data: pd.DataFrame, desired: pd.Series,
               broker: PaperBroker, policy: ExecutionPolicy | None = None,
               *, liquidate_at_end: bool = False) -> SleeveEngine:
    """Step a sleeve over every bar, optionally liquidating at the final close."""
    engine = SleeveEngine(strategy, data, desired, broker, policy)
    for i in range(len(data)):
        engine.step(i)
    if liquidate_at_end and engine.in_position:
        last = len(data) - 1
        reason = "max_drawdown" if engine.halted_at is not None else "end_of_sample"
        engine.liquidate(last, reason)
    return engine
