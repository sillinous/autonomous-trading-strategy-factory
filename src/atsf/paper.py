from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, nextafter
from numbers import Real

import pandas as pd


@dataclass(frozen=True)
class PaperConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0


@dataclass(frozen=True)
class PaperFill:
    timestamp: pd.Timestamp
    side: str
    quantity: float
    price: float
    fee: float


@dataclass(frozen=True)
class PaperSnapshot:
    timestamp: pd.Timestamp
    cash: float
    position: float
    equity: float


class PaperBroker:
    """Deterministic in-memory broker simulator. It cannot submit live orders."""

    def __init__(self, config: PaperConfig | None = None) -> None:
        self.config = config if config is not None else PaperConfig()
        if not isinstance(self.config, PaperConfig):
            raise ValueError("config must be a PaperConfig")
        if (
            not isinstance(self.config.initial_cash, Real)
            or isinstance(self.config.initial_cash, bool)
            or not isfinite(self.config.initial_cash)
            or self.config.initial_cash <= 0
        ):
            raise ValueError("initial_cash must be a positive finite real number")
        if (
            not isinstance(self.config.commission_bps, Real)
            or isinstance(self.config.commission_bps, bool)
            or not isfinite(self.config.commission_bps)
            or self.config.commission_bps < 0
        ):
            raise ValueError("commission_bps must be a finite non-negative real number")
        if (
            not isinstance(self.config.slippage_bps, Real)
            or isinstance(self.config.slippage_bps, bool)
            or not isfinite(self.config.slippage_bps)
            or self.config.slippage_bps < 0
        ):
            raise ValueError("slippage_bps must be a finite non-negative real number")
        if self.config.commission_bps + self.config.slippage_bps >= 5_000:
            raise ValueError("combined transaction costs must be below 5000 bps")
        self.cash = self.config.initial_cash
        self.position = 0.0
        self.fills: list[PaperFill] = []

    def mark(self, timestamp: pd.Timestamp, price: float) -> PaperSnapshot:
        if (
            not isinstance(price, Real)
            or isinstance(price, bool)
            or not isfinite(price)
            or price <= 0
        ):
            raise ValueError("price must be a positive finite real number")
        equity = self.cash + self.position * price
        if not isfinite(equity):
            raise ValueError("marked equity must remain finite")
        return PaperSnapshot(timestamp, self.cash, self.position, equity)

    def max_affordable_quantity(self, reference_price: float) -> float:
        """Return a buy quantity safely below the cash boundary after costs."""
        if (
            not isinstance(reference_price, Real)
            or isinstance(reference_price, bool)
            or not isfinite(reference_price)
            or reference_price <= 0
        ):
            raise ValueError("price must be a positive finite real number")
        fill_price = reference_price * (1.0 + self.config.slippage_bps / 10_000.0)
        cost_per_unit = fill_price * (1.0 + self.config.commission_bps / 10_000.0)
        if not isfinite(fill_price) or not isfinite(cost_per_unit) or cost_per_unit <= 0:
            raise ValueError("affordability arithmetic must remain finite")
        quantity = self.cash / cost_per_unit
        if not isfinite(quantity):
            raise ValueError("affordable quantity must remain finite")
        # Avoid a one-ULP overrun when execute() recomputes the same arithmetic.
        return max(0.0, nextafter(quantity, 0.0))

    def execute(self, timestamp: pd.Timestamp, side: str, quantity: float, reference_price: float) -> PaperFill:
        if not isinstance(side, str) or side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if (
            not isinstance(quantity, Real)
            or isinstance(quantity, bool)
            or not isfinite(quantity)
            or quantity <= 0
            or not isinstance(reference_price, Real)
            or isinstance(reference_price, bool)
            or not isfinite(reference_price)
            or reference_price <= 0
        ):
            raise ValueError("quantity and price must be positive finite real numbers")
        slippage = self.config.slippage_bps / 10_000.0
        fill_price = reference_price * (1.0 + slippage if side == "buy" else 1.0 - slippage)
        fee = fill_price * quantity * self.config.commission_bps / 10_000.0
        notional = fill_price * quantity
        if not isfinite(fill_price) or not isfinite(fee) or not isfinite(notional):
            raise ValueError("execution arithmetic must remain finite")
        if side == "buy":
            required = notional + fee
            if not isfinite(required):
                raise ValueError("execution cost must remain finite")
            if required > self.cash + 1e-12:
                raise ValueError("insufficient paper cash")
            new_cash = self.cash - required
            new_position = self.position + quantity
            if not isfinite(new_cash) or not isfinite(new_position):
                raise ValueError("post-trade cash and position must remain finite")
            self.cash = new_cash
            self.position = new_position
        else:
            if quantity > self.position + 1e-12:
                raise ValueError("insufficient paper position")
            proceeds = notional - fee
            new_cash = self.cash + proceeds
            new_position = max(0.0, self.position - quantity)
            if not isfinite(proceeds) or not isfinite(new_cash) or not isfinite(new_position):
                raise ValueError("post-trade cash and position must remain finite")
            self.cash = new_cash
            self.position = new_position
        fill = PaperFill(timestamp, side, quantity, fill_price, fee)
        self.fills.append(fill)
        return fill
