from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, nextafter

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
        self.config = config or PaperConfig()
        if self.config.initial_cash <= 0 or not isfinite(self.config.initial_cash):
            raise ValueError("initial_cash must be positive and finite")
        if self.config.commission_bps < 0 or not isfinite(self.config.commission_bps):
            raise ValueError("commission_bps must be non-negative and finite")
        if self.config.slippage_bps < 0 or not isfinite(self.config.slippage_bps):
            raise ValueError("slippage_bps must be non-negative and finite")
        self.cash = self.config.initial_cash
        self.position = 0.0
        self.fills: list[PaperFill] = []

    def mark(self, timestamp: pd.Timestamp, price: float) -> PaperSnapshot:
        if price <= 0 or not isfinite(price):
            raise ValueError("price must be positive and finite")
        return PaperSnapshot(timestamp, self.cash, self.position, self.cash + self.position * price)

    def max_affordable_quantity(self, reference_price: float) -> float:
        """Return a buy quantity safely below the cash boundary after costs."""
        if reference_price <= 0 or not isfinite(reference_price):
            raise ValueError("price must be positive and finite")
        fill_price = reference_price * (1.0 + self.config.slippage_bps / 10_000.0)
        cost_per_unit = fill_price * (1.0 + self.config.commission_bps / 10_000.0)
        quantity = self.cash / cost_per_unit
        # Avoid a one-ULP overrun when execute() recomputes the same arithmetic.
        return max(0.0, nextafter(quantity, 0.0))

    def execute(self, timestamp: pd.Timestamp, side: str, quantity: float, reference_price: float) -> PaperFill:
        if side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if quantity <= 0 or not isfinite(quantity) or reference_price <= 0 or not isfinite(reference_price):
            raise ValueError("quantity and price must be positive and finite")
        slippage = self.config.slippage_bps / 10_000.0
        fill_price = reference_price * (1.0 + slippage if side == "buy" else 1.0 - slippage)
        fee = fill_price * quantity * self.config.commission_bps / 10_000.0
        if side == "buy":
            required = fill_price * quantity + fee
            if required > self.cash + 1e-12:
                raise ValueError("insufficient paper cash")
            self.cash -= required
            self.position += quantity
        else:
            if quantity > self.position + 1e-12:
                raise ValueError("insufficient paper position")
            self.cash += fill_price * quantity - fee
            self.position = max(0.0, self.position - quantity)
        fill = PaperFill(timestamp, side, quantity, fill_price, fee)
        self.fills.append(fill)
        return fill
