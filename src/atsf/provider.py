from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd

from .data import validate_market_data


class MarketDataProvider(Protocol):
    """Source-neutral contract for normalized historical market data."""

    def load(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        """Return validated OHLCV data for one symbol."""


@dataclass(frozen=True)
class DataRequest:
    symbol: str
    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("start must be before end")


class FrameMarketDataProvider:
    """Deterministic provider backed by normalized in-process frames.

    Vendor adapters should implement MarketDataProvider and normalize their output
    through ``validate_market_data`` before returning it.
    """

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        if not frames:
            raise ValueError("at least one symbol is required")
        self._frames = {symbol: validate_market_data(frame) for symbol, frame in frames.items()}

    def load(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        if symbol not in self._frames:
            raise KeyError(f"unknown symbol: {symbol}")
        frame = self._frames[symbol]
        if start is not None:
            frame = frame.loc[frame.index >= start]
        if end is not None:
            frame = frame.loc[frame.index < end]
        if frame.empty:
            raise ValueError("requested market-data range is empty")
        return frame.copy()
