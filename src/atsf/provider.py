from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd

from .data import validate_market_data


@dataclass(frozen=True)
class ProviderMetadata:
    """Immutable provenance declared by the provider itself."""

    source: str
    timeframe: str
    schema_version: str = "ohlcv.v1"

    def __post_init__(self) -> None:
        for name in ("source", "timeframe", "schema_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is required")


class MarketDataProvider(Protocol):
    @property
    def metadata(self) -> ProviderMetadata: ...
    def load(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame: ...


@dataclass(frozen=True)
class DataRequest:
    symbol: str
    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end")


class FrameMarketDataProvider:
    """Deterministic provider backed by normalized in-process frames."""

    def __init__(self, frames: dict[str, pd.DataFrame], *, source: str = "frame", timeframe: str = "1d", schema_version: str = "ohlcv.v1") -> None:
        if not frames:
            raise ValueError("at least one symbol is required")
        self._frames = {symbol: validate_market_data(frame) for symbol, frame in frames.items()}
        self._metadata = ProviderMetadata(source, timeframe, schema_version)

    @property
    def metadata(self) -> ProviderMetadata:
        return self._metadata

    def load(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        if symbol not in self._frames:
            raise KeyError(f"unknown symbol: {symbol}")
        frame = self._frames[symbol]
        if start is not None:
            frame = frame.loc[frame.index >= start]
        # Historical API semantics use an exclusive upper bound.
        if end is not None:
            frame = frame.loc[frame.index < end]
        if frame.empty:
            raise ValueError("requested market-data range is empty")
        return frame.copy()
