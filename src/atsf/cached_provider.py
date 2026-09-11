from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from .data_cache import CacheKey, MarketDataCache


class CacheableMarketDataProvider(Protocol):
    def load(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        ...


class CachedMarketDataProvider:
    """Cache decorator preserving the source provider as the authoritative fetcher."""

    def __init__(
        self,
        provider: CacheableMarketDataProvider,
        cache: MarketDataCache,
        *,
        source: str,
        timeframe: str = "1d",
        schema_version: str = "ohlcv.v1",
    ) -> None:
        if not source.strip():
            raise ValueError("source cannot be empty")
        if not timeframe.strip():
            raise ValueError("timeframe cannot be empty")
        if not schema_version.strip():
            raise ValueError("schema_version cannot be empty")
        self.provider = provider
        self.cache = cache
        self.source = source.strip()
        self.timeframe = timeframe.strip()
        self.schema_version = schema_version.strip()

    def load(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        key = CacheKey(symbol, start, end, self.source, self.timeframe, self.schema_version)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        frame = self.provider.load(symbol, start, end)
        self.cache.put(key, frame)
        return frame.copy()
