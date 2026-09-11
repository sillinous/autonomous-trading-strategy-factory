from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from .data_cache import CacheKey, MarketDataCache
from .provider import ProviderMetadata


class CacheableMarketDataProvider(Protocol):
    @property
    def metadata(self) -> ProviderMetadata:
        ...

    def load(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        ...


class CachedMarketDataProvider:
    """Cache decorator preserving the source provider as authoritative provenance."""

    def __init__(self, provider: CacheableMarketDataProvider, cache: MarketDataCache) -> None:
        self.provider = provider
        self.cache = cache

    @property
    def metadata(self) -> ProviderMetadata:
        return self.provider.metadata

    def load(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        metadata = self.metadata
        key = CacheKey(
            symbol,
            start,
            end,
            metadata.source,
            metadata.timeframe,
            metadata.schema_version,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        frame = self.provider.load(symbol, start, end)
        self.cache.put(key, frame)
        return frame.copy()
