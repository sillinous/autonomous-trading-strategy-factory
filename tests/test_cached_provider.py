from datetime import datetime

import pandas as pd

from atsf.cached_provider import CachedMarketDataProvider
from atsf.data_cache import MarketDataCache
from atsf.provider import ProviderMetadata


class CountingProvider:
    metadata = ProviderMetadata("fixture", "1d", "ohlcv.v1")

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls = 0

    def load(self, symbol, start=None, end=None):
        self.calls += 1
        return self.frame.copy()


def frame():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {"open": [10, 11, 12], "high": [11, 12, 13], "low": [9, 10, 11], "close": [10.5, 11.5, 12.5], "volume": [100, 110, 120]},
        index=index,
        dtype=float,
    )


def test_cached_provider_fetches_once_and_preserves_metadata(tmp_path):
    source = CountingProvider(frame())
    provider = CachedMarketDataProvider(source, MarketDataCache(tmp_path))
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 4)
    first = provider.load("AAA", start, end)
    second = provider.load("AAA", start, end)
    assert source.calls == 1
    assert provider.metadata == source.metadata
    pd.testing.assert_frame_equal(first, second)


def test_cached_provider_cannot_override_source_contract(tmp_path):
    source = CountingProvider(frame())
    provider = CachedMarketDataProvider(source, MarketDataCache(tmp_path))
    assert provider.metadata.source == "fixture"
    assert provider.metadata.timeframe == "1d"
    assert provider.metadata.schema_version == "ohlcv.v1"
