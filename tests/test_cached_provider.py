from datetime import datetime

import pandas as pd

from atsf.cached_provider import CachedMarketDataProvider
from atsf.data_cache import MarketDataCache


class CountingProvider:
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


def test_cached_provider_fetches_once(tmp_path):
    source = CountingProvider(frame())
    provider = CachedMarketDataProvider(source, MarketDataCache(tmp_path), source="fixture")
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 4)
    first = provider.load("AAA", start, end)
    second = provider.load("AAA", start, end)
    assert source.calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_cached_provider_contract_changes_use_distinct_entries(tmp_path):
    source = CountingProvider(frame())
    cache = MarketDataCache(tmp_path)
    daily = CachedMarketDataProvider(source, cache, source="fixture", timeframe="1d")
    hourly = CachedMarketDataProvider(source, cache, source="fixture", timeframe="1h")
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 4)
    daily.load("AAA", start, end)
    hourly.load("AAA", start, end)
    assert source.calls == 2
