from datetime import datetime

import pandas as pd

from atsf.data_cache import CacheKey, MarketDataCache


def frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {"open": [10.0, 11.0, 12.0], "high": [11.0, 12.0, 13.0], "low": [9.0, 10.0, 11.0], "close": [10.5, 11.5, 12.5], "volume": [100.0, 110.0, 120.0]},
        index=index,
    )


def key() -> CacheKey:
    return CacheKey("AAA", datetime(2024, 1, 1), datetime(2024, 1, 4), "fixture", "1d", "ohlcv.v1")


def test_cache_round_trip_is_deterministic(tmp_path):
    cache = MarketDataCache(tmp_path)
    path = cache.put(key(), frame())
    restored = cache.get(key())
    assert path.exists()
    pd.testing.assert_frame_equal(restored, frame())


def test_cache_key_changes_when_contract_changes():
    base = key()
    assert base.value != CacheKey("AAA", base.start, base.end, "other", "1d", "ohlcv.v1").value
    assert base.value != CacheKey("AAA", base.start, base.end, "fixture", "1h", "ohlcv.v1").value
    assert base.value != CacheKey("AAA", base.start, base.end, "fixture", "1d", "ohlcv.v2").value


def test_cache_miss_returns_none(tmp_path):
    assert MarketDataCache(tmp_path).get(key()) is None
