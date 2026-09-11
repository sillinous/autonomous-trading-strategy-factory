import pandas as pd
import pytest

from atsf.provider import DataRequest, FrameMarketDataProvider, ProviderMetadata


def frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "open": [1, 2, 3, 4],
            "high": [1, 2, 3, 4],
            "low": [1, 2, 3, 4],
            "close": [1, 2, 3, 4],
            "volume": [10, 10, 10, 10],
        },
        index=index,
    )


def test_provider_validates_and_filters_range():
    provider = FrameMarketDataProvider({"TEST": frame()})
    result = provider.load("TEST", pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-04"))
    assert list(result["close"]) == [2, 3]
    result.iloc[0, result.columns.get_loc("close")] = 999
    assert provider.load("TEST").iloc[0]["close"] == 1


def test_provider_exposes_immutable_metadata():
    provider = FrameMarketDataProvider(
        {"TEST": frame()}, source="fixture", timeframe="1d", schema_version="ohlcv.v2"
    )
    assert provider.metadata == ProviderMetadata("fixture", "1d", "ohlcv.v2")
    with pytest.raises(Exception):
        provider.metadata.source = "spoofed"


def test_provider_rejects_unknown_or_empty_range():
    provider = FrameMarketDataProvider({"TEST": frame()})
    with pytest.raises(KeyError):
        provider.load("MISSING")
    with pytest.raises(ValueError):
        provider.load("TEST", pd.Timestamp("2027-01-01"))


def test_data_request_validates_symbol_and_order():
    assert DataRequest("TEST")
    with pytest.raises(ValueError):
        DataRequest(" ")
    with pytest.raises(ValueError):
        DataRequest("TEST", pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-01"))
