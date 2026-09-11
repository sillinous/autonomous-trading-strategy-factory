import json

import pandas as pd
import pytest

from atsf.alphavantage import AlphaVantageDailyProvider


PAYLOAD = {
    "Meta Data": {"2. Symbol": "TEST"},
    "Time Series (Daily)": {
        "2026-01-05": {
            "1. open": "105",
            "2. high": "110",
            "3. low": "100",
            "4. close": "108",
            "5. volume": "1000",
        },
        "2026-01-02": {
            "1. open": "101",
            "2. high": "104",
            "3. low": "99",
            "4. close": "103",
            "5. volume": "900",
        },
    },
}


def transport(url: str) -> bytes:
    assert "function=TIME_SERIES_DAILY" in url
    assert "symbol=TEST" in url
    return json.dumps(PAYLOAD).encode()


def test_alpha_vantage_daily_provider_normalizes_and_filters() -> None:
    provider = AlphaVantageDailyProvider("key", transport=transport)
    result = provider.load(
        "TEST",
        start=pd.Timestamp("2026-01-03"),
        end=pd.Timestamp("2026-01-06"),
    )

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert list(result.index) == [pd.Timestamp("2026-01-05")]
    assert result.iloc[0]["close"] == 108
    assert provider.source == "alphavantage"
    assert provider.timeframe == "1d"
    assert provider.schema_version == "ohlcv.v1"


def test_alpha_vantage_daily_provider_rejects_vendor_error() -> None:
    provider = AlphaVantageDailyProvider(
        "key", transport=lambda _: json.dumps({"Error Message": "bad symbol"}).encode()
    )
    with pytest.raises(ValueError, match="bad symbol"):
        provider.load("TEST")


def test_alpha_vantage_daily_provider_rejects_rate_limit() -> None:
    provider = AlphaVantageDailyProvider(
        "key", transport=lambda _: json.dumps({"Note": "rate limited"}).encode()
    )
    with pytest.raises(RuntimeError, match="rate limit"):
        provider.load("TEST")


def test_alpha_vantage_daily_provider_requires_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        AlphaVantageDailyProvider(" ")
