import json
import sqlite3

from atsf.alphavantage import AlphaVantageDailyProvider
from atsf.ingestion import ingest_requests
from atsf.ingestion_registry import ingest_and_register
from atsf.provider import DataRequest


def transport(_: str) -> bytes:
    return json.dumps(
        {
            "Time Series (Daily)": {
                "2026-01-02": {
                    "1. open": "101",
                    "2. high": "104",
                    "3. low": "99",
                    "4. close": "103",
                    "5. volume": "900",
                },
                "2026-01-05": {
                    "1. open": "105",
                    "2. high": "110",
                    "3. low": "100",
                    "4. close": "108",
                    "5. volume": "1000",
                },
            }
        }
    ).encode()


def test_real_provider_contract_flows_into_fingerprinted_registry() -> None:
    provider = AlphaVantageDailyProvider("key", transport=transport)
    requests = [DataRequest("TEST")]
    snapshot = ingest_requests(
        provider,
        "market",
        requests,
        source=provider.source,
        timeframe=provider.timeframe,
        schema_version=provider.schema_version,
    )
    connection = sqlite3.connect(":memory:")
    registered, record = ingest_and_register(
        provider,
        connection,
        "market",
        requests,
        source=provider.source,
        timeframe=provider.timeframe,
        schema_version=provider.schema_version,
    )

    assert snapshot.bundle.version == registered.bundle.version
    assert record.version == registered.bundle.version
    assert record.source == "alphavantage"
    assert record.timeframe == "1d"
    assert record.schema_version == "ohlcv.v1"
