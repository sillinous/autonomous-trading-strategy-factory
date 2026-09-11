import sqlite3

import pandas as pd
import pytest

from atsf.ingestion_registry import ingest_and_register
from atsf.provider import DataRequest, FrameMarketDataProvider


def frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    close = [100, 101, 102, 101, 103]
    return pd.DataFrame(
        {
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "volume": [1000] * 5,
        },
        index=index,
    )


def test_ingest_and_register_uses_requested_range_and_provider_metadata():
    connection = sqlite3.connect(":memory:")
    provider = FrameMarketDataProvider({"AAA": frame()}, source="fixture")
    snapshot, record = ingest_and_register(
        provider,
        connection,
        "market",
        [DataRequest("AAA", pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05"))],
        source="fixture",
    )
    assert snapshot.bundle.rows == 4
    assert record.version == snapshot.bundle.version
    assert record.rows == 4
    assert record.source == "fixture"
    connection.close()


def test_ingest_and_register_rejects_provenance_override():
    connection = sqlite3.connect(":memory:")
    provider = FrameMarketDataProvider({"AAA": frame()}, source="fixture")
    with pytest.raises(ValueError, match="source assertion"):
        ingest_and_register(provider, connection, "market", [DataRequest("AAA")], source="spoofed")
    connection.close()


def test_ingest_and_register_rejects_duplicate_requests():
    connection = sqlite3.connect(":memory:")
    provider = FrameMarketDataProvider({"AAA": frame()}, source="fixture")
    with pytest.raises(ValueError, match="unique"):
        ingest_and_register(
            provider,
            connection,
            "market",
            [DataRequest("AAA"), DataRequest("AAA")],
            source="fixture",
        )
    connection.close()
