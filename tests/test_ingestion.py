import pandas as pd
import pytest

from atsf.ingestion import ingest
from atsf.ingestion_registry import ingest_and_register
from atsf.provider import DataRequest, FrameMarketDataProvider
from atsf.registry import ExperimentRegistry


def frame(offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    close = [100, 101, 102, 101, 103]
    return pd.DataFrame(
        {
            "open": [value + offset for value in close],
            "high": [value + 1 + offset for value in close],
            "low": [value - 1 + offset for value in close],
            "close": [value + offset for value in close],
            "volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=index,
    )


def test_ingest_returns_validated_fingerprinted_snapshot():
    snapshot = ingest(
        FrameMarketDataProvider({"AAA": frame(), "BBB": frame(10)}),
        "market",
        ["BBB", "AAA"],
        source="fixture",
        timeframe="1d",
    )
    assert snapshot.dataset_id == "market"
    assert snapshot.bundle.symbols == ("AAA", "BBB")
    assert snapshot.bundle.rows == 10
    assert snapshot.bundle.source == "fixture"
    assert snapshot.bundle.timeframe == "1d"
    assert snapshot.bundle.schema_version == "ohlcv.v1"
    assert set(snapshot.data) == {"AAA", "BBB"}


def test_ingest_range_is_part_of_the_snapshot():
    provider = FrameMarketDataProvider({"AAA": frame()})
    full = ingest(provider, "market", ["AAA"])
    partial = ingest(provider, "market", ["AAA"], start=pd.Timestamp("2026-01-02"))
    assert full.bundle.version != partial.bundle.version
    assert partial.bundle.rows == 4


def test_ingest_rejects_duplicate_symbols():
    provider = FrameMarketDataProvider({"AAA": frame()})
    with pytest.raises(ValueError, match="unique"):
        ingest(provider, "market", ["AAA", "AAA"])


def test_ingest_and_register_fingerprints_exact_requested_ranges():
    provider = FrameMarketDataProvider({"AAA": frame(), "BBB": frame(10)})
    registry = ExperimentRegistry()
    requests = [
        DataRequest("AAA", pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")),
        DataRequest("BBB", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-04")),
    ]
    snapshot, record = ingest_and_register(
        provider,
        registry._connection,
        "requested-market",
        requests,
        source="fixture",
        timeframe="1d",
    )
    assert snapshot.bundle.version == record.version
    assert snapshot.bundle.rows == 6
    assert record.source == "fixture"
    assert record.timeframe == "1d"
    assert record.schema_version == "ohlcv.v1"
    assert registry.require_dataset("requested-market", record.version) == record
    assert all(len(frame) == 3 for frame in snapshot.data.values())
    registry.close()
