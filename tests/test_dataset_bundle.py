import pandas as pd
import pytest

from atsf.dataset_bundle import DATA_SCHEMA_VERSION, bundle_identity


def frame(offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "open": [100 + offset, 101 + offset, 102 + offset],
            "high": [101 + offset, 102 + offset, 103 + offset],
            "low": [99 + offset, 100 + offset, 101 + offset],
            "close": [100.5 + offset, 101.5 + offset, 102.5 + offset],
            "volume": [1000, 1100, 1200],
        },
        index=index,
    )


def test_bundle_identity_is_deterministic_and_order_independent():
    first = bundle_identity(
        {"AAA": frame(), "BBB": frame(10)}, "market", source="fixture", timeframe="1d"
    )
    second = bundle_identity(
        {"BBB": frame(10), "AAA": frame()}, "market", source="fixture", timeframe="1d"
    )
    assert first == second
    assert first.symbols == ("AAA", "BBB")
    assert first.rows == 6
    assert first.source == "fixture"
    assert first.timeframe == "1d"
    assert first.schema_version == DATA_SCHEMA_VERSION


def test_bundle_identity_changes_when_data_changes():
    first = bundle_identity({"AAA": frame()}, "market")
    second = bundle_identity({"AAA": frame(1)}, "market")
    assert first.version != second.version


def test_bundle_identity_changes_when_contract_changes():
    data = {"AAA": frame()}
    base = bundle_identity(data, "market", source="vendor-a", timeframe="1d")
    other_source = bundle_identity(data, "market", source="vendor-b", timeframe="1d")
    other_timeframe = bundle_identity(data, "market", source="vendor-a", timeframe="1h")
    other_schema = bundle_identity(data, "market", source="vendor-a", timeframe="1d", schema_version="ohlcv.v2")
    assert len({base.version, other_source.version, other_timeframe.version, other_schema.version}) == 4


def test_bundle_identity_rejects_empty_bundle():
    with pytest.raises(ValueError, match="cannot be empty"):
        bundle_identity({}, "market")
