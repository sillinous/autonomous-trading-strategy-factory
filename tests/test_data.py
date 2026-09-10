import pandas as pd
import pytest

from atsf.data import dataset_identity, load_csv, validate_market_data


def make_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame(
        {
            "open": [10, 11, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "close": [10.5, 11.5, 12.5],
            "volume": [100, 120, 140],
        },
        index=index,
    )


def test_dataset_identity_is_reproducible():
    data = make_data()
    first = dataset_identity(data, "fixture")
    second = dataset_identity(data.copy(), "fixture")
    assert first == second
    assert first.rows == 3
    assert first.start.startswith("2024-01-01")


def test_market_data_rejects_duplicate_timestamps():
    data = make_data()
    data.index = [data.index[0], data.index[0], data.index[2]]
    with pytest.raises(ValueError, match="duplicates"):
        validate_market_data(data)


def test_market_data_rejects_timezone_aware_timestamps():
    data = make_data()
    data.index = data.index.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        validate_market_data(data)


def test_market_data_rejects_invalid_ohlc():
    data = make_data()
    data.loc[data.index[1], "low"] = 99
    with pytest.raises(ValueError, match="low"):
        validate_market_data(data)


def test_csv_loader_normalizes_canonical_columns():
    csv = "timestamp,open,high,low,close,volume\n2024-01-01,10,11,9,10.5,100\n"
    loaded = load_csv(csv.encode())
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(loaded.index, pd.DatetimeIndex)
