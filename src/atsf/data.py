from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    version: str
    rows: int
    start: str
    end: str


def validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize OHLCV data without applying vendor-specific rules."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("market data index must be a DatetimeIndex")
    if data.empty:
        raise ValueError("market data cannot be empty")
    if not data.index.is_monotonic_increasing:
        raise ValueError("market data index must be monotonically increasing")
    if data.index.has_duplicates:
        raise ValueError("market data index must not contain duplicates")
    if data.index.tz is not None:
        raise ValueError("market data index must be timezone-naive")

    normalized = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        if not normalized[column].map(math.isfinite).all():
            raise ValueError("market data contains non-finite values")
    if normalized.isna().any().any():
        raise ValueError("market data contains missing values")
    if (normalized["volume"] < 0).any():
        raise ValueError("volume cannot be negative")
    if (normalized["low"] > normalized[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must not exceed OHLC values")
    if (normalized["high"] < normalized[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must not be below OHLC values")
    return normalized


def dataset_identity(data: pd.DataFrame, dataset_id: str) -> DatasetIdentity:
    """Return a content-derived version for reproducible experiment references."""
    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty")
    normalized = validate_market_data(data)
    payload = normalized.to_csv(index=True, date_format="%Y-%m-%dT%H:%M:%S.%f").encode("utf-8")
    version = hashlib.sha256(payload).hexdigest()[:16]
    return DatasetIdentity(
        dataset_id=dataset_id,
        version=version,
        rows=len(normalized),
        start=normalized.index[0].isoformat(),
        end=normalized.index[-1].isoformat(),
    )


def load_csv(source: str | bytes | io.BytesIO) -> pd.DataFrame:
    """Load a CSV into the canonical OHLCV representation."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    data = pd.read_csv(source, index_col=0, parse_dates=True)
    return validate_market_data(data)
