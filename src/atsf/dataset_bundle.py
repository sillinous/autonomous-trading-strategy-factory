from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .data import DatasetIdentity, dataset_identity, validate_market_data


DATA_SCHEMA_VERSION = "ohlcv.v1"


@dataclass(frozen=True)
class DatasetBundleIdentity:
    dataset_id: str
    version: str
    symbols: tuple[str, ...]
    rows: int
    start: str
    end: str
    source: str = "unspecified"
    timeframe: str = "1d"
    schema_version: str = DATA_SCHEMA_VERSION


def bundle_identity(
    data: dict[str, pd.DataFrame],
    dataset_id: str,
    *,
    source: str = "unspecified",
    timeframe: str = "1d",
    schema_version: str = DATA_SCHEMA_VERSION,
) -> DatasetBundleIdentity:
    """Create a deterministic identity for a market-data snapshot and its contract."""
    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty")
    if not data:
        raise ValueError("data bundle cannot be empty")
    if not source.strip():
        raise ValueError("source cannot be empty")
    if not timeframe.strip():
        raise ValueError("timeframe cannot be empty")
    if not schema_version.strip():
        raise ValueError("schema_version cannot be empty")
    if any(not symbol.strip() for symbol in data):
        raise ValueError("symbol identifiers cannot be empty")

    normalized_source = source.strip()
    normalized_timeframe = timeframe.strip()
    normalized_schema = schema_version.strip()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "source": normalized_source,
                "timeframe": normalized_timeframe,
                "schema_version": normalized_schema,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    identities: list[DatasetIdentity] = []
    for symbol in sorted(data):
        frame = validate_market_data(data[symbol])
        identity = dataset_identity(frame, symbol)
        identities.append(identity)
        digest.update(symbol.encode("utf-8"))
        digest.update(identity.version.encode("ascii"))
        digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())

    starts = [item.start for item in identities]
    ends = [item.end for item in identities]
    return DatasetBundleIdentity(
        dataset_id=dataset_id,
        version=digest.hexdigest()[:16],
        symbols=tuple(sorted(data)),
        rows=sum(item.rows for item in identities),
        start=min(starts),
        end=max(ends),
        source=normalized_source,
        timeframe=normalized_timeframe,
        schema_version=normalized_schema,
    )
