from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .dataset_bundle import DATA_SCHEMA_VERSION, DatasetBundleIdentity, bundle_identity
from .provider import DataRequest, MarketDataProvider


@dataclass(frozen=True)
class MarketDataSnapshot:
    dataset_id: str
    bundle: DatasetBundleIdentity
    data: dict[str, pd.DataFrame]

    def __post_init__(self) -> None:
        if self.dataset_id != self.bundle.dataset_id:
            raise ValueError("snapshot dataset_id must match its bundle identity")
        if set(self.data) != set(self.bundle.symbols):
            raise ValueError("snapshot symbols must match its bundle identity")


def ingest_requests(
    provider: MarketDataProvider,
    dataset_id: str,
    requests: list[DataRequest],
    *,
    source: str = "unspecified",
    timeframe: str = "1d",
    schema_version: str = DATA_SCHEMA_VERSION,
) -> MarketDataSnapshot:
    """Load each requested range once, validate it, and fingerprint its contract."""
    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty")
    if not requests:
        raise ValueError("at least one data request is required")
    if len({request.symbol for request in requests}) != len(requests):
        raise ValueError("data requests must contain unique symbols")
    data = {
        request.symbol: provider.load(request.symbol, request.start, request.end)
        for request in requests
    }
    if any(frame.empty for frame in data.values()):
        raise ValueError("requested market-data range is empty")
    bundle = bundle_identity(
        data,
        dataset_id,
        source=source,
        timeframe=timeframe,
        schema_version=schema_version,
    )
    return MarketDataSnapshot(dataset_id=dataset_id, bundle=bundle, data=data)


def ingest(
    provider: MarketDataProvider,
    dataset_id: str,
    symbols: list[str],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    source: str = "unspecified",
    timeframe: str = "1d",
    schema_version: str = DATA_SCHEMA_VERSION,
) -> MarketDataSnapshot:
    """Load, validate, and fingerprint a reproducible multi-symbol snapshot."""
    requests = [DataRequest(symbol, start, end) for symbol in symbols]
    return ingest_requests(
        provider,
        dataset_id,
        requests,
        source=source,
        timeframe=timeframe,
        schema_version=schema_version,
    )
