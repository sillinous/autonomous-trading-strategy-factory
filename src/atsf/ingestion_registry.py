from __future__ import annotations

import sqlite3

from .dataset_bundle import DATA_SCHEMA_VERSION, bundle_identity
from .dataset_registry import DatasetRecord, DatasetRegistry
from .ingestion import MarketDataSnapshot
from .provider import DataRequest, MarketDataProvider


def ingest_and_register(
    provider: MarketDataProvider,
    registry_connection: sqlite3.Connection,
    dataset_id: str,
    requests: list[DataRequest],
    *,
    source: str,
    timeframe: str = "1d",
    schema_version: str = DATA_SCHEMA_VERSION,
) -> tuple[MarketDataSnapshot, DatasetRecord]:
    """Ingest requested ranges once and atomically register their fingerprint."""
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

    requested_bundle = bundle_identity(
        data,
        dataset_id,
        source=source,
        timeframe=timeframe,
        schema_version=schema_version,
    )
    record = DatasetRegistry(registry_connection).register(requested_bundle, source=source)
    snapshot = MarketDataSnapshot(
        dataset_id=dataset_id,
        bundle=requested_bundle,
        data=data,
    )
    return snapshot, record
