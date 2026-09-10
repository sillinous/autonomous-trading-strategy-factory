from __future__ import annotations

import sqlite3

from .dataset_bundle import bundle_identity
from .dataset_registry import DatasetRecord, DatasetRegistry
from .ingestion import MarketDataSnapshot
from .provider import DataRequest, MarketDataProvider
from .ingestion import ingest


def ingest_and_register(
    provider: MarketDataProvider,
    registry_connection: sqlite3.Connection,
    dataset_id: str,
    requests: list[DataRequest],
    *,
    source: str,
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
    snapshot = ingest(provider, dataset_id, [request.symbol for request in requests])
    requested_bundle = bundle_identity(data, dataset_id)
    if requested_bundle.version != snapshot.bundle.version:
        raise ValueError("ingestion snapshot does not match requested ranges")
    record = DatasetRegistry(registry_connection).register(requested_bundle, source=source)
    return MarketDataSnapshot(dataset_id=dataset_id, bundle=requested_bundle, data=data), record
