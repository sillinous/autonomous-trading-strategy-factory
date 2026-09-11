from __future__ import annotations

import sqlite3

from .dataset_bundle import bundle_identity
from .dataset_registry import DatasetRecord, DatasetRegistry
from .ingestion import MarketDataSnapshot, _provider_contract
from .provider import DataRequest, MarketDataProvider


def ingest_and_register(
    provider: MarketDataProvider,
    registry_connection: sqlite3.Connection,
    dataset_id: str,
    requests: list[DataRequest],
    *,
    source: str | None = None,
    timeframe: str | None = None,
    schema_version: str | None = None,
) -> tuple[MarketDataSnapshot, DatasetRecord]:
    """Ingest requested ranges and atomically register provider provenance."""
    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty")
    if not requests:
        raise ValueError("at least one data request is required")
    if len({request.symbol for request in requests}) != len(requests):
        raise ValueError("data requests must contain unique symbols")
    provider_source, provider_timeframe, provider_schema = _provider_contract(
        provider, source=source, timeframe=timeframe, schema_version=schema_version
    )

    data = {
        request.symbol: provider.load(request.symbol, request.start, request.end)
        for request in requests
    }
    if any(frame.empty for frame in data.values()):
        raise ValueError("requested market-data range is empty")

    requested_bundle = bundle_identity(
        data,
        dataset_id,
        source=provider_source,
        timeframe=provider_timeframe,
        schema_version=provider_schema,
    )
    record = DatasetRegistry(registry_connection).register(requested_bundle, source=provider_source)
    snapshot = MarketDataSnapshot(
        dataset_id=dataset_id,
        bundle=requested_bundle,
        data=data,
    )
    return snapshot, record
