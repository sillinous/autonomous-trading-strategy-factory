from __future__ import annotations

import sqlite3

import pandas as pd

from .dataset_bundle import bundle_identity
from .dataset_registry import DatasetRecord, DatasetRegistry
from .ingestion import MarketDataSnapshot, ingest
from .provider import DataRequest, MarketDataProvider


def ingest_and_register(
    provider: MarketDataProvider,
    registry_connection: sqlite3.Connection,
    dataset_id: str,
    requests: list[DataRequest],
    *,
    source: str,
) -> tuple[MarketDataSnapshot, DatasetRecord]:
    """Ingest normalized provider data and atomically register its fingerprint."""
    if not requests:
        raise ValueError("at least one data request is required")
    snapshot = ingest(provider, dataset_id, [request.symbol for request in requests])
    requested = {
        request.symbol: provider.load(request.symbol, request.start, request.end)
        for request in requests
    }
    if any(frame.empty for frame in requested.values()):
        raise ValueError("requested market-data range is empty")
    data: dict[str, pd.DataFrame] = {}
    for symbol, frame in requested.items():
        data[symbol] = frame
    identity = bundle_identity(data, dataset_id)
    if identity.version != snapshot.bundle.version:
        raise ValueError("provider data changed during ingestion")
    record = DatasetRegistry(registry_connection).register(identity, source=source)
    return snapshot, record
