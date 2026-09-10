from __future__ import annotations

import sqlite3

from .dataset_registry import DatasetRecord, DatasetRegistry
from .ingestion import MarketDataSnapshot, ingest_requests
from .provider import DataRequest, MarketDataProvider


def ingest_and_register(
    provider: MarketDataProvider,
    registry_connection: sqlite3.Connection,
    dataset_id: str,
    requests: list[DataRequest],
    *,
    source: str,
) -> tuple[MarketDataSnapshot, DatasetRecord]:
    """Ingest requested ranges once and atomically register their fingerprint."""
    snapshot = ingest_requests(provider, dataset_id, requests)
    record = DatasetRegistry(registry_connection).register(snapshot.bundle, source=source)
    return snapshot, record
