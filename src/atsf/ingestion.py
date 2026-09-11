from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .dataset_bundle import DatasetBundleIdentity, bundle_identity
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


def _provider_contract(
    provider: MarketDataProvider,
    *,
    source: str | None,
    timeframe: str | None,
    schema_version: str | None,
) -> tuple[str, str, str]:
    metadata = provider.metadata
    if source is not None and source.strip() != metadata.source:
        raise ValueError("source assertion does not match provider metadata")
    if timeframe is not None and timeframe.strip() != metadata.timeframe:
        raise ValueError("timeframe assertion does not match provider metadata")
    if schema_version is not None and schema_version.strip() != metadata.schema_version:
        raise ValueError("schema_version assertion does not match provider metadata")
    return metadata.source, metadata.timeframe, metadata.schema_version


def ingest_requests(
    provider: MarketDataProvider,
    dataset_id: str,
    requests: list[DataRequest],
    *,
    source: str | None = None,
    timeframe: str | None = None,
    schema_version: str | None = None,
) -> MarketDataSnapshot:
    """Load requested ranges and fingerprint provider-declared provenance.

    Optional contract arguments are assertions only; they can never override the
    provider's metadata.
    """
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
    bundle = bundle_identity(
        data,
        dataset_id,
        source=provider_source,
        timeframe=provider_timeframe,
        schema_version=provider_schema,
    )
    return MarketDataSnapshot(dataset_id=dataset_id, bundle=bundle, data=data)


def ingest(
    provider: MarketDataProvider,
    dataset_id: str,
    symbols: list[str],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    source: str | None = None,
    timeframe: str | None = None,
    schema_version: str | None = None,
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
