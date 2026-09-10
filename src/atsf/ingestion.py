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


def ingest(
    provider: MarketDataProvider,
    dataset_id: str,
    symbols: list[str],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> MarketDataSnapshot:
    """Load, validate, and fingerprint a reproducible multi-symbol snapshot."""
    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty")
    if not symbols:
        raise ValueError("at least one symbol is required")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")
    requests = [DataRequest(symbol, start, end) for symbol in symbols]
    data = {request.symbol: provider.load(request.symbol, request.start, request.end) for request in requests}
    bundle = bundle_identity(data, dataset_id)
    return MarketDataSnapshot(dataset_id=dataset_id, bundle=bundle, data=data)
