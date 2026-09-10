from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .data import DatasetIdentity, dataset_identity, validate_market_data


@dataclass(frozen=True)
class DatasetBundleIdentity:
    dataset_id: str
    version: str
    symbols: tuple[str, ...]
    rows: int
    start: str
    end: str


def bundle_identity(data: dict[str, pd.DataFrame], dataset_id: str) -> DatasetBundleIdentity:
    """Create a deterministic identity for a multi-symbol market-data snapshot."""
    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty")
    if not data:
        raise ValueError("data bundle cannot be empty")
    if any(not symbol.strip() for symbol in data):
        raise ValueError("symbol identifiers cannot be empty")

    digest = hashlib.sha256()
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
    )
