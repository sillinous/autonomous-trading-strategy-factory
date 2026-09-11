from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data import validate_market_data


@dataclass(frozen=True)
class CacheKey:
    symbol: str
    start: datetime | None
    end: datetime | None
    source: str
    timeframe: str
    schema_version: str

    @property
    def value(self) -> str:
        payload = {
            "symbol": self.symbol,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "source": self.source,
            "timeframe": self.timeframe,
            "schema_version": self.schema_version,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


class MarketDataCache:
    """Filesystem cache for validated provider responses.

    Cached frames remain immutable inputs: every read is validated and returned as
    a copy, preventing accidental mutation of the reproducibility boundary.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: CacheKey) -> Path:
        safe_symbol = key.symbol.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe_symbol}-{key.value}.csv"

    def get(self, key: CacheKey) -> pd.DataFrame | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        return validate_market_data(pd.read_csv(path, index_col=0, parse_dates=True)).copy()

    def put(self, key: CacheKey, frame: pd.DataFrame) -> Path:
        normalized = validate_market_data(frame)
        path = self.path_for(key)
        temp = path.with_suffix(".tmp")
        normalized.to_csv(temp)
        temp.replace(path)
        return path
