from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

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
        payload = {"symbol": self.symbol, "start": self.start.isoformat() if self.start else None,
                   "end": self.end.isoformat() if self.end else None, "source": self.source,
                   "timeframe": self.timeframe, "schema_version": self.schema_version}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def frame_fingerprint(frame: pd.DataFrame) -> str:
    normalized = validate_market_data(frame)
    payload = {"columns": list(normalized.columns), "dtypes": [str(dtype) for dtype in normalized.dtypes],
               "index_dtype": str(normalized.index.dtype)}
    content = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    digest.update(content)
    return digest.hexdigest()


class MarketDataCache:
    """Filesystem cache for validated, tamper-evident provider responses."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_key(key: CacheKey | Callable[[], CacheKey]) -> CacheKey:
        return key() if callable(key) else key

    def path_for(self, key: CacheKey | Callable[[], CacheKey]) -> Path:
        key = self._resolve_key(key)
        safe_symbol = "".join(c if c.isalnum() or c in "-_" else "_" for c in key.symbol)
        return self.root / f"{safe_symbol}-{key.value}.csv"

    def manifest_path_for(self, key: CacheKey | Callable[[], CacheKey]) -> Path:
        return self.path_for(key).with_suffix(".json")

    def get(self, key: CacheKey) -> pd.DataFrame | None:
        path = self.path_for(key)
        manifest_path = self.manifest_path_for(key)
        if not path.exists() or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("cache_key") != key.value:
                return None
            restored = validate_market_data(pd.read_csv(path, index_col=0, parse_dates=True))
            if frame_fingerprint(restored) != manifest.get("frame_fingerprint"):
                return None
            return restored.copy()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def put(self, key: CacheKey, frame: pd.DataFrame) -> Path:
        normalized = validate_market_data(frame)
        path = self.path_for(key)
        manifest_path = self.manifest_path_for(key)
        temp = path.with_suffix(".data.tmp")
        manifest_temp = manifest_path.with_suffix(".manifest.tmp")
        fingerprint = frame_fingerprint(normalized)
        normalized.to_csv(temp)
        manifest = {"cache_key": key.value, "frame_fingerprint": fingerprint, "rows": len(normalized)}
        manifest_temp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        temp.replace(path)
        manifest_temp.replace(manifest_path)
        return path
