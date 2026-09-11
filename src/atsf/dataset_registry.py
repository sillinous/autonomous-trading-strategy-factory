from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from .dataset_bundle import DatasetBundleIdentity


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    version: str
    symbols: tuple[str, ...]
    rows: int
    start: str
    end: str
    source: str
    timeframe: str
    schema_version: str


class DatasetRegistry:
    """Immutable registry for canonical market-data bundle identities."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT NOT NULL,
                version TEXT NOT NULL,
                symbols_json TEXT NOT NULL,
                rows INTEGER NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                source TEXT NOT NULL,
                timeframe TEXT NOT NULL DEFAULT '1d',
                schema_version TEXT NOT NULL DEFAULT 'ohlcv.v1',
                PRIMARY KEY (dataset_id, version)
            )
            """
        )
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(datasets)")}
        if "timeframe" not in columns:
            self._connection.execute("ALTER TABLE datasets ADD COLUMN timeframe TEXT NOT NULL DEFAULT '1d'")
        if "schema_version" not in columns:
            self._connection.execute(
                "ALTER TABLE datasets ADD COLUMN schema_version TEXT NOT NULL DEFAULT 'ohlcv.v1'"
            )
        self._connection.commit()

    def register(
        self,
        identity: DatasetBundleIdentity,
        *,
        source: str | None = None,
    ) -> DatasetRecord:
        """Register identity metadata; ``source`` is an assertion, never an override."""
        if identity.rows <= 0:
            raise ValueError("dataset must contain rows")
        if source is not None and source.strip() != identity.source:
            raise ValueError("dataset source does not match its identity")
        normalized_source = identity.source.strip()
        if not normalized_source:
            raise ValueError("dataset identity source is required")
        record = DatasetRecord(
            dataset_id=identity.dataset_id,
            version=identity.version,
            symbols=identity.symbols,
            rows=identity.rows,
            start=identity.start,
            end=identity.end,
            source=normalized_source,
            timeframe=identity.timeframe,
            schema_version=identity.schema_version,
        )
        existing = self.get(identity.dataset_id, identity.version)
        if existing is not None:
            if existing != record:
                raise ValueError("dataset registration is immutable")
            return existing
        self._connection.execute(
            """
            INSERT INTO datasets
                (dataset_id, version, symbols_json, rows, start, end, source, timeframe, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.dataset_id,
                record.version,
                json.dumps(record.symbols, separators=(",", ":")),
                record.rows,
                record.start,
                record.end,
                record.source,
                record.timeframe,
                record.schema_version,
            ),
        )
        self._connection.commit()
        return record

    def get(self, dataset_id: str, version: str) -> DatasetRecord | None:
        row = self._connection.execute(
            """
            SELECT dataset_id, version, symbols_json, rows, start, end, source, timeframe, schema_version
            FROM datasets
            WHERE dataset_id = ? AND version = ?
            """,
            (dataset_id, version),
        ).fetchone()
        if row is None:
            return None
        return DatasetRecord(
            dataset_id=row[0],
            version=row[1],
            symbols=tuple(json.loads(row[2])),
            rows=row[3],
            start=row[4],
            end=row[5],
            source=row[6],
            timeframe=row[7],
            schema_version=row[8],
        )

    def require(self, dataset_id: str, version: str) -> DatasetRecord:
        record = self.get(dataset_id, version)
        if record is None:
            raise KeyError(f"unknown dataset: {dataset_id}@{version}")
        return record
