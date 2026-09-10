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
                PRIMARY KEY (dataset_id, version)
            )
            """
        )
        self._connection.commit()

    def register(
        self,
        identity: DatasetBundleIdentity,
        *,
        source: str,
    ) -> DatasetRecord:
        if not source.strip():
            raise ValueError("source is required")
        if identity.rows <= 0:
            raise ValueError("dataset must contain rows")
        record = DatasetRecord(
            dataset_id=identity.dataset_id,
            version=identity.version,
            symbols=identity.symbols,
            rows=identity.rows,
            start=identity.start,
            end=identity.end,
            source=source.strip(),
        )
        existing = self.get(identity.dataset_id, identity.version)
        if existing is not None:
            if existing != record:
                raise ValueError("dataset registration is immutable")
            return existing
        self._connection.execute(
            """
            INSERT INTO datasets
                (dataset_id, version, symbols_json, rows, start, end, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.dataset_id,
                record.version,
                json.dumps(record.symbols, separators=(",", ":")),
                record.rows,
                record.start,
                record.end,
                record.source,
            ),
        )
        self._connection.commit()
        return record

    def get(self, dataset_id: str, version: str) -> DatasetRecord | None:
        row = self._connection.execute(
            """
            SELECT dataset_id, version, symbols_json, rows, start, end, source
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
        )

    def require(self, dataset_id: str, version: str) -> DatasetRecord:
        record = self.get(dataset_id, version)
        if record is None:
            raise KeyError(f"unknown dataset: {dataset_id}@{version}")
        return record
