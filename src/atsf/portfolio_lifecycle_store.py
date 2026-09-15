from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import sqlite3
from typing import Any, Mapping


_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PortfolioLifecycleRecord:
    portfolio_id: str
    generation: int
    strategy_ids: tuple[str, ...]
    weights: tuple[tuple[str, float], ...]
    health_status: str
    decision: str
    total_return: float
    volatility: float
    max_risk_fraction: float
    breached_limits: tuple[str, ...]
    replace_strategy_ids: tuple[str, ...]
    execution_authority: bool
    created_at: str
    fingerprint: str
    schema_version: int = _SCHEMA_VERSION


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values are not supported")
        return value
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    raise TypeError(f"unsupported canonical value: {type(value)!r}")


def _fingerprint_payload(record: PortfolioLifecycleRecord) -> dict[str, Any]:
    return _canonical(
        {
            "portfolio_id": record.portfolio_id,
            "generation": record.generation,
            "strategy_ids": record.strategy_ids,
            "weights": record.weights,
            "health_status": record.health_status,
            "decision": record.decision,
            "total_return": record.total_return,
            "volatility": record.volatility,
            "max_risk_fraction": record.max_risk_fraction,
            "breached_limits": record.breached_limits,
            "replace_strategy_ids": record.replace_strategy_ids,
            "execution_authority": record.execution_authority,
            "created_at": record.created_at,
            "schema_version": record.schema_version,
        }
    )


def _compute_fingerprint(record: PortfolioLifecycleRecord) -> str:
    payload = json.dumps(_fingerprint_payload(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_record(record: PortfolioLifecycleRecord) -> None:
    if not record.portfolio_id.strip():
        raise ValueError("portfolio_id must be non-empty")
    if record.generation < 0:
        raise ValueError("generation must be non-negative")
    if record.schema_version != _SCHEMA_VERSION:
        raise ValueError("unsupported portfolio lifecycle schema version")
    if record.execution_authority:
        raise ValueError("portfolio lifecycle records cannot grant execution authority")
    if not record.strategy_ids:
        raise ValueError("strategy_ids must be non-empty")
    if len(set(record.strategy_ids)) != len(record.strategy_ids):
        raise ValueError("strategy_ids must be unique")
    if tuple(sorted(record.strategy_ids)) != record.strategy_ids:
        raise ValueError("strategy_ids must be sorted")
    if tuple(sorted(record.weights)) != record.weights:
        raise ValueError("weights must be sorted")
    if tuple(strategy_id for strategy_id, _ in record.weights) != record.strategy_ids:
        raise ValueError("weights must cover strategy_ids exactly")
    for _, weight in record.weights:
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weights must be finite and non-negative")
    for name, value in (
        ("total_return", record.total_return),
        ("volatility", record.volatility),
        ("max_risk_fraction", record.max_risk_fraction),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not record.fingerprint:
        raise ValueError("fingerprint must be non-empty")
    if record.fingerprint != _compute_fingerprint(record):
        raise ValueError("portfolio lifecycle fingerprint mismatch")
    try:
        datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at must be ISO-8601") from exc


class PortfolioLifecycleStore:
    """Append-only SQLite persistence for auditable portfolio decisions."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        columns = self._connection.execute("PRAGMA table_info(portfolio_lifecycle)").fetchall()
        if not columns:
            self._connection.execute(
                """
                CREATE TABLE portfolio_lifecycle (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    strategy_ids TEXT NOT NULL,
                    weights TEXT NOT NULL,
                    health_status TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    total_return REAL NOT NULL,
                    volatility REAL NOT NULL,
                    max_risk_fraction REAL NOT NULL,
                    breached_limits TEXT NOT NULL,
                    replace_strategy_ids TEXT NOT NULL,
                    execution_authority INTEGER NOT NULL CHECK (execution_authority = 0),
                    created_at TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    schema_version INTEGER NOT NULL CHECK (schema_version = 1)
                )
                """
            )
            self._connection.commit()
            return
        names = {str(row[1]) for row in columns}
        required = {
            "sequence", "portfolio_id", "generation", "strategy_ids", "weights", "health_status",
            "decision", "total_return", "volatility", "max_risk_fraction", "breached_limits",
            "replace_strategy_ids", "execution_authority", "created_at", "fingerprint", "schema_version",
        }
        if names != required:
            raise ValueError("unsupported portfolio lifecycle schema; explicit migration is required")
        versions = self._connection.execute("SELECT DISTINCT schema_version FROM portfolio_lifecycle").fetchall()
        if any(int(row[0]) != _SCHEMA_VERSION for row in versions):
            raise ValueError("unsupported portfolio lifecycle schema version")
        authority = self._connection.execute("SELECT COUNT(*) FROM portfolio_lifecycle WHERE execution_authority != 0").fetchone()[0]
        if authority:
            raise ValueError("portfolio lifecycle store contains execution authority")

    @staticmethod
    def new_record(
        *,
        portfolio_id: str,
        generation: int,
        strategy_ids: tuple[str, ...],
        weights: Mapping[str, float],
        health_status: str,
        decision: str,
        total_return: float,
        volatility: float,
        max_risk_fraction: float,
        breached_limits: tuple[str, ...],
        replace_strategy_ids: tuple[str, ...],
        created_at: str | None = None,
    ) -> PortfolioLifecycleRecord:
        record = PortfolioLifecycleRecord(
            portfolio_id=portfolio_id,
            generation=generation,
            strategy_ids=tuple(sorted(strategy_ids)),
            weights=tuple(sorted((str(k), float(v)) for k, v in weights.items())),
            health_status=health_status,
            decision=decision,
            total_return=float(total_return),
            volatility=float(volatility),
            max_risk_fraction=float(max_risk_fraction),
            breached_limits=tuple(breached_limits),
            replace_strategy_ids=tuple(replace_strategy_ids),
            execution_authority=False,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            fingerprint="",
        )
        record = PortfolioLifecycleRecord(**{**record.__dict__, "fingerprint": _compute_fingerprint(record)})
        _validate_record(record)
        return record

    def append(self, record: PortfolioLifecycleRecord) -> PortfolioLifecycleRecord:
        _validate_record(record)
        latest = self.latest(record.portfolio_id)
        if latest is not None and record.generation <= latest.generation:
            raise ValueError("portfolio lifecycle generation must increase monotonically")
        self._connection.execute(
            """
            INSERT INTO portfolio_lifecycle
            (portfolio_id, generation, strategy_ids, weights, health_status, decision,
             total_return, volatility, max_risk_fraction, breached_limits,
             replace_strategy_ids, execution_authority, created_at, fingerprint, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.portfolio_id,
                record.generation,
                json.dumps(record.strategy_ids, separators=(",", ":")),
                json.dumps(record.weights, separators=(",", ":")),
                record.health_status,
                record.decision,
                record.total_return,
                record.volatility,
                record.max_risk_fraction,
                json.dumps(record.breached_limits, separators=(",", ":")),
                json.dumps(record.replace_strategy_ids, separators=(",", ":")),
                0,
                record.created_at,
                record.fingerprint,
                record.schema_version,
            ),
        )
        self._connection.commit()
        return record

    def latest(self, portfolio_id: str) -> PortfolioLifecycleRecord | None:
        row = self._connection.execute(
            "SELECT * FROM portfolio_lifecycle WHERE portfolio_id = ? ORDER BY generation DESC, sequence DESC LIMIT 1",
            (portfolio_id,),
        ).fetchone()
        return self._decode(row) if row else None

    def verify(self, portfolio_id: str | None = None) -> None:
        query = "SELECT * FROM portfolio_lifecycle"
        args: tuple[str, ...] = ()
        if portfolio_id is not None:
            query += " WHERE portfolio_id = ?"
            args = (portfolio_id,)
        query += " ORDER BY sequence"
        rows = self._connection.execute(query, args).fetchall()
        last_generation: dict[str, int] = {}
        for row in rows:
            record = self._decode(row)
            previous = last_generation.get(record.portfolio_id)
            if previous is not None and record.generation <= previous:
                raise ValueError("portfolio lifecycle generation ordering is invalid")
            last_generation[record.portfolio_id] = record.generation

    def _decode(self, row: sqlite3.Row) -> PortfolioLifecycleRecord:
        record = PortfolioLifecycleRecord(
            portfolio_id=str(row["portfolio_id"]),
            generation=int(row["generation"]),
            strategy_ids=tuple(json.loads(row["strategy_ids"])),
            weights=tuple((str(k), float(v)) for k, v in json.loads(row["weights"])),
            health_status=str(row["health_status"]),
            decision=str(row["decision"]),
            total_return=float(row["total_return"]),
            volatility=float(row["volatility"]),
            max_risk_fraction=float(row["max_risk_fraction"]),
            breached_limits=tuple(json.loads(row["breached_limits"])),
            replace_strategy_ids=tuple(json.loads(row["replace_strategy_ids"])),
            execution_authority=bool(row["execution_authority"]),
            created_at=str(row["created_at"]),
            fingerprint=str(row["fingerprint"]),
            schema_version=int(row["schema_version"]),
        )
        _validate_record(record)
        return record
