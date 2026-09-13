from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .lifecycle import StrategyLifecycleStage


@dataclass(frozen=True)
class PersistedLifecycle:
    strategy_id: str
    stage: StrategyLifecycleStage
    reason: str


class LifecycleStore:
    """Durable fail-closed store for the authoritative promotion lifecycle stage."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS strategy_lifecycle (
                strategy_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                reason TEXT NOT NULL
            )"""
        )
        self._connection.commit()

    def save(self, strategy_id: str, stage: StrategyLifecycleStage, *, reason: str) -> PersistedLifecycle:
        if not strategy_id.strip():
            raise ValueError("strategy_id is required")
        if not reason.strip():
            raise ValueError("reason is required")
        record = PersistedLifecycle(strategy_id, stage, reason)
        row = self._connection.execute(
            "SELECT stage, reason FROM strategy_lifecycle WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        if row is not None:
            if row["stage"] != stage.value or row["reason"] != reason:
                raise ValueError("lifecycle state already exists and is immutable")
            return record
        with self._connection:
            self._connection.execute(
                "INSERT INTO strategy_lifecycle(strategy_id, stage, reason) VALUES (?, ?, ?)",
                (strategy_id, stage.value, reason),
            )
        return record

    def get(self, strategy_id: str) -> PersistedLifecycle | None:
        row = self._connection.execute(
            "SELECT strategy_id, stage, reason FROM strategy_lifecycle WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            stage = StrategyLifecycleStage(row["stage"])
        except ValueError as exc:
            raise ValueError("persisted lifecycle stage is invalid") from exc
        return PersistedLifecycle(row["strategy_id"], stage, row["reason"])

    def restore_stage(self, strategy_id: str, default: StrategyLifecycleStage) -> StrategyLifecycleStage:
        record = self.get(strategy_id)
        return default if record is None else record.stage
