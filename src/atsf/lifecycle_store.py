from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from .lifecycle import StrategyLifecycleStage, can_promote_stage


@dataclass(frozen=True)
class PersistedLifecycle:
    strategy_id: str
    stage: StrategyLifecycleStage
    reason: str


@dataclass(frozen=True)
class LifecycleEvent:
    sequence: int
    strategy_id: str
    source_stage: StrategyLifecycleStage | None
    target_stage: StrategyLifecycleStage
    reason: str
    integrity_hash: str


class LifecycleStore:
    """Durable, integrity-checked lifecycle state with append-only history."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS strategy_lifecycle (
                strategy_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                reason TEXT NOT NULL,
                integrity_hash TEXT
            )"""
        )
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS strategy_lifecycle_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                source_stage TEXT,
                target_stage TEXT NOT NULL,
                reason TEXT NOT NULL,
                integrity_hash TEXT NOT NULL,
                UNIQUE(strategy_id, integrity_hash)
            )"""
        )
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(strategy_lifecycle)").fetchall()}
        if "integrity_hash" not in columns:
            self._connection.execute("ALTER TABLE strategy_lifecycle ADD COLUMN integrity_hash TEXT")
        rows = self._connection.execute(
            "SELECT strategy_id, stage, reason FROM strategy_lifecycle WHERE integrity_hash IS NULL"
        ).fetchall()
        for row in rows:
            self._connection.execute(
                "UPDATE strategy_lifecycle SET integrity_hash = ? WHERE strategy_id = ?",
                (self._state_hash(row[0], row[1], row[2]), row[0]),
            )
        self._connection.commit()

    @staticmethod
    def _state_hash(strategy_id: str, stage: str, reason: str) -> str:
        payload = json.dumps(
            {"strategy_id": strategy_id, "stage": stage, "reason": reason},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _event_hash(strategy_id: str, source: str | None, target: str, reason: str) -> str:
        payload = json.dumps(
            {"strategy_id": strategy_id, "source_stage": source, "target_stage": target, "reason": reason},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _append_event(self, strategy_id: str, source: StrategyLifecycleStage | None, target: StrategyLifecycleStage, reason: str) -> None:
        source_value = None if source is None else source.value
        self._connection.execute(
            "INSERT INTO strategy_lifecycle_events(strategy_id, source_stage, target_stage, reason, integrity_hash) VALUES (?, ?, ?, ?, ?)",
            (strategy_id, source_value, target.value, reason, self._event_hash(strategy_id, source_value, target.value, reason)),
        )

    def save(self, strategy_id: str, stage: StrategyLifecycleStage, *, reason: str) -> PersistedLifecycle:
        if not strategy_id.strip():
            raise ValueError("strategy_id is required")
        if not reason.strip():
            raise ValueError("reason is required")
        record = PersistedLifecycle(strategy_id, stage, reason)
        row = self._connection.execute(
            "SELECT stage, reason, integrity_hash FROM strategy_lifecycle WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        if row is not None:
            expected = self._state_hash(strategy_id, row["stage"], row["reason"])
            if row["integrity_hash"] != expected:
                raise ValueError("persisted lifecycle state integrity check failed")
            if row["stage"] != stage.value or row["reason"] != reason:
                raise ValueError("lifecycle state already exists and is immutable")
            return record
        with self._connection:
            self._connection.execute(
                "INSERT INTO strategy_lifecycle(strategy_id, stage, reason, integrity_hash) VALUES (?, ?, ?, ?)",
                (strategy_id, stage.value, reason, self._state_hash(strategy_id, stage.value, reason)),
            )
            self._append_event(strategy_id, None, stage, reason)
        return record

    def transition(
        self,
        strategy_id: str,
        source: StrategyLifecycleStage,
        target: StrategyLifecycleStage,
        *,
        reason: str,
        commit: bool = True,
    ) -> PersistedLifecycle:
        if not strategy_id.strip():
            raise ValueError("strategy_id is required")
        if not reason.strip():
            raise ValueError("reason is required")
        if not can_promote_stage(source, target):
            raise ValueError(f"invalid lifecycle transition: {source.value} -> {target.value}")
        row = self._connection.execute(
            "SELECT stage, reason, integrity_hash FROM strategy_lifecycle WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        if row is None:
            raise ValueError("persisted lifecycle state is missing")
        if row["integrity_hash"] != self._state_hash(strategy_id, row["stage"], row["reason"]):
            raise ValueError("persisted lifecycle state integrity check failed")
        if row["stage"] != source.value:
            raise ValueError("persisted lifecycle source stage does not match admission source")
        self._connection.execute(
            "UPDATE strategy_lifecycle SET stage = ?, reason = ?, integrity_hash = ? WHERE strategy_id = ?",
            (target.value, reason, self._state_hash(strategy_id, target.value, reason), strategy_id),
        )
        self._append_event(strategy_id, source, target, reason)
        if commit:
            self._connection.commit()
        return PersistedLifecycle(strategy_id, target, reason)

    def history(self, strategy_id: str) -> tuple[LifecycleEvent, ...]:
        rows = self._connection.execute(
            "SELECT sequence, strategy_id, source_stage, target_stage, reason, integrity_hash FROM strategy_lifecycle_events WHERE strategy_id = ? ORDER BY sequence",
            (strategy_id,),
        ).fetchall()
        events: list[LifecycleEvent] = []
        for row in rows:
            expected = self._event_hash(row["strategy_id"], row["source_stage"], row["target_stage"], row["reason"])
            if row["integrity_hash"] != expected:
                raise ValueError("persisted lifecycle event integrity check failed")
            try:
                source = None if row["source_stage"] is None else StrategyLifecycleStage(row["source_stage"])
                target = StrategyLifecycleStage(row["target_stage"])
            except ValueError as exc:
                raise ValueError("persisted lifecycle event stage is invalid") from exc
            events.append(LifecycleEvent(row["sequence"], row["strategy_id"], source, target, row["reason"], row["integrity_hash"]))
        return tuple(events)

    def get(self, strategy_id: str) -> PersistedLifecycle | None:
        row = self._connection.execute(
            "SELECT strategy_id, stage, reason, integrity_hash FROM strategy_lifecycle WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        if row is None:
            return None
        if row["integrity_hash"] != self._state_hash(row["strategy_id"], row["stage"], row["reason"]):
            raise ValueError("persisted lifecycle state integrity check failed")
        try:
            stage = StrategyLifecycleStage(row["stage"])
        except ValueError as exc:
            raise ValueError("persisted lifecycle stage is invalid") from exc
        return PersistedLifecycle(row["strategy_id"], stage, row["reason"])

    def restore_stage(self, strategy_id: str, default: StrategyLifecycleStage) -> StrategyLifecycleStage:
        record = self.get(strategy_id)
        return default if record is None else record.stage
