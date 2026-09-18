from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from .research_checkpoint import ResearchCheckpoint

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResearchCheckpointRecord:
    checkpoint_id: str
    next_generation: int
    cycle_id: str
    state_digest: str
    payload_json: str
    schema_version: int = _SCHEMA_VERSION


class ResearchCheckpointStore:
    """Durable, immutable storage for validated research restart checkpoints."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._create_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                next_generation INTEGER NOT NULL UNIQUE,
                cycle_id TEXT NOT NULL,
                state_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL DEFAULT 1,
                CHECK (schema_version = 1)
            );
            """
        )
        columns = {row[1] for row in self._connection.execute(
            "PRAGMA table_info(research_checkpoints)"
        ).fetchall()}
        required = {
            "checkpoint_id", "next_generation", "cycle_id",
            "state_digest", "payload_json", "schema_version",
        }
        if columns != required:
            raise ValueError(
                "unsupported research checkpoint store schema; explicit migration is required"
            )
        versions = self._connection.execute(
            "SELECT DISTINCT schema_version FROM research_checkpoints"
        ).fetchall()
        if any(row[0] != _SCHEMA_VERSION for row in versions):
            raise ValueError("unsupported research checkpoint store schema version")
        self.verify()
        self._connection.commit()

    @staticmethod
    def _checkpoint_id(checkpoint: ResearchCheckpoint) -> str:
        return f"generation-{checkpoint.next_generation}"

    def save_in_transaction(
        self, checkpoint: ResearchCheckpoint, *, cycle_id: str
    ) -> ResearchCheckpointRecord:
        if not isinstance(cycle_id, str) or not cycle_id.strip():
            raise ValueError("cycle_id is required")
        if checkpoint.next_generation < 1:
            raise ValueError("durable checkpoint must follow a completed generation")
        payload_json = checkpoint.to_json()
        decoded = ResearchCheckpoint.from_json(payload_json)
        if decoded.state_digest != checkpoint.state_digest:
            raise ValueError("checkpoint serialization integrity mismatch")

        checkpoint_id = self._checkpoint_id(checkpoint)
        record = ResearchCheckpointRecord(
            checkpoint_id,
            checkpoint.next_generation,
            cycle_id,
            checkpoint.state_digest,
            payload_json,
        )
        self.verify()
        existing = self._connection.execute(
            "SELECT next_generation, cycle_id, state_digest, payload_json, schema_version "
            "FROM research_checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()
        expected = (
            checkpoint.next_generation, cycle_id, checkpoint.state_digest,
            payload_json, _SCHEMA_VERSION,
        )
        if existing is not None:
            if tuple(existing) != expected:
                raise ValueError("research checkpoint is immutable")
            return record
        prior = self._connection.execute(
            "SELECT next_generation FROM research_checkpoints "
            "ORDER BY next_generation DESC LIMIT 1"
        ).fetchone()
        if prior is not None and checkpoint.next_generation <= prior[0]:
            raise ValueError("research checkpoint generations must advance")
        self._connection.execute(
            "INSERT INTO research_checkpoints("
            "checkpoint_id, next_generation, cycle_id, state_digest, payload_json, schema_version"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                checkpoint_id, checkpoint.next_generation, cycle_id,
                checkpoint.state_digest, payload_json, _SCHEMA_VERSION,
            ),
        )
        return record

    def save(self, checkpoint: ResearchCheckpoint, *, cycle_id: str) -> ResearchCheckpointRecord:
        with self._connection:
            return self.save_in_transaction(checkpoint, cycle_id=cycle_id)

    def get(self, next_generation: int) -> ResearchCheckpointRecord | None:
        row = self._connection.execute(
            "SELECT checkpoint_id, next_generation, cycle_id, state_digest, payload_json, schema_version "
            "FROM research_checkpoints WHERE next_generation = ?",
            (next_generation,),
        ).fetchone()
        return None if row is None else ResearchCheckpointRecord(*row)

    def load(self, next_generation: int) -> ResearchCheckpoint:
        record = self.get(next_generation)
        if record is None:
            raise ValueError("research checkpoint not found")
        try:
            checkpoint = ResearchCheckpoint.from_json(record.payload_json)
        except ValueError as exc:
            raise ValueError(
                "research checkpoint payload integrity verification failed"
            ) from exc
        if (
            checkpoint.next_generation != record.next_generation
            or checkpoint.state_digest != record.state_digest
        ):
            raise ValueError("research checkpoint record integrity verification failed")
        return checkpoint

    def latest(self) -> ResearchCheckpoint | None:
        row = self._connection.execute(
            "SELECT next_generation FROM research_checkpoints "
            "ORDER BY next_generation DESC LIMIT 1"
        ).fetchone()
        return None if row is None else self.load(row[0])

    def verify(self) -> None:
        rows = self._connection.execute(
            "SELECT checkpoint_id, next_generation, cycle_id, state_digest, payload_json, schema_version "
            "FROM research_checkpoints ORDER BY next_generation"
        ).fetchall()
        previous_generation = 0
        seen_ids: set[str] = set()
        for row in rows:
            record = ResearchCheckpointRecord(*row)
            if record.schema_version != _SCHEMA_VERSION:
                raise ValueError("unsupported research checkpoint store schema version")
            if record.checkpoint_id in seen_ids:
                raise ValueError("duplicate research checkpoint ID")
            seen_ids.add(record.checkpoint_id)
            if record.next_generation <= previous_generation:
                raise ValueError("research checkpoint generation ordering is invalid")
            if record.checkpoint_id != f"generation-{record.next_generation}":
                raise ValueError("research checkpoint ID does not match generation")
            if not record.cycle_id.strip():
                raise ValueError("research checkpoint cycle ID is required")
            try:
                checkpoint = ResearchCheckpoint.from_json(record.payload_json)
            except ValueError as exc:
                raise ValueError(
                    "research checkpoint payload integrity verification failed"
                ) from exc
            if checkpoint.next_generation != record.next_generation:
                raise ValueError("research checkpoint generation mismatch")
            if checkpoint.state_digest != record.state_digest:
                raise ValueError("research checkpoint digest mismatch")
            previous_generation = record.next_generation
