from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite

from .registry import ExperimentRegistry
from .sandbox_execution import ExecutionState


@dataclass(frozen=True)
class ExecutionEvent:
    intent_id: str
    sequence: int
    state: ExecutionState
    timestamp: float
    event_fingerprint: str
    detail: str = ""


class SandboxExecutionJournal:
    """Append-only SQLite event journal for deterministic sandbox recovery."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        self._connection = registry._connection
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS sandbox_execution_events (
                intent_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                state TEXT NOT NULL,
                timestamp REAL NOT NULL,
                event_fingerprint TEXT NOT NULL UNIQUE,
                detail TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (intent_id, sequence),
                CHECK (state IN ('CREATED', 'VALIDATED', 'ADMITTED', 'CONSUMED', 'FILLED', 'REJECTED', 'CANCELLED', 'EXPIRED'))
            )"""
        )
        self._connection.commit()

    @staticmethod
    def _fingerprint(intent_id: str, sequence: int, state: ExecutionState, timestamp: float, detail: str) -> str:
        payload = {
            "detail": detail,
            "intent_id": intent_id,
            "sequence": sequence,
            "state": state.value,
            "timestamp": timestamp,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(encoded.encode("utf-8")).hexdigest()

    def append(
        self,
        intent_id: str,
        state: ExecutionState,
        *,
        timestamp: float,
        detail: str = "",
    ) -> ExecutionEvent:
        if not intent_id.strip():
            raise ValueError("intent_id is required")
        if not isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        previous = self.latest(intent_id)
        sequence = 1 if previous is None else previous.sequence + 1
        event = ExecutionEvent(
            intent_id=intent_id,
            sequence=sequence,
            state=state,
            timestamp=timestamp,
            event_fingerprint=self._fingerprint(intent_id, sequence, state, timestamp, detail),
            detail=detail,
        )
        with self._connection:
            self._connection.execute(
                """INSERT INTO sandbox_execution_events(
                    intent_id, sequence, state, timestamp, event_fingerprint, detail
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (intent_id, sequence, state.value, timestamp, event.event_fingerprint, detail),
            )
        return event

    def events(self, intent_id: str) -> tuple[ExecutionEvent, ...]:
        rows = self._connection.execute(
            "SELECT * FROM sandbox_execution_events WHERE intent_id = ? ORDER BY sequence",
            (intent_id,),
        ).fetchall()
        return tuple(
            ExecutionEvent(
                intent_id=row["intent_id"],
                sequence=row["sequence"],
                state=ExecutionState(row["state"]),
                timestamp=row["timestamp"],
                event_fingerprint=row["event_fingerprint"],
                detail=row["detail"],
            )
            for row in rows
        )

    def latest(self, intent_id: str) -> ExecutionEvent | None:
        row = self._connection.execute(
            "SELECT * FROM sandbox_execution_events WHERE intent_id = ? ORDER BY sequence DESC LIMIT 1",
            (intent_id,),
        ).fetchone()
        if row is None:
            return None
        return ExecutionEvent(
            intent_id=row["intent_id"],
            sequence=row["sequence"],
            state=ExecutionState(row["state"]),
            timestamp=row["timestamp"],
            event_fingerprint=row["event_fingerprint"],
            detail=row["detail"],
        )

    def verify(self, intent_id: str) -> bool:
        previous_sequence = 0
        for event in self.events(intent_id):
            if event.sequence != previous_sequence + 1:
                return False
            if event.event_fingerprint != self._fingerprint(
                event.intent_id, event.sequence, event.state, event.timestamp, event.detail
            ):
                return False
            previous_sequence = event.sequence
        return True
