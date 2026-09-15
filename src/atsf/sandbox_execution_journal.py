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
    previous_fingerprint: str = ""


_ALLOWED_TRANSITIONS: dict[ExecutionState, frozenset[ExecutionState]] = {
    ExecutionState.CREATED: frozenset(
        {ExecutionState.VALIDATED, ExecutionState.REJECTED, ExecutionState.CANCELLED, ExecutionState.EXPIRED}
    ),
    ExecutionState.VALIDATED: frozenset(
        {ExecutionState.ADMITTED, ExecutionState.REJECTED, ExecutionState.CANCELLED, ExecutionState.EXPIRED}
    ),
    ExecutionState.ADMITTED: frozenset(
        {ExecutionState.CONSUMED, ExecutionState.REJECTED, ExecutionState.CANCELLED, ExecutionState.EXPIRED}
    ),
    ExecutionState.CONSUMED: frozenset({ExecutionState.FILLED, ExecutionState.REJECTED, ExecutionState.EXPIRED}),
    ExecutionState.FILLED: frozenset(),
    ExecutionState.REJECTED: frozenset(),
    ExecutionState.CANCELLED: frozenset(),
    ExecutionState.EXPIRED: frozenset(),
}


class SandboxExecutionJournal:
    """Append-only SQLite event journal with chained, tamper-evident fingerprints."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        self._connection = registry._connection
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(sandbox_execution_events)").fetchall()
        }
        if not columns:
            self._connection.execute(
                """CREATE TABLE sandbox_execution_events (
                    intent_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_fingerprint TEXT NOT NULL UNIQUE,
                    detail TEXT NOT NULL DEFAULT '',
                    previous_fingerprint TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (intent_id, sequence),
                    CHECK (state IN ('CREATED', 'VALIDATED', 'ADMITTED', 'CONSUMED', 'FILLED', 'REJECTED', 'CANCELLED', 'EXPIRED'))
                )"""
            )
        elif "previous_fingerprint" not in columns:
            self._connection.execute(
                "ALTER TABLE sandbox_execution_events ADD COLUMN previous_fingerprint TEXT NOT NULL DEFAULT ''"
            )
        self._connection.commit()

    @staticmethod
    def _fingerprint(
        intent_id: str,
        sequence: int,
        state: ExecutionState,
        timestamp: float,
        detail: str,
        previous_fingerprint: str,
    ) -> str:
        payload = {
            "detail": detail,
            "intent_id": intent_id,
            "previous_fingerprint": previous_fingerprint,
            "sequence": sequence,
            "state": state.value,
            "timestamp": timestamp,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(encoded.encode("utf-8")).hexdigest()

    def append_in_transaction(
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
        if previous is None:
            if state is not ExecutionState.CREATED:
                raise ValueError("first execution event must be CREATED")
            sequence = 1
            previous_fingerprint = ""
        else:
            if state not in _ALLOWED_TRANSITIONS[previous.state]:
                raise ValueError(
                    f"invalid execution transition: {previous.state.value} -> {state.value}"
                )
            sequence = previous.sequence + 1
            previous_fingerprint = previous.event_fingerprint
        event = ExecutionEvent(
            intent_id=intent_id,
            sequence=sequence,
            state=state,
            timestamp=timestamp,
            event_fingerprint=self._fingerprint(
                intent_id, sequence, state, timestamp, detail, previous_fingerprint
            ),
            detail=detail,
            previous_fingerprint=previous_fingerprint,
        )
        self._connection.execute(
            """INSERT INTO sandbox_execution_events(
                intent_id, sequence, state, timestamp, event_fingerprint, detail, previous_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                intent_id, sequence, state.value, timestamp, event.event_fingerprint,
                detail, previous_fingerprint,
            ),
        )
        return event

    def append(self, intent_id: str, state: ExecutionState, *, timestamp: float, detail: str = "") -> ExecutionEvent:
        with self._connection:
            return self.append_in_transaction(intent_id, state, timestamp=timestamp, detail=detail)

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
                previous_fingerprint=row["previous_fingerprint"],
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
            intent_id=row["intent_id"], sequence=row["sequence"], state=ExecutionState(row["state"]),
            timestamp=row["timestamp"], event_fingerprint=row["event_fingerprint"],
            detail=row["detail"], previous_fingerprint=row["previous_fingerprint"],
        )

    def verify(self, intent_id: str) -> bool:
        previous_sequence = 0
        previous_state: ExecutionState | None = None
        previous_fingerprint = ""
        for event in self.events(intent_id):
            if event.sequence != previous_sequence + 1:
                return False
            if previous_state is None:
                if event.state is not ExecutionState.CREATED:
                    return False
            elif event.state not in _ALLOWED_TRANSITIONS[previous_state]:
                return False
            if event.previous_fingerprint != previous_fingerprint:
                return False
            if event.event_fingerprint != self._fingerprint(
                event.intent_id, event.sequence, event.state, event.timestamp,
                event.detail, event.previous_fingerprint,
            ):
                return False
            previous_sequence = event.sequence
            previous_state = event.state
            previous_fingerprint = event.event_fingerprint
        return True
