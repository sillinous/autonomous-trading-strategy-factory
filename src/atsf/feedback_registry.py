from __future__ import annotations

import json
from typing import Any

from .feedback_provenance import FeedbackProvenance, verify_feedback_provenance
from .registry import ExperimentRegistry


class FeedbackEventStore:
    """Persistent, write-once storage for strategy health-to-research feedback events."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        self._connection = registry._connection
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_feedback_events (
                event_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                event_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    def save(self, event: FeedbackProvenance) -> None:
        if not verify_feedback_provenance(event):
            raise ValueError("feedback provenance is invalid")
        payload = json.dumps(
            {
                "event_id": event.event_id,
                "strategy_id": event.strategy_id,
                "previous_state": event.previous_state,
                "resulting_state": event.resulting_state,
                "report_fingerprint": event.report_fingerprint,
                "research_request_id": event.research_request_id,
                "research_fingerprint": event.research_fingerprint,
                "fingerprint": event.fingerprint,
            },
            sort_keys=True,
            allow_nan=False,
        )
        existing = self._connection.execute(
            "SELECT event_json FROM strategy_feedback_events WHERE event_id = ?",
            (event.event_id,),
        ).fetchone()
        if existing is not None:
            if existing[0] != payload:
                raise ValueError("feedback event already exists and is immutable")
            return
        with self._connection:
            self._connection.execute(
                "INSERT INTO strategy_feedback_events(event_id, strategy_id, event_json) VALUES (?, ?, ?)",
                (event.event_id, event.strategy_id, payload),
            )

    def get(self, event_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT event_json FROM strategy_feedback_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def list_for_strategy(self, strategy_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT event_json FROM strategy_feedback_events WHERE strategy_id = ? ORDER BY event_id",
            (strategy_id,),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]
