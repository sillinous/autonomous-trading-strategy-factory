from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchCycleRecord:
    cycle_id: str
    generation: int
    plan_json: str
    feedback_json: str
    admissions_json: str


class ResearchCycleRegistry:
    """Durable, append-only persistence for closed-loop research-cycle state."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._create_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the shared connection for coordinated persistence operations."""
        return self._connection

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_cycles (
                cycle_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                plan_json TEXT NOT NULL,
                feedback_json TEXT NOT NULL,
                admissions_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_research_cycles_generation
                ON research_cycles(generation, cycle_id);
            """
        )
        self._connection.commit()

    @staticmethod
    def _payload(value: Any) -> str:
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))

    def save_cycle(
        self,
        cycle_id: str,
        generation: int,
        *,
        plan: Any,
        feedback: Any,
        admissions: Any,
    ) -> ResearchCycleRecord:
        if not isinstance(cycle_id, str) or not cycle_id.strip():
            raise ValueError("cycle_id is required")
        if not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        plan_json = self._payload(plan)
        feedback_json = self._payload(feedback)
        admissions_json = self._payload(admissions)
        with self._connection:
            existing = self._connection.execute(
                "SELECT generation, plan_json, feedback_json, admissions_json "
                "FROM research_cycles WHERE cycle_id = ?",
                (cycle_id,),
            ).fetchone()
            if existing is not None:
                if tuple(existing) != (generation, plan_json, feedback_json, admissions_json):
                    raise ValueError("research cycle is immutable")
                return ResearchCycleRecord(cycle_id, *existing)
            self._connection.execute(
                "INSERT INTO research_cycles(cycle_id, generation, plan_json, feedback_json, admissions_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (cycle_id, generation, plan_json, feedback_json, admissions_json),
            )
        return ResearchCycleRecord(cycle_id, generation, plan_json, feedback_json, admissions_json)

    def get_cycle(self, cycle_id: str) -> ResearchCycleRecord | None:
        row = self._connection.execute(
            "SELECT cycle_id, generation, plan_json, feedback_json, admissions_json "
            "FROM research_cycles WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        return None if row is None else ResearchCycleRecord(*row)

    def list_cycles(self) -> list[ResearchCycleRecord]:
        rows = self._connection.execute(
            "SELECT cycle_id, generation, plan_json, feedback_json, admissions_json "
            "FROM research_cycles ORDER BY generation, cycle_id"
        ).fetchall()
        return [ResearchCycleRecord(*row) for row in rows]
