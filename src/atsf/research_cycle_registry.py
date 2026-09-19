from __future__ import annotations

import hashlib
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
    portfolio_feedback_json: str = "null"


class ResearchCycleRegistry:
    """Durable, append-only persistence for closed-loop research-cycle state."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._create_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @staticmethod
    def _payload(value: Any) -> str:
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))

    @staticmethod
    def _digest(record: ResearchCycleRecord) -> str:
        payload = "|".join((record.cycle_id, str(record.generation), record.plan_json, record.feedback_json, record.admissions_json, record.portfolio_feedback_json))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _create_schema(self) -> None:
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS research_cycles (
                cycle_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                plan_json TEXT NOT NULL,
                feedback_json TEXT NOT NULL,
                admissions_json TEXT NOT NULL,
                portfolio_feedback_json TEXT NOT NULL DEFAULT 'null'
            );
            CREATE INDEX IF NOT EXISTS idx_research_cycles_generation ON research_cycles(generation, cycle_id);
            CREATE TABLE IF NOT EXISTS research_cycle_audit (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id TEXT NOT NULL UNIQUE,
                generation INTEGER NOT NULL,
                payload_digest TEXT NOT NULL,
                previous_digest TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(cycle_id) REFERENCES research_cycles(cycle_id)
            );
        """)
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(research_cycles)").fetchall()}
        if "portfolio_feedback_json" not in columns:
            self._connection.execute("ALTER TABLE research_cycles ADD COLUMN portfolio_feedback_json TEXT NOT NULL DEFAULT 'null'")
        self._backfill_audit()
        self.verify()
        self._connection.commit()

    def _backfill_audit(self) -> None:
        """Create audit history only for an entirely uninitialized audit table.

        A partially populated audit table is treated as corruption rather than
        silently repaired, preserving evidence of missing historical coverage.
        """
        cycle_count = self._connection.execute("SELECT COUNT(*) FROM research_cycles").fetchone()[0]
        audit_count = self._connection.execute("SELECT COUNT(*) FROM research_cycle_audit").fetchone()[0]
        if audit_count == cycle_count:
            return
        if audit_count != 0:
            raise ValueError("research cycle audit coverage is incomplete")
        if cycle_count == 0:
            return

        rows = self._connection.execute(
            "SELECT cycle_id, generation, plan_json, feedback_json, admissions_json, portfolio_feedback_json FROM research_cycles ORDER BY generation, cycle_id"
        ).fetchall()
        previous = ""
        for row in rows:
            record = ResearchCycleRecord(*row)
            digest = self._digest(record)
            self._connection.execute(
                "INSERT INTO research_cycle_audit(cycle_id, generation, payload_digest, previous_digest) VALUES (?, ?, ?, ?)",
                (record.cycle_id, record.generation, digest, previous),
            )
            previous = digest

    def save_cycle_in_transaction(
        self,
        cycle_id: str,
        generation: int,
        *,
        plan: Any,
        feedback: Any,
        admissions: Any,
        portfolio_feedback: Any = None,
    ) -> ResearchCycleRecord:
        """Persist one cycle without committing the caller's transaction."""
        if not isinstance(cycle_id, str) or not cycle_id.strip():
            raise ValueError("cycle_id is required")
        if not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        expected_cycle_id = f"generation-{generation}"
        if cycle_id != expected_cycle_id:
            raise ValueError("cycle_id does not match generation")
        plan_json = self._payload(plan)
        feedback_json = self._payload(feedback)
        admissions_json = self._payload(admissions)
        portfolio_feedback_json = self._payload(portfolio_feedback)
        record = ResearchCycleRecord(cycle_id, generation, plan_json, feedback_json, admissions_json, portfolio_feedback_json)
        self.verify()
        existing = self._connection.execute(
            "SELECT generation, plan_json, feedback_json, admissions_json, portfolio_feedback_json FROM research_cycles WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        expected = (generation, plan_json, feedback_json, admissions_json, portfolio_feedback_json)
        if existing is not None:
            if tuple(existing) != expected:
                raise ValueError("research cycle is immutable")
            return record
        previous_row = self._connection.execute(
            "SELECT generation, payload_digest FROM research_cycle_audit ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if previous_row is not None and generation != previous_row[0] + 1:
            raise ValueError("research cycle generation sequence is invalid")
        self._connection.execute(
            "INSERT INTO research_cycles(cycle_id, generation, plan_json, feedback_json, admissions_json, portfolio_feedback_json) VALUES (?, ?, ?, ?, ?, ?)",
            (cycle_id, generation, plan_json, feedback_json, admissions_json, portfolio_feedback_json),
        )
        self._connection.execute(
            "INSERT INTO research_cycle_audit(cycle_id, generation, payload_digest, previous_digest) VALUES (?, ?, ?, ?)",
            (cycle_id, generation, self._digest(record), "" if previous_row is None else previous_row[1]),
        )
        return record

    def save_cycle(self, cycle_id: str, generation: int, *, plan: Any, feedback: Any, admissions: Any, portfolio_feedback: Any = None) -> ResearchCycleRecord:
        with self._connection:
            return self.save_cycle_in_transaction(
                cycle_id,
                generation,
                plan=plan,
                feedback=feedback,
                admissions=admissions,
                portfolio_feedback=portfolio_feedback,
            )

    def verify(self) -> None:
        """Verify cycle payloads and the tamper-evident audit chain."""
        rows = self._connection.execute(
            "SELECT sequence, cycle_id, generation, payload_digest, previous_digest FROM research_cycle_audit ORDER BY sequence"
        ).fetchall()
        previous = ""
        previous_generation = -1
        seen: set[str] = set()
        for _, cycle_id, generation, digest, previous_digest in rows:
            if cycle_id in seen:
                raise ValueError("duplicate research cycle audit entry")
            seen.add(cycle_id)
            if cycle_id != f"generation-{generation}":
                raise ValueError("research cycle ID does not match generation")
            if generation != previous_generation + 1:
                raise ValueError("research cycle generation sequence is invalid")
            record = self.get_cycle(cycle_id)
            if record is None or record.generation != generation:
                raise ValueError("research cycle audit record mismatch")
            expected = self._digest(record)
            if digest != expected or previous_digest != previous:
                raise ValueError("research cycle audit integrity verification failed")
            previous = digest
            previous_generation = generation
        count = self._connection.execute("SELECT COUNT(*) FROM research_cycles").fetchone()[0]
        if len(rows) != count:
            raise ValueError("research cycle audit coverage is incomplete")

    def get_cycle(self, cycle_id: str) -> ResearchCycleRecord | None:
        row = self._connection.execute("SELECT cycle_id, generation, plan_json, feedback_json, admissions_json, portfolio_feedback_json FROM research_cycles WHERE cycle_id = ?", (cycle_id,)).fetchone()
        return None if row is None else ResearchCycleRecord(*row)

    def list_cycles(self) -> list[ResearchCycleRecord]:
        rows = self._connection.execute("SELECT cycle_id, generation, plan_json, feedback_json, admissions_json, portfolio_feedback_json FROM research_cycles ORDER BY generation, cycle_id").fetchall()
        return [ResearchCycleRecord(*row) for row in rows]
