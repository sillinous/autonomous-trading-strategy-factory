from __future__ import annotations

import json
from typing import Any

from .certificate_integrity import verify_persisted_certificate
from .paper_admission_record import PaperAdmissionRecord
from .registry import ExperimentRegistry


class PaperAdmissionStore:
    """Durable, immutable persistence boundary for paper admissions."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        self._registry = registry
        self._connection = registry._connection
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create or safely migrate the admission table to composite run/strategy identity."""
        table = self._connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'paper_admissions'"
        ).fetchone()
        if table is None:
            self._create_table()
            return

        unique_run_only = False
        for index in self._connection.execute("PRAGMA index_list('paper_admissions')").fetchall():
            if not index["unique"]:
                continue
            columns = self._connection.execute(
                f"PRAGMA index_info('{index['name']}')"
            ).fetchall()
            names = tuple(column["name"] for column in columns)
            if names == ("run_id",):
                unique_run_only = True
                break
        if not unique_run_only:
            return

        with self._connection:
            self._connection.execute("ALTER TABLE paper_admissions RENAME TO paper_admissions_legacy")
            self._create_table(commit=False)
            self._connection.execute(
                """INSERT INTO paper_admissions(
                    admission_id, strategy_id, run_id, certificate_id,
                    source_stage, target_stage, reason, admission_json, created_at
                ) SELECT admission_id, strategy_id, run_id, certificate_id,
                    source_stage, target_stage, reason, admission_json, created_at
                FROM paper_admissions_legacy"""
            )
            self._connection.execute("DROP TABLE paper_admissions_legacy")

    def _create_table(self, *, commit: bool = True) -> None:
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS paper_admissions (
                admission_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                certificate_id TEXT NOT NULL,
                source_stage TEXT NOT NULL,
                target_stage TEXT NOT NULL,
                reason TEXT NOT NULL,
                admission_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (run_id, strategy_id),
                FOREIGN KEY (run_id) REFERENCES portfolio_runs(run_id)
            )"""
        )
        if commit:
            self._connection.commit()

    @staticmethod
    def _payload(record: PaperAdmissionRecord) -> str:
        return json.dumps(
            {
                "admission_id": record.admission_id,
                "strategy_id": record.strategy_id,
                "run_id": record.run_id,
                "certificate_id": record.certificate_id,
                "source_stage": record.source_stage,
                "target_stage": record.target_stage,
                "reason": record.reason,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def save(self, record: PaperAdmissionRecord) -> PaperAdmissionRecord:
        payload = self._payload(record)
        existing = self._connection.execute(
            "SELECT admission_id, admission_json FROM paper_admissions WHERE run_id = ? AND strategy_id = ?",
            (record.run_id, record.strategy_id),
        ).fetchone()
        if existing is not None:
            if existing["admission_id"] != record.admission_id or existing["admission_json"] != payload:
                raise ValueError("paper admission already exists and is immutable")
            return record
        with self._connection:
            self._connection.execute(
                """INSERT INTO paper_admissions(
                    admission_id, strategy_id, run_id, certificate_id,
                    source_stage, target_stage, reason, admission_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.admission_id,
                    record.strategy_id,
                    record.run_id,
                    record.certificate_id,
                    record.source_stage,
                    record.target_stage,
                    record.reason,
                    payload,
                ),
            )
        return record

    def get(self, run_id: str, strategy_id: str | None = None) -> PaperAdmissionRecord | None:
        if strategy_id is None:
            rows = self._connection.execute(
                "SELECT admission_json FROM paper_admissions WHERE run_id = ? ORDER BY strategy_id",
                (run_id,),
            ).fetchall()
            if not rows:
                return None
            if len(rows) > 1:
                raise ValueError("paper admission strategy_id is required for multi-strategy run")
            row = rows[0]
        else:
            row = self._connection.execute(
                "SELECT admission_json FROM paper_admissions WHERE run_id = ? AND strategy_id = ?",
                (run_id, strategy_id),
            ).fetchone()
            if row is None:
                return None
        data: dict[str, Any] = json.loads(row["admission_json"])
        return PaperAdmissionRecord(**data)

    def list_for_run(self, run_id: str) -> tuple[PaperAdmissionRecord, ...]:
        rows = self._connection.execute(
            "SELECT admission_json FROM paper_admissions WHERE run_id = ? ORDER BY strategy_id",
            (run_id,),
        ).fetchall()
        return tuple(PaperAdmissionRecord(**json.loads(row["admission_json"])) for row in rows)

    def verify(self, run_id: str, strategy_id: str | None = None) -> bool:
        try:
            record = self.get(run_id, strategy_id)
        except ValueError:
            return False
        if record is None:
            return False
        if record.target_stage != "PAPER" or record.source_stage != "PROMOTED":
            return False
        if self._connection.execute(
            "SELECT 1 FROM portfolio_runs WHERE run_id = ? AND halted = 0", (run_id,)
        ).fetchone() is None:
            return False
        certificate = verify_persisted_certificate(self._registry, run_id)
        if not certificate.valid or certificate.certificate_id != record.certificate_id:
            return False
        return self._payload(record) == json.dumps(
            {
                "admission_id": record.admission_id,
                "strategy_id": record.strategy_id,
                "run_id": record.run_id,
                "certificate_id": record.certificate_id,
                "source_stage": record.source_stage,
                "target_stage": record.target_stage,
                "reason": record.reason,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
