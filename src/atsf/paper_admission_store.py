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
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS paper_admissions (
                admission_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                run_id TEXT NOT NULL UNIQUE,
                certificate_id TEXT NOT NULL,
                source_stage TEXT NOT NULL,
                target_stage TEXT NOT NULL,
                reason TEXT NOT NULL,
                admission_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES portfolio_runs(run_id)
            )"""
        )
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
            "SELECT admission_id, admission_json FROM paper_admissions WHERE run_id = ?",
            (record.run_id,),
        ).fetchone()
        if existing is not None:
            if existing["admission_id"] != record.admission_id or existing["admission_json"] != payload:
                raise ValueError("paper admission already exists and is immutable")
            return record
        with self._connection:
            self._connection.execute(
                "INSERT INTO paper_admissions(admission_id, strategy_id, run_id, certificate_id, source_stage, target_stage, reason, admission_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (record.admission_id, record.strategy_id, record.run_id, record.certificate_id, record.source_stage, record.target_stage, record.reason, payload),
            )
        return record

    def get(self, run_id: str) -> PaperAdmissionRecord | None:
        row = self._connection.execute(
            "SELECT admission_json FROM paper_admissions WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        data: dict[str, Any] = json.loads(row["admission_json"])
        return PaperAdmissionRecord(**data)

    def verify(self, run_id: str) -> bool:
        record = self.get(run_id)
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
