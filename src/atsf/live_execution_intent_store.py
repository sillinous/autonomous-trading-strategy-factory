from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite

from .live_execution_intent import LiveExecutionIntent, verify_execution_intent
from .live_risk_gateway import LiveRiskCertificate, validate_live_risk_certificate
from .registry import ExperimentRegistry


@dataclass(frozen=True)
class ExecutionIntentRecord:
    """Durable journal record; it never represents broker submission."""

    intent_id: str
    strategy_id: str
    certificate_fingerprint: str
    intent_fingerprint: str
    status: str
    created_at: float
    updated_at: float
    reason: str = ""


class LiveExecutionIntentStore:
    """SQLite-backed write-once journal with atomic consumption semantics."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        self._connection = registry._connection
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS live_execution_intents (
                intent_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                certificate_fingerprint TEXT NOT NULL,
                intent_fingerprint TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                CHECK (status IN ('CREATED', 'CONSUMED', 'REJECTED', 'CANCELLED'))
            )"""
        )
        self._connection.commit()

    @staticmethod
    def _payload(record: ExecutionIntentRecord) -> str:
        return json.dumps(
            {
                "intent_id": record.intent_id,
                "strategy_id": record.strategy_id,
                "certificate_fingerprint": record.certificate_fingerprint,
                "intent_fingerprint": record.intent_fingerprint,
                "status": record.status,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "reason": record.reason,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def _record(row) -> ExecutionIntentRecord:
        return ExecutionIntentRecord(
            intent_id=row["intent_id"],
            strategy_id=row["strategy_id"],
            certificate_fingerprint=row["certificate_fingerprint"],
            intent_fingerprint=row["intent_fingerprint"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            reason=row["reason"],
        )

    def get(self, intent_id: str) -> ExecutionIntentRecord | None:
        row = self._connection.execute(
            "SELECT * FROM live_execution_intents WHERE intent_id = ?", (intent_id,)
        ).fetchone()
        return None if row is None else self._record(row)

    def record(self, intent: LiveExecutionIntent, *, now: float) -> ExecutionIntentRecord:
        """Persist an intent exactly once; identical retries return the durable record."""
        if not verify_execution_intent(intent):
            raise ValueError("execution intent fingerprint is invalid")
        if not isfinite(now):
            raise ValueError("now must be finite")
        candidate = ExecutionIntentRecord(
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            certificate_fingerprint=intent.certificate_fingerprint,
            intent_fingerprint=intent.fingerprint,
            status="CREATED",
            created_at=intent.created_at,
            updated_at=now,
        )
        existing = self.get(intent.intent_id)
        if existing is not None:
            if self._payload(existing) != self._payload(candidate):
                raise ValueError("execution intent already exists and is immutable")
            return existing
        duplicate = self._connection.execute(
            "SELECT intent_id FROM live_execution_intents WHERE intent_fingerprint = ?",
            (intent.fingerprint,),
        ).fetchone()
        if duplicate is not None:
            return self.get(duplicate["intent_id"])
        with self._connection:
            self._connection.execute(
                """INSERT INTO live_execution_intents(
                    intent_id, strategy_id, certificate_fingerprint, intent_fingerprint,
                    status, created_at, updated_at, reason
                ) VALUES (?, ?, ?, ?, 'CREATED', ?, ?, '')""",
                (
                    candidate.intent_id,
                    candidate.strategy_id,
                    candidate.certificate_fingerprint,
                    candidate.intent_fingerprint,
                    candidate.created_at,
                    candidate.updated_at,
                ),
            )
        return candidate

    def consume(
        self,
        intent: LiveExecutionIntent,
        certificate: LiveRiskCertificate,
        *,
        now: float,
        kill_switch_engaged: bool = False,
    ) -> ExecutionIntentRecord:
        """Atomically mark an admitted intent consumed; no broker call is performed."""
        if not verify_execution_intent(intent):
            raise ValueError("execution intent fingerprint is invalid")
        if certificate.fingerprint != intent.certificate_fingerprint:
            raise ValueError("certificate fingerprint does not match execution intent")
        if not isfinite(now):
            raise ValueError("now must be finite")
        allowed, reasons = validate_live_risk_certificate(
            certificate,
            strategy_id=intent.strategy_id,
            now=now,
            requested_capital_fraction=intent.quantity_fraction,
            kill_switch_engaged=kill_switch_engaged,
        )
        if not allowed:
            raise ValueError("execution intent rejected: " + "; ".join(reasons))
        current = self.get(intent.intent_id)
        if current is None:
            raise ValueError("execution intent has not been journaled")
        if current.intent_fingerprint != intent.fingerprint:
            raise ValueError("journaled execution intent fingerprint mismatch")
        if current.status != "CREATED":
            raise ValueError(f"execution intent is already {current.status.lower()}")
        with self._connection:
            updated = self._connection.execute(
                """UPDATE live_execution_intents
                   SET status = 'CONSUMED', updated_at = ?, reason = 'consumed'
                 WHERE intent_id = ? AND intent_fingerprint = ? AND status = 'CREATED'""",
                (now, intent.intent_id, intent.fingerprint),
            )
            if updated.rowcount != 1:
                raise ValueError("execution intent could not be consumed atomically")
        return self.get(intent.intent_id)
