from __future__ import annotations

import json

from .research_queue import ResearchReason, ResearchRequest
from .registry import ExperimentRegistry


class ResearchRequestStore:
    """Immutable persistence for autonomous replacement-research requests."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        self._connection = registry._connection
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS research_requests (
                request_id TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    @staticmethod
    def _payload(request: ResearchRequest) -> dict:
        return {
            "request_id": request.request_id,
            "source_strategy_id": request.source_strategy_id,
            "reason": request.reason.value,
            "priority": request.priority,
            "constraints": list(request.constraints),
        }

    def save(self, request: ResearchRequest) -> ResearchRequest:
        if not request.request_id:
            raise ValueError("request_id cannot be empty")
        payload = self._payload(request)
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        row = self._connection.execute(
            "SELECT request_json FROM research_requests WHERE request_id = ?",
            (request.request_id,),
        ).fetchone()
        if row is not None:
            if row[0] != serialized:
                raise ValueError(f"research request is immutable: {request.request_id}")
            return request
        self._connection.execute(
            "INSERT INTO research_requests(request_id, request_json) VALUES (?, ?)",
            (request.request_id, serialized),
        )
        self._connection.commit()
        return request

    def get(self, request_id: str) -> ResearchRequest | None:
        if not request_id:
            raise ValueError("request_id cannot be empty")
        row = self._connection.execute(
            "SELECT request_json FROM research_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return ResearchRequest(
            request_id=payload["request_id"],
            source_strategy_id=payload.get("source_strategy_id"),
            reason=ResearchReason(payload["reason"]),
            priority=int(payload["priority"]),
            constraints=tuple(payload.get("constraints", [])),
        )

    def list_for_strategy(self, strategy_id: str) -> tuple[ResearchRequest, ...]:
        if not strategy_id:
            raise ValueError("strategy_id cannot be empty")
        rows = self._connection.execute(
            "SELECT request_id FROM research_requests ORDER BY request_id"
        ).fetchall()
        requests = []
        for (request_id,) in rows:
            request = self.get(request_id)
            if request is not None and request.source_strategy_id == strategy_id:
                requests.append(request)
        return tuple(requests)
