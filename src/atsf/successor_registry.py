from __future__ import annotations

import json

from .generator import StrategyCandidate
from .registry import ExperimentRegistry


class SuccessorCandidateStore:
    """Immutable persistence for typed successor hypotheses produced by research requests."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        self._connection = registry._connection
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS successor_candidates (
                candidate_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                parent_strategy_id TEXT,
                mutation TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                candidate_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    @staticmethod
    def _payload(candidate: StrategyCandidate) -> dict:
        return {
            "candidate_id": candidate.candidate_id,
            "request_id": candidate.request_id,
            "parent_strategy_id": candidate.parent_strategy_id,
            "mutation": candidate.mutation,
            "strategy": candidate.strategy.model_dump(mode="json"),
        }

    def save(self, candidate: StrategyCandidate, strategy_id: str) -> StrategyCandidate:
        if not candidate.candidate_id or not candidate.request_id:
            raise ValueError("candidate_id and request_id are required")
        if not strategy_id:
            raise ValueError("strategy_id cannot be empty")
        serialized = json.dumps(self._payload(candidate), sort_keys=True, separators=(",", ":"))
        row = self._connection.execute(
            "SELECT request_id, parent_strategy_id, mutation, strategy_id, candidate_json FROM successor_candidates WHERE candidate_id = ?",
            (candidate.candidate_id,),
        ).fetchone()
        if row is not None:
            existing = tuple(row)
            expected = (candidate.request_id, candidate.parent_strategy_id, candidate.mutation, strategy_id, serialized)
            if existing != expected:
                raise ValueError(f"successor candidate is immutable: {candidate.candidate_id}")
            return candidate
        self._connection.execute(
            "INSERT INTO successor_candidates(candidate_id, request_id, parent_strategy_id, mutation, strategy_id, candidate_json) VALUES (?, ?, ?, ?, ?, ?)",
            (candidate.candidate_id, candidate.request_id, candidate.parent_strategy_id, candidate.mutation, strategy_id, serialized),
        )
        self._connection.commit()
        return candidate

    def get(self, candidate_id: str) -> dict | None:
        row = self._connection.execute(
            "SELECT * FROM successor_candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        return None if row is None else dict(row)

    def list_for_request(self, request_id: str) -> tuple[dict, ...]:
        if not request_id:
            raise ValueError("request_id cannot be empty")
        rows = self._connection.execute(
            "SELECT * FROM successor_candidates WHERE request_id = ? ORDER BY candidate_id",
            (request_id,),
        ).fetchall()
        return tuple(dict(row) for row in rows)
