from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .experiment import ExperimentResult, ExperimentSpec
from .lineage import LineageRecord
from .population import strategy_id
from .strategy import StrategySpec


class ExperimentRegistry:
    """Small SQLite persistence boundary for reproducible research metadata."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._connection = sqlite3.connect(str(path))
        self._connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                seed INTEGER NOT NULL,
                status TEXT NOT NULL,
                score REAL,
                reason TEXT,
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS lineage (
                strategy_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                parent_ids_json TEXT NOT NULL,
                operator TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS evaluation_evidence (
                experiment_id TEXT PRIMARY KEY,
                evidence_json TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_dataset
                ON experiments(dataset_id, dataset_version);
            CREATE INDEX IF NOT EXISTS idx_experiments_strategy
                ON experiments(strategy_id);
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def save_strategy(self, strategy: StrategySpec) -> str:
        identifier = strategy_id(strategy)
        payload = json.dumps(strategy.model_dump(mode="json"), sort_keys=True, allow_nan=False)
        self._connection.execute(
            "INSERT OR REPLACE INTO strategies(strategy_id, definition_json) VALUES (?, ?)",
            (identifier, payload),
        )
        self._connection.commit()
        return identifier

    def get_strategy(self, identifier: str) -> StrategySpec | None:
        row = self._connection.execute(
            "SELECT definition_json FROM strategies WHERE strategy_id = ?", (identifier,)
        ).fetchone()
        if row is None:
            return None
        return StrategySpec.model_validate(json.loads(row["definition_json"]))

    def save_experiment(self, spec: ExperimentSpec, result: ExperimentResult) -> None:
        identifier = self.save_strategy(spec.strategy)
        self._connection.execute(
            """INSERT OR REPLACE INTO experiments
            (experiment_id, strategy_id, dataset_id, dataset_version, seed, status, score, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.experiment_id,
                identifier,
                spec.dataset_id,
                spec.dataset_version,
                spec.seed,
                result.status,
                result.score,
                result.reason,
            ),
        )
        self._connection.commit()

    def save_evaluation_evidence(self, experiment_id: str, evidence: dict[str, Any]) -> None:
        """Persist JSON-safe evaluation evidence without coupling the registry to result classes."""
        payload = json.dumps(evidence, sort_keys=True, allow_nan=False)
        self._connection.execute(
            """INSERT OR REPLACE INTO evaluation_evidence(experiment_id, evidence_json)
            VALUES (?, ?)""",
            (experiment_id, payload),
        )
        self._connection.commit()

    def get_evaluation_evidence(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT evidence_json FROM evaluation_evidence WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["evidence_json"])

    def get_experiment(self, identifier: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (identifier,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_experiments(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        if dataset_id is None:
            rows = self._connection.execute(
                "SELECT * FROM experiments ORDER BY experiment_id"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM experiments WHERE dataset_id = ? ORDER BY experiment_id",
                (dataset_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def rank_experiments(
        self,
        dataset_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return eligible experiments ordered by score, then stable experiment ID."""
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        query = (
            "SELECT * FROM experiments WHERE status = 'eligible' "
            + ("AND dataset_id = ? " if dataset_id is not None else "")
            + "ORDER BY score DESC, experiment_id ASC"
        )
        params: tuple[Any, ...] = (dataset_id,) if dataset_id is not None else ()
        if limit is not None:
            query += " LIMIT ?"
            params += (limit,)
        rows = self._connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def save_lineage(self, record: LineageRecord) -> None:
        self._connection.execute(
            """INSERT OR REPLACE INTO lineage
            (strategy_id, generation, parent_ids_json, operator, parameters_json)
            VALUES (?, ?, ?, ?, ?)""",
            (
                record.strategy_id,
                record.generation,
                json.dumps(record.parent_ids),
                record.operator,
                json.dumps(record.parameters, sort_keys=True),
            ),
        )
        self._connection.commit()

    def get_lineage(self, identifier: str) -> LineageRecord | None:
        row = self._connection.execute(
            "SELECT * FROM lineage WHERE strategy_id = ?", (identifier,)
        ).fetchone()
        if row is None:
            return None
        return LineageRecord(
            strategy_id=row["strategy_id"],
            generation=row["generation"],
            parent_ids=tuple(json.loads(row["parent_ids_json"])),
            operator=row["operator"],
            parameters=json.loads(row["parameters_json"]),
        )
