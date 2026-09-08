from __future__ import annotations

import json
import sqlite3
from pathlib import Path

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
                reason TEXT
            );
            CREATE TABLE IF NOT EXISTS lineage (
                strategy_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                parent_ids_json TEXT NOT NULL,
                operator TEXT NOT NULL,
                parameters_json TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def save_strategy(self, strategy: StrategySpec) -> str:
        identifier = strategy_id(strategy)
        payload = json.dumps(strategy.model_dump(mode="json"), sort_keys=True)
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
