from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from .experiment import ExperimentResult, ExperimentSpec
from .lineage import LineageRecord
from .population import strategy_id
from .strategy import StrategySpec


class ExperimentRegistry:
    """SQLite persistence boundary for reproducible research and portfolio metadata."""

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
            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS portfolio_members (
                portfolio_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                weight REAL NOT NULL,
                rank INTEGER NOT NULL,
                PRIMARY KEY (portfolio_id, strategy_id),
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS portfolio_runs (
                run_id TEXT PRIMARY KEY,
                portfolio_id TEXT NOT NULL,
                final_equity REAL NOT NULL,
                halted INTEGER NOT NULL,
                halt_reason TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
            );
            CREATE TABLE IF NOT EXISTS portfolio_attribution (
                run_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                return_contribution REAL NOT NULL,
                risk_contribution REAL NOT NULL,
                PRIMARY KEY (run_id, strategy_id),
                FOREIGN KEY (run_id) REFERENCES portfolio_runs(run_id),
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
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
            (result.experiment_id, identifier, spec.dataset_id, spec.dataset_version,
             spec.seed, result.status, result.score, result.reason),
        )
        self._connection.commit()

    def save_evaluation_evidence(self, experiment_id: str, evidence: dict[str, Any]) -> None:
        payload = json.dumps(evidence, sort_keys=True, allow_nan=False)
        self._connection.execute(
            "INSERT OR REPLACE INTO evaluation_evidence(experiment_id, evidence_json) VALUES (?, ?)",
            (experiment_id, payload),
        )
        self._connection.commit()

    def get_evaluation_evidence(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT evidence_json FROM evaluation_evidence WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return None if row is None else json.loads(row["evidence_json"])

    def get_experiment(self, identifier: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT * FROM experiments WHERE experiment_id = ?", (identifier,)).fetchone()
        return None if row is None else dict(row)

    def list_experiments(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        if dataset_id is None:
            rows = self._connection.execute("SELECT * FROM experiments ORDER BY experiment_id").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM experiments WHERE dataset_id = ? ORDER BY experiment_id", (dataset_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def rank_experiments(self, dataset_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        query = "SELECT * FROM experiments WHERE status = 'eligible' "
        params: tuple[Any, ...] = ()
        if dataset_id is not None:
            query += "AND dataset_id = ? "
            params = (dataset_id,)
        query += "ORDER BY score DESC, experiment_id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params += (limit,)
        return [dict(row) for row in self._connection.execute(query, params).fetchall()]

    def save_lineage(self, record: LineageRecord) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO lineage(strategy_id, generation, parent_ids_json, operator, parameters_json) VALUES (?, ?, ?, ?, ?)",
            (record.strategy_id, record.generation, json.dumps(record.parent_ids), record.operator,
             json.dumps(record.parameters, sort_keys=True)),
        )
        self._connection.commit()

    def get_lineage(self, identifier: str) -> LineageRecord | None:
        row = self._connection.execute("SELECT * FROM lineage WHERE strategy_id = ?", (identifier,)).fetchone()
        if row is None:
            return None
        return LineageRecord(row["strategy_id"], row["generation"], tuple(json.loads(row["parent_ids_json"])),
                             row["operator"], json.loads(row["parameters_json"]))

    def save_portfolio(self, portfolio_id: str, definition: dict[str, Any], members: dict[str, float]) -> None:
        if not portfolio_id or not members:
            raise ValueError("portfolio_id and members are required")
        payload = json.dumps(definition, sort_keys=True, allow_nan=False)
        if any(not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0 for weight in members.values()):
            raise ValueError("portfolio weights must be finite and non-negative numbers")
        if sum(members.values()) > 1.0 + 1e-12:
            raise ValueError("portfolio weights exceed 100%")
        with self._connection:
            for identifier in members:
                if self.get_strategy(identifier) is None:
                    raise ValueError(f"unknown strategy: {identifier}")
            self._connection.execute("INSERT OR REPLACE INTO portfolios(portfolio_id, definition_json) VALUES (?, ?)", (portfolio_id, payload))
            self._connection.execute("DELETE FROM portfolio_members WHERE portfolio_id = ?", (portfolio_id,))
            self._connection.executemany(
                "INSERT INTO portfolio_members(portfolio_id, strategy_id, weight, rank) VALUES (?, ?, ?, ?)",
                [(portfolio_id, sid, float(weight), rank) for rank, (sid, weight) in enumerate(members.items())],
            )

    def get_portfolio(self, portfolio_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT definition_json FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)).fetchone()
        if row is None:
            return None
        members = self._connection.execute(
            "SELECT strategy_id, weight FROM portfolio_members WHERE portfolio_id = ? ORDER BY rank", (portfolio_id,)
        ).fetchall()
        return {"portfolio_id": portfolio_id, "definition": json.loads(row["definition_json"]),
                "members": {item["strategy_id"]: item["weight"] for item in members}}

    def save_portfolio_run(self, run_id: str, portfolio_id: str, final_equity: float, halted: bool,
                           halt_reason: str | None, attribution: list[dict[str, Any]]) -> None:
        if not math.isfinite(final_equity):
            raise ValueError("final_equity must be finite")
        if self._connection.execute("SELECT 1 FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)).fetchone() is None:
            raise ValueError(f"unknown portfolio: {portfolio_id}")
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO portfolio_runs(run_id, portfolio_id, final_equity, halted, halt_reason) VALUES (?, ?, ?, ?, ?)",
                (run_id, portfolio_id, final_equity, int(halted), halt_reason),
            )
            self._connection.execute("DELETE FROM portfolio_attribution WHERE run_id = ?", (run_id,))
            self._connection.executemany(
                "INSERT INTO portfolio_attribution(run_id, strategy_id, return_contribution, risk_contribution) VALUES (?, ?, ?, ?)",
                [(run_id, item["strategy_id"], item["return_contribution"], item["risk_contribution"])
                 for item in attribution],
            )

    def get_portfolio_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT * FROM portfolio_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        rows = self._connection.execute(
            "SELECT strategy_id, return_contribution, risk_contribution FROM portfolio_attribution WHERE run_id = ? ORDER BY strategy_id",
            (run_id,),
        ).fetchall()
        return {"run_id": row["run_id"], "portfolio_id": row["portfolio_id"],
                "final_equity": row["final_equity"], "halted": bool(row["halted"]),
                "halt_reason": row["halt_reason"], "attribution": [dict(item) for item in rows]}
