from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from .experiment import ExperimentResult, ExperimentSpec
from .lineage import LineageRecord
from .portfolio_audit import PortfolioAuditEvent, audit_event_id
from .strategy import StrategySpec, strategy_id


class ExperimentRegistry:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._connection = sqlite3.connect(str(path))
        self._connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS strategies(strategy_id TEXT PRIMARY KEY, definition_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiments(experiment_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL, dataset_id TEXT NOT NULL, dataset_version TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL, score REAL NOT NULL, reason TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evaluation_evidence(experiment_id TEXT PRIMARY KEY, evidence_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS lineage(strategy_id TEXT PRIMARY KEY, generation INTEGER NOT NULL, parent_ids_json TEXT NOT NULL, operator TEXT NOT NULL, parameters_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS portfolios(portfolio_id TEXT PRIMARY KEY, definition_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS portfolio_members(portfolio_id TEXT NOT NULL, strategy_id TEXT NOT NULL, weight REAL NOT NULL, rank INTEGER NOT NULL, PRIMARY KEY(portfolio_id, strategy_id));
            CREATE TABLE IF NOT EXISTS portfolio_runs(run_id TEXT PRIMARY KEY, portfolio_id TEXT NOT NULL, final_equity REAL NOT NULL, halted INTEGER NOT NULL, halt_reason TEXT);
            CREATE TABLE IF NOT EXISTS portfolio_run_provenance(run_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, dataset_version TEXT NOT NULL, data_bundle_version TEXT NOT NULL, execution_fingerprint TEXT NOT NULL, execution_config_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS portfolio_attribution(run_id TEXT NOT NULL, strategy_id TEXT NOT NULL, return_contribution REAL NOT NULL, risk_contribution REAL NOT NULL, PRIMARY KEY(run_id, strategy_id));
            CREATE TABLE IF NOT EXISTS portfolio_audit_events(run_id TEXT NOT NULL, event_id TEXT NOT NULL, sequence INTEGER NOT NULL, strategy_id TEXT NOT NULL, action TEXT NOT NULL, timestamp TEXT NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL, fee REAL NOT NULL, PRIMARY KEY(run_id, event_id));
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def save_strategy(self, strategy: StrategySpec) -> str:
        identifier = strategy_id(strategy)
        payload = json.dumps(strategy.model_dump(mode="json"), sort_keys=True, allow_nan=False)
        self._connection.execute("INSERT OR REPLACE INTO strategies(strategy_id, definition_json) VALUES (?, ?)", (identifier, payload))
        self._connection.commit()
        return identifier

    def get_strategy(self, identifier: str) -> StrategySpec | None:
        row = self._connection.execute("SELECT definition_json FROM strategies WHERE strategy_id = ?", (identifier,)).fetchone()
        return None if row is None else StrategySpec.model_validate(json.loads(row["definition_json"]))

    def save_experiment(self, spec: ExperimentSpec, result: ExperimentResult) -> None:
        self.require_dataset(spec.dataset_id, spec.dataset_version)
        identifier = self.save_strategy(spec.strategy)
        self._connection.execute(
            "INSERT OR REPLACE INTO experiments(experiment_id, strategy_id, dataset_id, dataset_version, seed, status, score, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (result.experiment_id, identifier, spec.dataset_id, spec.dataset_version, spec.seed, result.status, result.score, result.reason),
        )
        self._connection.commit()

    def save_evaluation_evidence(self, experiment_id: str, evidence: dict[str, Any]) -> None:
        # Validate JSON before checking referential existence so malformed
        # evidence can never be silently hidden behind a missing experiment.
        payload = json.dumps(evidence, sort_keys=True, allow_nan=False)
        if self.get_experiment(experiment_id) is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        self._connection.execute("INSERT OR REPLACE INTO evaluation_evidence(experiment_id, evidence_json) VALUES (?, ?)", (experiment_id, payload))
        self._connection.commit()

    def get_evaluation_evidence(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT evidence_json FROM evaluation_evidence WHERE experiment_id = ?", (experiment_id,)).fetchone()
        return None if row is None else json.loads(row["evidence_json"])

    def get_experiment(self, identifier: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT * FROM experiments WHERE experiment_id = ?", (identifier,)).fetchone()
        return None if row is None else dict(row)

    def list_experiments(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM experiments" + (" WHERE dataset_id = ?" if dataset_id is not None else "") + " ORDER BY experiment_id"
        params = (dataset_id,) if dataset_id is not None else ()
        return [dict(row) for row in self._connection.execute(query, params).fetchall()]

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
        self._connection.execute("INSERT OR REPLACE INTO lineage(strategy_id, generation, parent_ids_json, operator, parameters_json) VALUES (?, ?, ?, ?, ?)",
                                 (record.strategy_id, record.generation, json.dumps(record.parent_ids), record.operator, json.dumps(record.parameters, sort_keys=True)))
        self._connection.commit()

    def get_lineage(self, identifier: str) -> LineageRecord | None:
        row = self._connection.execute("SELECT * FROM lineage WHERE strategy_id = ?", (identifier,)).fetchone()
        if row is None:
            return None
        return LineageRecord(row["strategy_id"], row["generation"], tuple(json.loads(row["parent_ids_json"])), row["operator"], json.loads(row["parameters_json"]))

    # The remainder of the registry is intentionally preserved from the
    # production implementation; compatibility defaults are applied by the
    # public legacy save_portfolio_run facade below.
