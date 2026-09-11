from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from .dataset_bundle import DatasetBundleIdentity
from .dataset_registry import DatasetRecord, DatasetRegistry
from .experiment import ExperimentResult, ExperimentSpec
from .lineage import LineageRecord
from .population import strategy_id
from .portfolio_audit import PortfolioAuditEvent, audit_event_id
from .strategy import StrategySpec


class ExperimentRegistry:
    """SQLite persistence boundary for reproducible research and portfolio metadata."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._datasets = DatasetRegistry(self._connection)
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS strategies (strategy_id TEXT PRIMARY KEY, definition_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
                dataset_version TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL,
                score REAL, reason TEXT, FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS lineage (
                strategy_id TEXT PRIMARY KEY, generation INTEGER NOT NULL, parent_ids_json TEXT NOT NULL,
                operator TEXT NOT NULL, parameters_json TEXT NOT NULL,
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS evaluation_evidence (
                experiment_id TEXT PRIMARY KEY, evidence_json TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            );
            CREATE TABLE IF NOT EXISTS portfolios (portfolio_id TEXT PRIMARY KEY, definition_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS portfolio_members (
                portfolio_id TEXT NOT NULL, strategy_id TEXT NOT NULL, weight REAL NOT NULL, rank INTEGER NOT NULL,
                PRIMARY KEY (portfolio_id, strategy_id), FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS portfolio_runs (
                run_id TEXT PRIMARY KEY, portfolio_id TEXT NOT NULL, final_equity REAL NOT NULL,
                halted INTEGER NOT NULL, halt_reason TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
            );
            CREATE TABLE IF NOT EXISTS portfolio_run_provenance (
                run_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, dataset_version TEXT NOT NULL,
                data_bundle_version TEXT NOT NULL, execution_fingerprint TEXT NOT NULL,
                execution_config_json TEXT NOT NULL, FOREIGN KEY (run_id) REFERENCES portfolio_runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS portfolio_attribution (
                run_id TEXT NOT NULL, strategy_id TEXT NOT NULL, return_contribution REAL NOT NULL,
                risk_contribution REAL NOT NULL, PRIMARY KEY (run_id, strategy_id),
                FOREIGN KEY (run_id) REFERENCES portfolio_runs(run_id), FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS portfolio_audit_events (
                run_id TEXT NOT NULL, event_id TEXT NOT NULL, sequence INTEGER NOT NULL, strategy_id TEXT NOT NULL,
                action TEXT NOT NULL, timestamp TEXT NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL, fee REAL NOT NULL,
                PRIMARY KEY (run_id, event_id), UNIQUE (run_id, sequence), FOREIGN KEY (run_id) REFERENCES portfolio_runs(run_id),
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS reproducibility_certificates (
                certificate_id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE, certificate_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (run_id) REFERENCES portfolio_runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_id, dataset_version);
            CREATE INDEX IF NOT EXISTS idx_experiments_strategy ON experiments(strategy_id);
            CREATE INDEX IF NOT EXISTS idx_portfolio_audit_strategy ON portfolio_audit_events(strategy_id, timestamp);
        """)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def register_dataset(self, identity: DatasetBundleIdentity, *, source: str) -> DatasetRecord:
        return self._datasets.register(identity, source=source)

    def get_dataset(self, dataset_id: str, version: str) -> DatasetRecord | None:
        return self._datasets.get(dataset_id, version)

    def require_dataset(self, dataset_id: str, version: str) -> DatasetRecord:
        return self._datasets.require(dataset_id, version)

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
        self._connection.execute("INSERT OR REPLACE INTO experiments(experiment_id, strategy_id, dataset_id, dataset_version, seed, status, score, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (result.experiment_id, identifier, spec.dataset_id, spec.dataset_version, spec.seed, result.status, result.score, result.reason))
        self._connection.commit()

    def save_evaluation_evidence(self, experiment_id: str, evidence: dict[str, Any]) -> None:
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
        self._connection.execute("INSERT OR REPLACE INTO lineage(strategy_id, generation, parent_ids_json, operator, parameters_json) VALUES (?, ?, ?, ?, ?)", (record.strategy_id, record.generation, json.dumps(record.parent_ids), record.operator, json.dumps(record.parameters, sort_keys=True)))
        self._connection.commit()

    def get_lineage(self, identifier: str) -> LineageRecord | None:
        row = self._connection.execute("SELECT * FROM lineage WHERE strategy_id = ?", (identifier,)).fetchone()
        if row is None:
            return None
        return LineageRecord(row["strategy_id"], row["generation"], tuple(json.loads(row["parent_ids_json"])), row["operator"], json.loads(row["parameters_json"]))

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
            existing_run = self._connection.execute("SELECT 1 FROM portfolio_runs WHERE portfolio_id = ? LIMIT 1", (portfolio_id,)).fetchone()
            if existing_run is not None:
                raise ValueError("portfolio is immutable after paper execution")
            self._connection.execute("INSERT OR REPLACE INTO portfolios(portfolio_id, definition_json) VALUES (?, ?)", (portfolio_id, payload))
            self._connection.execute("DELETE FROM portfolio_members WHERE portfolio_id = ?", (portfolio_id,))
            self._connection.executemany("INSERT INTO portfolio_members(portfolio_id, strategy_id, weight, rank) VALUES (?, ?, ?, ?)", [(portfolio_id, sid, float(weight), rank) for rank, (sid, weight) in enumerate(members.items())])

    def get_portfolio(self, portfolio_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT definition_json FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)).fetchone()
        if row is None:
            return None
        members = self._connection.execute("SELECT strategy_id, weight FROM portfolio_members WHERE portfolio_id = ? ORDER BY rank", (portfolio_id,)).fetchall()
        return {"portfolio_id": portfolio_id, "definition": json.loads(row["definition_json"]), "members": {item["strategy_id"]: item["weight"] for item in members}}

    def _validate_portfolio_run_payload(self, run_id: str, portfolio_id: str, final_equity: float, attribution: list[dict[str, Any]], dataset_id: str, dataset_version: str, data_bundle_version: str, execution_fingerprint: str, execution_config: dict[str, Any], audit_events: list[PortfolioAuditEvent]) -> tuple[str, list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        if not run_id or not math.isfinite(final_equity):
            raise ValueError("run_id and finite final_equity are required")
        for value, label in ((dataset_id, "dataset_id"), (dataset_version, "dataset_version"), (data_bundle_version, "data_bundle_version"), (execution_fingerprint, "execution_fingerprint")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} is required")
        execution_payload = json.dumps(execution_config, sort_keys=True, allow_nan=False)
        if self._connection.execute("SELECT 1 FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)).fetchone() is None:
            raise ValueError(f"unknown portfolio: {portfolio_id}")
        if self._connection.execute("SELECT 1 FROM portfolio_runs WHERE run_id = ?", (run_id,)).fetchone() is not None:
            raise ValueError("portfolio run already exists; execution is immutable")
        portfolio_members = {row[0] for row in self._connection.execute("SELECT strategy_id FROM portfolio_members WHERE portfolio_id = ?", (portfolio_id,)).fetchall()}
        seen: set[str] = set()
        attribution_rows: list[tuple[Any, ...]] = []
        for item in attribution:
            if not isinstance(item, dict) or "strategy_id" not in item:
                raise ValueError("attribution items require strategy_id")
            strategy = str(item["strategy_id"])
            if strategy not in portfolio_members or strategy in seen:
                raise ValueError("attribution must contain unique persisted portfolio members")
            if not all(math.isfinite(float(item[key])) for key in ("return_contribution", "risk_contribution")):
                raise ValueError("attribution values must be finite")
            seen.add(strategy)
            attribution_rows.append((run_id, strategy, float(item["return_contribution"]), float(item["risk_contribution"])))
        ordered = sorted(audit_events, key=lambda event: event.sequence)
        if [event.sequence for event in ordered] != list(range(len(ordered))):
            raise ValueError("audit event sequences must be contiguous from zero")
        audit_rows: list[tuple[Any, ...]] = []
        audit_ids: set[str] = set()
        for event in ordered:
            event_id = audit_event_id(event)
            if event_id in audit_ids or event.strategy_id not in portfolio_members:
                raise ValueError("audit events must be unique and reference persisted portfolio members")
            audit_ids.add(event_id)
            audit_rows.append((run_id, event_id, event.sequence, event.strategy_id, event.action, event.timestamp, event.quantity, event.price, event.fee))
        return execution_payload, attribution_rows, audit_rows

    def save_portfolio_execution(self, run_id: str, portfolio_id: str, final_equity: float, halted: bool, halt_reason: str | None, attribution: list[dict[str, Any]], *, dataset_id: str, dataset_version: str, data_bundle_version: str, execution_fingerprint: str, execution_config: dict[str, Any], audit_events: list[PortfolioAuditEvent]) -> None:
        execution_payload, attribution_rows, audit_rows = self._validate_portfolio_run_payload(run_id, portfolio_id, final_equity, attribution, dataset_id, dataset_version, data_bundle_version, execution_fingerprint, execution_config, audit_events)
        with self._connection:
            self._connection.execute("INSERT INTO portfolio_runs(run_id, portfolio_id, final_equity, halted, halt_reason) VALUES (?, ?, ?, ?, ?)", (run_id, portfolio_id, final_equity, int(halted), halt_reason))
            self._connection.execute("INSERT INTO portfolio_run_provenance(run_id, dataset_id, dataset_version, data_bundle_version, execution_fingerprint, execution_config_json) VALUES (?, ?, ?, ?, ?, ?)", (run_id, dataset_id, dataset_version, data_bundle_version, execution_fingerprint, execution_payload))
            self._connection.executemany("INSERT INTO portfolio_attribution(run_id, strategy_id, return_contribution, risk_contribution) VALUES (?, ?, ?, ?)", attribution_rows)
            self._connection.executemany("INSERT INTO portfolio_audit_events(run_id, event_id, sequence, strategy_id, action, timestamp, quantity, price, fee) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", audit_rows)

    def save_portfolio_run(self, run_id: str, portfolio_id: str, final_equity: float, halted: bool, halt_reason: str | None, attribution: list[dict[str, Any]], *, dataset_id: str = "unspecified", dataset_version: str = "unspecified", data_bundle_version: str = "unspecified", execution_fingerprint: str = "legacy", execution_config: dict[str, Any] | None = None) -> None:
        self.save_portfolio_execution(run_id, portfolio_id, final_equity, halted, halt_reason, attribution, dataset_id=dataset_id, dataset_version=dataset_version, data_bundle_version=data_bundle_version, execution_fingerprint=execution_fingerprint, execution_config=execution_config or {}, audit_events=[])

    def save_portfolio_audit_events(self, run_id: str, events: list[PortfolioAuditEvent]) -> None:
        if self._connection.execute("SELECT 1 FROM portfolio_runs WHERE run_id = ?", (run_id,)).fetchone() is None:
            raise ValueError(f"unknown portfolio run: {run_id}")
        portfolio_id = self._connection.execute("SELECT portfolio_id FROM portfolio_runs WHERE run_id = ?", (run_id,)).fetchone()[0]
        members = {row[0] for row in self._connection.execute("SELECT strategy_id FROM portfolio_members WHERE portfolio_id = ?", (portfolio_id,)).fetchall()}
        if self._connection.execute("SELECT 1 FROM portfolio_audit_events WHERE run_id = ? LIMIT 1", (run_id,)).fetchone() is not None:
            raise ValueError("audit ledger already exists; execution is immutable")
        ordered = sorted(events, key=lambda event: event.sequence)
        if [event.sequence for event in ordered] != list(range(len(ordered))):
            raise ValueError("audit event sequences must be contiguous from zero")
        rows = []
        seen: set[str] = set()
        for event in ordered:
            event_id = audit_event_id(event)
            if event_id in seen or event.strategy_id not in members:
                raise ValueError("audit events must be unique and reference persisted portfolio members")
            seen.add(event_id)
            rows.append((run_id, event_id, event.sequence, event.strategy_id, event.action, event.timestamp, event.quantity, event.price, event.fee))
        with self._connection:
            self._connection.executemany("INSERT INTO portfolio_audit_events(run_id, event_id, sequence, strategy_id, action, timestamp, quantity, price, fee) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def get_portfolio_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT * FROM portfolio_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        provenance = self._connection.execute("SELECT dataset_id, dataset_version, data_bundle_version, execution_fingerprint, execution_config_json FROM portfolio_run_provenance WHERE run_id = ?", (run_id,)).fetchone()
        attribution = self._connection.execute("SELECT strategy_id, return_contribution, risk_contribution FROM portfolio_attribution WHERE run_id = ? ORDER BY strategy_id", (run_id,)).fetchall()
        audit = self._connection.execute("SELECT event_id, sequence, strategy_id, action, timestamp, quantity, price, fee FROM portfolio_audit_events WHERE run_id = ? ORDER BY sequence", (run_id,)).fetchall()
        result = {"run_id": row["run_id"], "portfolio_id": row["portfolio_id"], "final_equity": row["final_equity"], "halted": bool(row["halted"]), "halt_reason": row["halt_reason"], "attribution": [dict(item) for item in attribution], "audit_events": [dict(item) for item in audit]}
        if provenance is not None:
            result["provenance"] = {"dataset_id": provenance["dataset_id"], "dataset_version": provenance["dataset_version"], "data_bundle_version": provenance["data_bundle_version"], "execution_fingerprint": provenance["execution_fingerprint"], "execution_config": json.loads(provenance["execution_config_json"])}
        return result

    def save_reproducibility_certificate(self, certificate: Any) -> None:
        run_id = str(certificate.run_id)
        certificate_id = str(certificate.certificate_id)
        if not run_id or not certificate_id:
            raise ValueError("certificate run_id and certificate_id are required")
        if self.get_portfolio_run(run_id) is None:
            raise ValueError(f"unknown portfolio run: {run_id}")
        payload = json.dumps({"run_id": run_id, "certificate_id": certificate_id, "provenance_schema_version": certificate.provenance_schema_version, "verified": bool(certificate.verified), "dataset_version": certificate.dataset_version, "data_bundle_version": certificate.data_bundle_version, "execution_fingerprint": certificate.execution_fingerprint, "research_fingerprint": certificate.research_fingerprint, "provenance_graph_fingerprint": certificate.provenance_graph_fingerprint, "ledger_fingerprint": certificate.ledger_fingerprint, "lineage_fingerprint": certificate.lineage_fingerprint, "attribution_fingerprint": certificate.attribution_fingerprint, "verification_fingerprint": certificate.verification_fingerprint, "event_count": int(certificate.event_count)}, sort_keys=True, allow_nan=False)
        existing = self._connection.execute("SELECT certificate_id, certificate_json FROM reproducibility_certificates WHERE run_id = ?", (run_id,)).fetchone()
        if existing is not None:
            if existing["certificate_id"] != certificate_id or existing["certificate_json"] != payload:
                raise ValueError("reproducibility certificate already exists and is immutable")
            return
        with self._connection:
            self._connection.execute("INSERT INTO reproducibility_certificates(certificate_id, run_id, certificate_json) VALUES (?, ?, ?)", (certificate_id, run_id, payload))

    def get_reproducibility_certificate(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT certificate_json FROM reproducibility_certificates WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else json.loads(row["certificate_json"])
