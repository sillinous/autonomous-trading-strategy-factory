from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .dataset_bundle import bundle_identity
from .execution_lineage import FillLineage, verify_fill_lineage
from .execution_manifest import ExecutionManifest, execution_manifest
from .ledger_replay import validate_execution_ledger
from .portfolio_audit import PortfolioAuditEvent
from .portfolio_run import build_portfolio_run_identity
from .registry import ExperimentRegistry
from .signals import strategy_signals


@dataclass(frozen=True)
class ReplayVerification:
    run_id: str
    valid: bool
    manifest: ExecutionManifest
    reason: str | None = None


def _failure(run_id: str, reason: str, manifest: ExecutionManifest | None = None) -> ReplayVerification:
    return ReplayVerification(run_id, False, manifest or ExecutionManifest(0, ""), reason)


def _stored_lineage(config: dict, run_id: str, manifest: ExecutionManifest) -> tuple[FillLineage, ...] | ReplayVerification:
    raw = config.get("fill_lineage")
    if not isinstance(raw, list) or len(raw) != manifest.event_count:
        return _failure(run_id, "stored execution is missing a complete fill lineage", manifest)
    lineage: list[FillLineage] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"event_id", "decision_id", "reason"}:
            return _failure(run_id, "stored fill lineage has an invalid record", manifest)
        event_id = item["event_id"]
        decision = item["decision_id"]
        reason = item["reason"]
        if not all(isinstance(value, str) and value for value in (event_id, decision, reason)):
            return _failure(run_id, "stored fill lineage contains invalid identifiers", manifest)
        if event_id in seen:
            return _failure(run_id, "stored fill lineage contains duplicate event IDs", manifest)
        seen.add(event_id)
        lineage.append(FillLineage(event_id, decision, reason))
    return tuple(lineage)


def verify_persisted_portfolio_run(
    store: ExperimentRegistry,
    run_id: str,
    data: dict[str, pd.DataFrame] | None = None,
) -> ReplayVerification:
    """Verify persisted identity, provenance, accounting, and optionally signal replay."""
    run = store.get_portfolio_run(run_id)
    if run is None:
        raise ValueError(f"unknown portfolio run: {run_id}")

    provenance = run.get("provenance") or {}
    config = provenance.get("execution_config") or {}
    if not isinstance(config, dict):
        return _failure(run_id, "stored execution configuration is invalid")

    stored_dataset_version = provenance.get("dataset_version")
    stored_bundle_version = provenance.get("data_bundle_version")
    stored_execution_fingerprint = provenance.get("execution_fingerprint")
    if not isinstance(stored_dataset_version, str) or not stored_dataset_version:
        return _failure(run_id, "stored execution is missing dataset_version")
    if not isinstance(stored_bundle_version, str) or not stored_bundle_version:
        return _failure(run_id, "stored execution is missing data_bundle_version")
    if not isinstance(stored_execution_fingerprint, str) or not stored_execution_fingerprint:
        return _failure(run_id, "stored execution is missing execution fingerprint")

    portfolio = store.get_portfolio(str(run["portfolio_id"]))
    if portfolio is None:
        return _failure(run_id, "persisted portfolio referenced by run is missing")
    weights = dict(portfolio["members"])
    try:
        identity_config = {
            key: value for key, value in config.items()
            if key not in {"ledger_fingerprint", "ledger_event_count", "fill_lineage"}
        }
        expected = build_portfolio_run_identity(
            str(run["portfolio_id"]),
            stored_dataset_version,
            weights,
            execution_config=identity_config,
            data_fingerprint=stored_bundle_version,
        )
    except (TypeError, ValueError):
        return _failure(run_id, "persisted execution identity inputs are invalid")

    if expected.run_id != run_id:
        return _failure(run_id, "portfolio run ID does not match persisted execution inputs")
    if expected.execution_fingerprint != stored_execution_fingerprint:
        return _failure(run_id, "execution fingerprint does not match persisted execution inputs")

    events = tuple(
        PortfolioAuditEvent(
            sequence=int(row["sequence"]),
            strategy_id=str(row["strategy_id"]),
            action=str(row["action"]),
            timestamp=str(row["timestamp"]),
            quantity=float(row["quantity"]),
            price=float(row["price"]),
            fee=float(row["fee"]),
        )
        for row in run["audit_events"]
    )
    try:
        actual = execution_manifest(events)
    except (TypeError, ValueError) as exc:
        return _failure(run_id, str(exc))

    stored_fingerprint = config.get("ledger_fingerprint")
    stored_count = config.get("ledger_event_count")
    if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
        return _failure(run_id, "stored execution is missing a ledger fingerprint commitment", actual)
    if not isinstance(stored_count, int):
        return _failure(run_id, "stored execution is missing a valid ledger event count", actual)
    if stored_count != actual.event_count or stored_fingerprint != actual.ledger_fingerprint:
        return _failure(run_id, "persisted execution ledger fingerprint mismatch", actual)

    lineage = _stored_lineage(config, run_id, actual)
    if isinstance(lineage, ReplayVerification):
        return lineage
    expected_event_ids = tuple(item.event_id for item in lineage)
    actual_event_ids = tuple(__import__("atsf.portfolio_audit", fromlist=["audit_event_id"]).audit_event_id(event) for event in sorted(events, key=lambda item: item.sequence))
    if expected_event_ids != actual_event_ids:
        return _failure(run_id, "stored fill lineage event IDs do not match the audit ledger", actual)

    if provenance.get("dataset_id") != portfolio["definition"].get("dataset_id"):
        return _failure(run_id, "run dataset_id does not match persisted portfolio")
    if stored_dataset_version != portfolio["definition"].get("dataset_version"):
        return _failure(run_id, "run dataset_version does not match persisted portfolio")
    if stored_bundle_version != portfolio["definition"].get("data_bundle_version"):
        return _failure(run_id, "run data_bundle_version does not match persisted portfolio")

    try:
        initial_cash = float(config["initial_cash"])
        commission_bps = float(config["commission_bps"])
        strategy_cash = {strategy_id: initial_cash * float(weight) for strategy_id, weight in weights.items() if float(weight) > 0}
        reserve = initial_cash - sum(strategy_cash.values())
        validate_execution_ledger(
            events,
            initial_cash_by_strategy=strategy_cash,
            commission_bps=commission_bps,
            initial_cash=sum(strategy_cash.values()),
            final_equity=float(run["final_equity"]) - reserve,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _failure(run_id, f"execution ledger accounting mismatch: {exc}", actual)

    if data is not None:
        if set(data) != set(weights):
            return _failure(run_id, "replay market data members do not match persisted portfolio", actual)
        dataset = store.require_dataset(str(portfolio["definition"]["dataset_id"]), stored_dataset_version)
        try:
            bundle = bundle_identity(
                data,
                str(portfolio["definition"]["dataset_id"]),
                source=dataset.source,
                timeframe=dataset.timeframe,
                schema_version=dataset.schema_version,
            )
        except (TypeError, ValueError) as exc:
            return _failure(run_id, f"replay market data is invalid: {exc}", actual)
        if bundle.version != stored_bundle_version:
            return _failure(run_id, "replay market data does not match persisted data bundle", actual)
        strategies = {}
        for strategy_id in weights:
            strategy = store.get_strategy(strategy_id)
            if strategy is None:
                return _failure(run_id, f"persisted strategy is missing: {strategy_id}", actual)
            strategies[strategy_id] = strategy
        try:
            signals = {strategy_id: strategy_signals(data[strategy_id], strategy) for strategy_id, strategy in strategies.items()}
            liquidation_timestamp = pd.Timestamp(events[-1].timestamp) if bool(run["halted"]) and events else None
            if not verify_fill_lineage(
                tuple(sorted(events, key=lambda item: item.sequence)),
                lineage,
                signals,
                halted=bool(run["halted"]),
                liquidation_timestamp=liquidation_timestamp,
            ):
                return _failure(run_id, "persisted fill lineage does not reproduce from stored strategies and market data", actual)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return _failure(run_id, f"signal replay failed: {exc}", actual)

    return ReplayVerification(run_id, True, actual)
