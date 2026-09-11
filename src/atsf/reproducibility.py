from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .portfolio_replay import ReplayVerification
from .registry import ExperimentRegistry


@dataclass(frozen=True)
class ReproducibilityCertificate:
    """Deterministic attestation of a verified, provenance-complete paper execution."""

    run_id: str
    certificate_id: str
    provenance_schema_version: str
    verified: bool
    dataset_version: str
    data_bundle_version: str
    execution_fingerprint: str
    research_fingerprint: str
    ledger_fingerprint: str
    lineage_fingerprint: str
    attribution_fingerprint: str
    verification_fingerprint: str
    event_count: int


def _fingerprint(payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _lineage_chain(store: ExperimentRegistry, strategy_ids: set[str]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    pending = sorted(strategy_ids)
    while pending:
        identifier = pending.pop(0)
        if identifier in records:
            continue
        strategy = store.get_strategy(identifier)
        lineage = store.get_lineage(identifier)
        if strategy is None or lineage is None:
            raise ValueError(f"strategy provenance is incomplete: {identifier}")
        records[identifier] = {
            "strategy_id": identifier,
            "definition": strategy.model_dump(mode="json"),
            "generation": lineage.generation,
            "parent_ids": list(lineage.parent_ids),
            "operator": lineage.operator,
            "parameters": lineage.parameters,
        }
        pending.extend(parent for parent in lineage.parent_ids if parent not in records)
    return [records[identifier] for identifier in sorted(records)]


def _research_provenance(store: ExperimentRegistry, run: dict[str, Any]) -> dict[str, Any]:
    portfolio_id = run.get("portfolio_id")
    portfolio = store.get_portfolio(str(portfolio_id)) if portfolio_id else None
    if portfolio is None:
        raise ValueError("portfolio provenance is missing")
    definition = portfolio.get("definition")
    members = portfolio.get("members")
    if not isinstance(definition, dict) or not isinstance(members, dict) or not members:
        raise ValueError("portfolio provenance is incomplete")

    experiment_map = definition.get("experiment_ids")
    if not isinstance(experiment_map, dict) or not experiment_map:
        raise ValueError("portfolio experiment provenance is missing")
    experiment_ids = sorted({str(value) for value in experiment_map.values()})
    experiments: list[dict[str, Any]] = []
    strategy_ids: set[str] = set(members)
    for experiment_id in experiment_ids:
        experiment = store.get_experiment(experiment_id)
        evidence = store.get_evaluation_evidence(experiment_id)
        if experiment is None or evidence is None:
            raise ValueError(f"experiment provenance is incomplete: {experiment_id}")
        strategy_ids.add(str(experiment["strategy_id"]))
        experiments.append(
            {
                "experiment_id": experiment_id,
                "strategy_id": experiment["strategy_id"],
                "dataset_id": experiment["dataset_id"],
                "dataset_version": experiment["dataset_version"],
                "seed": experiment["seed"],
                "status": experiment["status"],
                "score": experiment["score"],
                "reason": experiment["reason"],
                "evidence": evidence,
            }
        )

    run_attribution = run.get("attribution")
    if not isinstance(run_attribution, list) or not run_attribution:
        raise ValueError("portfolio attribution is missing")
    attribution = sorted(run_attribution, key=lambda item: str(item.get("strategy_id")))
    provenance = run.get("provenance") or {}
    config = provenance.get("execution_config")
    if not isinstance(config, dict):
        raise ValueError("execution configuration is invalid")

    return {
        "portfolio_id": portfolio_id,
        "portfolio_definition": definition,
        "portfolio_members": {key: members[key] for key in sorted(members)},
        "experiments": experiments,
        "lineage": _lineage_chain(store, strategy_ids),
        "attribution": attribution,
        "dataset": {
            "dataset_id": provenance.get("dataset_id"),
            "dataset_version": provenance.get("dataset_version"),
            "data_bundle_version": provenance.get("data_bundle_version"),
            "data_source": definition.get("data_source"),
            "data_timeframe": definition.get("data_timeframe"),
            "data_schema_version": definition.get("data_schema_version"),
        },
        "execution_config": config,
    }


def build_reproducibility_certificate(
    store: ExperimentRegistry,
    run: dict[str, Any],
    verification: ReplayVerification,
) -> ReproducibilityCertificate:
    """Build a deterministic certificate only from a successful, complete replay verification."""
    if not verification.valid:
        raise ValueError("cannot certify an unverified portfolio run")
    provenance = run.get("provenance") or {}
    config = provenance.get("execution_config") or {}
    if not isinstance(config, dict):
        raise ValueError("execution configuration is invalid")
    dataset_version = provenance.get("dataset_version")
    bundle_version = provenance.get("data_bundle_version")
    execution_fingerprint = provenance.get("execution_fingerprint")
    if not all(isinstance(value, str) and value for value in (dataset_version, bundle_version, execution_fingerprint)):
        raise ValueError("execution provenance is incomplete")

    research = _research_provenance(store, run)
    ledger_fingerprint = verification.manifest.ledger_fingerprint
    lineage_fingerprint = _fingerprint(research["lineage"])
    attribution_fingerprint = _fingerprint(research["attribution"])
    verification_fingerprint = _fingerprint(
        {
            "valid": verification.valid,
            "reason": verification.reason,
            "event_count": verification.manifest.event_count,
            "ledger_fingerprint": ledger_fingerprint,
        }
    )
    research_fingerprint = _fingerprint(
        {
            "portfolio_id": research["portfolio_id"],
            "portfolio_definition": research["portfolio_definition"],
            "portfolio_members": research["portfolio_members"],
            "experiments": research["experiments"],
            "lineage_fingerprint": lineage_fingerprint,
            "dataset": research["dataset"],
            "execution_config": research["execution_config"],
            "attribution_fingerprint": attribution_fingerprint,
        },
        length=24,
    )
    payload = {
        "provenance_schema_version": "1",
        "run_id": str(run.get("run_id")),
        "dataset_version": dataset_version,
        "data_bundle_version": bundle_version,
        "execution_fingerprint": execution_fingerprint,
        "research_fingerprint": research_fingerprint,
        "ledger_fingerprint": ledger_fingerprint,
        "lineage_fingerprint": lineage_fingerprint,
        "attribution_fingerprint": attribution_fingerprint,
        "verification_fingerprint": verification_fingerprint,
        "event_count": verification.manifest.event_count,
    }
    certificate_id = _fingerprint(payload, length=24)
    return ReproducibilityCertificate(
        run_id=str(run["run_id"]),
        certificate_id=certificate_id,
        provenance_schema_version="1",
        verified=True,
        dataset_version=dataset_version,
        data_bundle_version=bundle_version,
        execution_fingerprint=execution_fingerprint,
        research_fingerprint=research_fingerprint,
        ledger_fingerprint=ledger_fingerprint,
        lineage_fingerprint=lineage_fingerprint,
        attribution_fingerprint=attribution_fingerprint,
        verification_fingerprint=verification_fingerprint,
        event_count=verification.manifest.event_count,
    )
