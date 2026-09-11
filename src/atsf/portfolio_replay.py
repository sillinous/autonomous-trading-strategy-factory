from __future__ import annotations

from dataclasses import dataclass

from .execution_manifest import ExecutionManifest, execution_manifest
from .portfolio_audit import PortfolioAuditEvent
from .registry import ExperimentRegistry


@dataclass(frozen=True)
class ReplayVerification:
    run_id: str
    valid: bool
    stored_manifest: ExecutionManifest
    actual_manifest: ExecutionManifest
    reason: str | None = None


def verify_persisted_portfolio_run(store: ExperimentRegistry, run_id: str) -> ReplayVerification:
    """Verify the persisted audit ledger for a paper run without executing orders."""
    run = store.get_portfolio_run(run_id)
    if run is None:
        raise ValueError(f"unknown portfolio run: {run_id}")
    rows = run["audit_events"]
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
        for row in rows
    )
    actual = execution_manifest(events)
    stored = ExecutionManifest(event_count=len(events), ledger_fingerprint=run.get("ledger_fingerprint", actual.ledger_fingerprint))
    valid = actual == stored
    return ReplayVerification(run_id, valid, stored, actual, None if valid else "persisted execution ledger fingerprint mismatch")
