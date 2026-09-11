from __future__ import annotations

from dataclasses import dataclass

from .execution_manifest import ExecutionManifest, execution_manifest
from .portfolio_audit import PortfolioAuditEvent
from .registry import ExperimentRegistry


@dataclass(frozen=True)
class ReplayVerification:
    run_id: str
    valid: bool
    manifest: ExecutionManifest
    reason: str | None = None


def verify_persisted_portfolio_run(store: ExperimentRegistry, run_id: str) -> ReplayVerification:
    """Validate a persisted paper-run ledger without executing any orders."""
    run = store.get_portfolio_run(run_id)
    if run is None:
        raise ValueError(f"unknown portfolio run: {run_id}")
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
        manifest = execution_manifest(events)
    except ValueError as exc:
        return ReplayVerification(run_id, False, ExecutionManifest(0, ""), str(exc))
    return ReplayVerification(run_id, True, manifest)
