from __future__ import annotations

from dataclasses import dataclass

from .execution_manifest import ExecutionManifest, execution_manifest
from .portfolio_audit import PortfolioAuditEvent


@dataclass(frozen=True)
class ReplayVerification:
    """Result of deterministic audit-ledger replay verification."""

    passed: bool
    expected: ExecutionManifest
    actual: ExecutionManifest
    reason: str | None = None


def verify_replay(
    events: list[PortfolioAuditEvent] | tuple[PortfolioAuditEvent, ...],
    expected: ExecutionManifest,
) -> ReplayVerification:
    """Recompute the execution ledger commitment and fail closed on mismatch."""
    try:
        actual = execution_manifest(events)
    except ValueError as exc:
        return ReplayVerification(False, expected, ExecutionManifest(0, ""), str(exc))
    if actual != expected:
        return ReplayVerification(False, expected, actual, "execution ledger fingerprint mismatch")
    return ReplayVerification(True, expected, actual)
