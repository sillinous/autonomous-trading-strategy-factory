from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite

from .portfolio_audit import PortfolioAuditEvent, audit_event_id


@dataclass(frozen=True)
class ExecutionManifest:
    """Immutable digest of the ordered paper-execution audit ledger."""

    event_count: int
    ledger_fingerprint: str


def execution_manifest(events: list[PortfolioAuditEvent] | tuple[PortfolioAuditEvent, ...]) -> ExecutionManifest:
    ordered = sorted(events, key=lambda event: event.sequence)
    if [event.sequence for event in ordered] != list(range(len(ordered))):
        raise ValueError("audit event sequences must be contiguous from zero")
    event_ids = [audit_event_id(event) for event in ordered]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("audit events must have unique identities")
    payload = {
        "event_count": len(ordered),
        "events": event_ids,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:16]
    return ExecutionManifest(len(ordered), digest)


def verify_execution_manifest(
    events: list[PortfolioAuditEvent] | tuple[PortfolioAuditEvent, ...],
    manifest: ExecutionManifest,
) -> bool:
    """Return whether an event ledger exactly matches a persisted manifest."""
    actual = execution_manifest(events)
    return actual == manifest
