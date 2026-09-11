from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .portfolio_replay import ReplayVerification


@dataclass(frozen=True)
class ReproducibilityCertificate:
    """Deterministic attestation of a verified immutable paper execution."""

    run_id: str
    certificate_id: str
    verified: bool
    dataset_version: str
    data_bundle_version: str
    execution_fingerprint: str
    ledger_fingerprint: str
    lineage_fingerprint: str
    event_count: int


def _lineage_fingerprint(config: dict[str, Any]) -> str:
    lineage = config.get("fill_lineage")
    if not isinstance(lineage, list):
        raise ValueError("execution is missing fill lineage")
    encoded = json.dumps(lineage, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_reproducibility_certificate(
    run: dict[str, Any],
    verification: ReplayVerification,
) -> ReproducibilityCertificate:
    """Build a deterministic certificate only from a successful replay verification."""
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
    lineage_fingerprint = _lineage_fingerprint(config)
    payload = {
        "run_id": str(run.get("run_id")),
        "dataset_version": dataset_version,
        "data_bundle_version": bundle_version,
        "execution_fingerprint": execution_fingerprint,
        "ledger_fingerprint": verification.manifest.ledger_fingerprint,
        "lineage_fingerprint": lineage_fingerprint,
        "event_count": verification.manifest.event_count,
    }
    certificate_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:24]
    return ReproducibilityCertificate(
        run_id=str(run["run_id"]),
        certificate_id=certificate_id,
        verified=True,
        dataset_version=dataset_version,
        data_bundle_version=bundle_version,
        execution_fingerprint=execution_fingerprint,
        ledger_fingerprint=verification.manifest.ledger_fingerprint,
        lineage_fingerprint=lineage_fingerprint,
        event_count=verification.manifest.event_count,
    )
