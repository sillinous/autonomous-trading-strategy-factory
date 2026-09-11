from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .registry import ExperimentRegistry


@dataclass(frozen=True)
class CertificateIntegrity:
    run_id: str
    certificate_id: str
    valid: bool
    reason: str | None


def _certificate_id(certificate: dict[str, Any]) -> str:
    payload = {
        "provenance_schema_version": certificate.get("provenance_schema_version"),
        "run_id": certificate.get("run_id"),
        "dataset_version": certificate.get("dataset_version"),
        "data_bundle_version": certificate.get("data_bundle_version"),
        "execution_fingerprint": certificate.get("execution_fingerprint"),
        "research_fingerprint": certificate.get("research_fingerprint"),
        "provenance_graph_fingerprint": certificate.get("provenance_graph_fingerprint"),
        "ledger_fingerprint": certificate.get("ledger_fingerprint"),
        "lineage_fingerprint": certificate.get("lineage_fingerprint"),
        "attribution_fingerprint": certificate.get("attribution_fingerprint"),
        "verification_fingerprint": certificate.get("verification_fingerprint"),
        "event_count": certificate.get("event_count"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def verify_persisted_certificate(store: ExperimentRegistry, run_id: str) -> CertificateIntegrity:
    certificate = store.get_reproducibility_certificate(run_id)
    if certificate is None:
        return CertificateIntegrity(run_id, "", False, "reproducibility certificate not found")
    certificate_id = str(certificate.get("certificate_id", ""))
    if certificate.get("run_id") != run_id:
        return CertificateIntegrity(run_id, certificate_id, False, "certificate run_id does not match requested run")
    if certificate.get("verified") is not True:
        return CertificateIntegrity(run_id, certificate_id, False, "certificate is not marked verified")
    required = (
        "provenance_schema_version", "dataset_version", "data_bundle_version",
        "execution_fingerprint", "research_fingerprint", "provenance_graph_fingerprint",
        "ledger_fingerprint", "lineage_fingerprint", "attribution_fingerprint",
        "verification_fingerprint", "event_count",
    )
    if any(key not in certificate for key in required):
        return CertificateIntegrity(run_id, certificate_id, False, "certificate is structurally incomplete")
    try:
        expected = _certificate_id(certificate)
    except (TypeError, ValueError):
        return CertificateIntegrity(run_id, certificate_id, False, "certificate contains non-canonical values")
    if not certificate_id or not expected == certificate_id:
        return CertificateIntegrity(run_id, certificate_id, False, "certificate_id does not match certificate contents")
    return CertificateIntegrity(run_id, certificate_id, True, None)
