from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class PaperAdmissionRecord:
    admission_id: str
    strategy_id: str
    run_id: str
    certificate_id: str
    source_stage: str
    target_stage: str
    reason: str


def build_admission_record(
    strategy_id: str,
    run_id: str,
    certificate_id: str,
    *,
    reason: str = "verified reproducibility evidence",
) -> PaperAdmissionRecord:
    """Build a deterministic, content-addressed paper-admission record."""
    payload = {
        "strategy_id": strategy_id,
        "run_id": run_id,
        "certificate_id": certificate_id,
        "source_stage": "PROMOTED",
        "target_stage": "PAPER",
        "reason": reason,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    admission_id = hashlib.sha256(encoded).hexdigest()[:24]
    return PaperAdmissionRecord(admission_id=admission_id, **payload)
