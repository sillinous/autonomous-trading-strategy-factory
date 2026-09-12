from __future__ import annotations

from dataclasses import dataclass

from .certificate_integrity import verify_persisted_certificate
from .paper_admission_store import PaperAdmissionStore
from .registry import ExperimentRegistry


@dataclass(frozen=True)
class PaperReplayResult:
    replayable: bool
    admission_id: str | None
    strategy_id: str | None
    reasons: tuple[str, ...]


def verify_paper_replay(registry: ExperimentRegistry, run_id: str) -> PaperReplayResult:
    """Fail-closed verification of a persisted PAPER admission."""
    store = PaperAdmissionStore(registry)
    record = store.get(run_id)
    if record is None:
        return PaperReplayResult(False, None, None, ("paper admission is missing",))
    reasons: list[str] = []
    if record.source_stage != "PROMOTED":
        reasons.append("admission source stage is not PROMOTED")
    if record.target_stage != "PAPER":
        reasons.append("admission target stage is not PAPER")
    run = registry._connection.execute(
        "SELECT halted FROM portfolio_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run is None:
        reasons.append("portfolio execution run is missing")
    elif run["halted"]:
        reasons.append("portfolio execution run is halted")
    certificate = verify_persisted_certificate(registry, run_id)
    if not certificate.valid:
        reasons.append(f"reproducibility certificate is invalid: {certificate.reason}")
    elif certificate.certificate_id != record.certificate_id:
        reasons.append("admission certificate identity does not match persisted certificate")
    if not store.verify(run_id):
        reasons.append("durable paper admission verification failed")
    return PaperReplayResult(
        not reasons,
        record.admission_id,
        record.strategy_id,
        tuple(dict.fromkeys(reasons)),
    )
