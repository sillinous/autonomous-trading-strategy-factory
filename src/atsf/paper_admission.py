from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .lifecycle import (
    PromotionLifecycleEvent,
    StrategyLifecycleStage,
    transition_promotion_stage,
)
from .paper_admission_record import build_admission_record
from .paper_admission_store import PaperAdmissionStore
from .registry import ExperimentRegistry


@dataclass(frozen=True)
class PaperAdmissionDecision:
    admitted: bool
    event: PromotionLifecycleEvent | None
    reasons: tuple[str, ...]
    admission_id: str | None = None


def admit_to_paper(
    strategy_id: str,
    *,
    source_stage: StrategyLifecycleStage,
    run: dict[str, Any],
    certificate: dict[str, Any] | None,
    reason: str = "verified reproducibility evidence",
    registry: ExperimentRegistry | None = None,
) -> PaperAdmissionDecision:
    """Fail-closed boundary from PROMOTED to PAPER.

    When a registry is supplied, the admission is durably persisted before the
    lifecycle event is returned. No broker or live-execution capability exists
    in this boundary.
    """
    reasons: list[str] = []
    if source_stage is not StrategyLifecycleStage.PROMOTED:
        reasons.append("strategy must be in promoted lifecycle stage")
    if not isinstance(run, dict) or not run.get("run_id"):
        reasons.append("portfolio execution provenance is missing")
    elif run.get("halted"):
        reasons.append("portfolio execution is halted")
    if certificate is None:
        reasons.append("verified reproducibility certificate is missing")
    else:
        if certificate.get("run_id") != run.get("run_id"):
            reasons.append("certificate run_id does not match portfolio run")
        if certificate.get("verified") is not True:
            reasons.append("reproducibility certificate is not verified")
        if not certificate.get("certificate_id"):
            reasons.append("certificate identity is missing")
        for field in ("dataset_version", "data_bundle_version", "execution_fingerprint"):
            if not certificate.get(field):
                reasons.append(f"certificate {field} is missing")
    if reasons:
        return PaperAdmissionDecision(False, None, tuple(reasons))

    record = build_admission_record(
        strategy_id,
        str(run["run_id"]),
        str(certificate["certificate_id"]),
        reason=reason,
    )
    if registry is not None:
        try:
            PaperAdmissionStore(registry).save(record)
        except (KeyError, ValueError) as exc:
            return PaperAdmissionDecision(False, None, (str(exc),), record.admission_id)

    event = transition_promotion_stage(
        strategy_id,
        source_stage,
        StrategyLifecycleStage.PAPER,
        reason=reason,
    )
    return PaperAdmissionDecision(True, event, (), record.admission_id)
