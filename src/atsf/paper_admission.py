from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .lifecycle import (
    PromotionLifecycleEvent,
    StrategyLifecycleStage,
    transition_promotion_stage,
)


@dataclass(frozen=True)
class PaperAdmissionDecision:
    admitted: bool
    event: PromotionLifecycleEvent | None
    reasons: tuple[str, ...]


def admit_to_paper(
    strategy_id: str,
    *,
    source_stage: StrategyLifecycleStage,
    run: dict[str, Any],
    certificate: dict[str, Any] | None,
    reason: str = "verified reproducibility evidence",
) -> PaperAdmissionDecision:
    """Fail-closed boundary from PROMOTED to PAPER.

    A strategy may enter paper execution only when its lifecycle is explicitly
    PROMOTED and the referenced portfolio execution has a persisted, verified
    reproducibility certificate. No broker or live-execution capability is
    involved in this gate.
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
    event = transition_promotion_stage(
        strategy_id,
        source_stage,
        StrategyLifecycleStage.PAPER,
        reason=reason,
    )
    return PaperAdmissionDecision(True, event, ())
