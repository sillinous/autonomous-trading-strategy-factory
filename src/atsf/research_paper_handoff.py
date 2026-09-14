from __future__ import annotations

from dataclasses import dataclass

from .registry import ExperimentRegistry
from .robustness_review import RobustnessReview, RobustnessReviewDecision
from .successor_paper_handoff import handoff_successor_to_paper


@dataclass(frozen=True)
class ResearchPaperHandoffDecision:
    """Auditable boundary from an admitted research candidate to PAPER."""

    strategy_id: str
    run_id: str
    admitted: bool
    admission_id: str | None
    review_fingerprint: str
    execution_authority: bool
    reasons: tuple[str, ...]


def handoff_research_candidate_to_paper(
    registry: ExperimentRegistry,
    strategy_id: str,
    run_id: str,
    review: RobustnessReview,
    *,
    reason: str = "verified robustness-review handoff",
) -> ResearchPaperHandoffDecision:
    """Require a successful robustness review before invoking the PAPER boundary.

    This function delegates durable portfolio, certificate, lifecycle, and paper
    admission verification to the existing successor handoff. It does not grant
    live execution authority.
    """
    reasons: list[str] = []
    if not strategy_id.strip():
        reasons.append("strategy_id is required")
    if not run_id.strip():
        reasons.append("run_id is required")
    if review.candidate_id != strategy_id:
        reasons.append("robustness review candidate_id does not match strategy")
    if review.decision is not RobustnessReviewDecision.ADMIT:
        reasons.append("robustness review did not admit the candidate")
    if review.execution_authority:
        reasons.append("robustness review incorrectly carries execution authority")
    if not review.evidence_fingerprint:
        reasons.append("robustness review evidence fingerprint is missing")

    if reasons:
        return ResearchPaperHandoffDecision(
            strategy_id,
            run_id,
            False,
            None,
            review.evidence_fingerprint,
            False,
            tuple(reasons),
        )

    handoff = handoff_successor_to_paper(
        registry,
        strategy_id,
        run_id,
        reason=reason,
    )
    return ResearchPaperHandoffDecision(
        strategy_id=strategy_id,
        run_id=run_id,
        admitted=handoff.admitted,
        admission_id=handoff.admission_id,
        review_fingerprint=review.evidence_fingerprint,
        execution_authority=False,
        reasons=handoff.reasons,
    )
