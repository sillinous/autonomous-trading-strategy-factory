from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .orchestrator import CandidateEvaluation
from .population import Candidate
from .research_provenance import CandidateProvenance, _candidate_record
from .research_scheduler import ResearchSchedule, ResearchScheduleAction


class RobustnessReviewDecision(str, Enum):
    ADMIT = "admit"
    REJECT = "reject"


@dataclass(frozen=True)
class RobustnessReviewPolicy:
    """Admission gates for robustness review; never grants execution authority."""

    require_validation: bool = True
    require_promotion_eligibility: bool = True
    require_robustness_pass: bool = True


@dataclass(frozen=True)
class RobustnessReview:
    """Immutable, auditable admission result for a single candidate."""

    candidate_id: str
    generation: int
    decision: RobustnessReviewDecision
    execution_authority: bool
    evidence_fingerprint: str
    reasons: tuple[str, ...]



def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and isfinite(float(value))


def _validate_evidence(
    candidate: Candidate,
    evaluation: CandidateEvaluation,
    provenance: CandidateProvenance,
) -> list[str]:
    reasons: list[str] = []
    if provenance.strategy_id != candidate.strategy_id:
        reasons.append("provenance strategy_id does not match candidate")
    if evaluation.candidate_id != candidate.strategy_id:
        reasons.append("evaluation candidate_id does not match candidate")
    if provenance.generation < 0:
        reasons.append("provenance generation is invalid")
    if provenance.research_seed < 0:
        reasons.append("provenance research seed is invalid")

    try:
        expected = _candidate_record(
            candidate,
            provenance.generation,
            evaluation,
            provenance.research_seed,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        reasons.append(f"provenance reconstruction failed: {exc}")
        return reasons

    if expected.genome_digest != provenance.genome_digest:
        reasons.append("provenance genome digest does not match candidate")
    if expected.evaluation_digest != provenance.evaluation_digest:
        reasons.append("provenance evaluation digest does not match evidence")
    return reasons


def review_robustness(
    candidate: Candidate,
    evaluation: CandidateEvaluation,
    schedule: ResearchSchedule,
    provenance: CandidateProvenance,
    *,
    policy: RobustnessReviewPolicy | None = None,
) -> RobustnessReview:
    """Admit a candidate to robustness review only when all evidence gates pass."""
    policy = policy or RobustnessReviewPolicy()
    reasons = _validate_evidence(candidate, evaluation, provenance)

    if schedule.action is not ResearchScheduleAction.READY_FOR_ROBUSTNESS_REVIEW:
        reasons.append("research scheduler has not authorized robustness review")
    if schedule.execution_authority:
        reasons.append("research scheduler incorrectly carries execution authority")
    if schedule.generation != provenance.generation:
        reasons.append("schedule generation does not match provenance generation")
    if policy.require_validation and not evaluation.validation_passed:
        reasons.append("candidate validation did not pass")
    if policy.require_promotion_eligibility and not evaluation.promotion.eligible:
        reasons.append("candidate is not promotion-eligible")
    if evaluation.robustness.candidate_id != candidate.strategy_id:
        reasons.append("robustness evidence candidate_id does not match candidate")
    if policy.require_robustness_pass and not evaluation.robustness.passed:
        reasons.append("robustness stress scenarios did not pass")

    for name, value in (
        ("oos_sharpe", evaluation.walk_forward.oos_sharpe),
        ("fitness_score", evaluation.fitness.score),
        ("monte_carlo_pass_rate", evaluation.monte_carlo.pass_rate),
    ):
        if not _finite(value):
            reasons.append(f"{name} is non-finite")

    if not _finite(evaluation.robustness.baseline.equity.iloc[-1]):
        reasons.append("robustness baseline final equity is non-finite")

    decision = (
        RobustnessReviewDecision.ADMIT
        if not reasons
        else RobustnessReviewDecision.REJECT
    )
    fingerprint = expected_fingerprint(candidate, evaluation, provenance)
    return RobustnessReview(
        candidate_id=candidate.strategy_id,
        generation=provenance.generation,
        decision=decision,
        execution_authority=False,
        evidence_fingerprint=fingerprint,
        reasons=tuple(reasons),
    )


def expected_fingerprint(
    candidate: Candidate,
    evaluation: CandidateEvaluation,
    provenance: CandidateProvenance,
) -> str:
    """Return the provenance fingerprint expected for the supplied evidence."""
    return _candidate_record(
        candidate,
        provenance.generation,
        evaluation,
        provenance.research_seed,
    ).evaluation_digest
