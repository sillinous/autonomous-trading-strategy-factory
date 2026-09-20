from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from .research_health import (
    ResearchHealth,
    ResearchHealthDecision,
    ResearchHealthPolicy,
    ResearchHealthStatus,
    assess_research_health,
)
from .research_cycle import ResearchCycleResult
from .research_history import ResearchHistory


class ResearchScheduleAction(str, Enum):
    CONTINUE_RESEARCH = "continue_research"
    INCREASE_EXPLORATION = "increase_exploration"
    PAUSE_RESEARCH = "pause_research"
    READY_FOR_ROBUSTNESS_REVIEW = "ready_for_robustness_review"


@dataclass(frozen=True)
class ResearchSchedulePolicy:
    """Explicit scheduling gates; this layer has no execution authority."""

    min_generations_before_review: int = 1
    require_promotion_candidate: bool = True

    def __post_init__(self) -> None:
        if self.min_generations_before_review < 1:
            raise ValueError("min_generations_before_review must be positive")


@dataclass(frozen=True)
class ResearchSchedule:
    """Immutable next-step decision derived only from research evidence."""

    action: ResearchScheduleAction
    health: ResearchHealth
    generation: int
    execution_authority: bool
    reasons: tuple[str, ...]
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if self.execution_authority:
            raise ValueError("research schedule cannot grant execution authority")
        payload = {
            "action": self.action.value,
            "generation": self.generation,
            "health": {
                "status": self.health.status.value,
                "decision": self.health.decision.value,
                "diversity_collapsed": self.health.diversity_collapsed,
                "novelty_exhausted": self.health.novelty_exhausted,
                "stagnating": self.health.stagnating,
                "promotion_eligibility_collapsed": self.health.promotion_eligibility_collapsed,
                "exploration_saturated": self.health.exploration_saturated,
                "unique_strategy_ratio": self.health.unique_strategy_ratio,
                "novel_strategy_ratio": self.health.novel_strategy_ratio,
                "promotion_eligible_ratio": self.health.promotion_eligible_ratio,
                "mean_genome_distance": self.health.mean_genome_distance,
                "reasons": self.health.reasons,
            },
            "reasons": self.reasons,
            "execution_authority": False,
        }
        canonical = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.fingerprint and self.fingerprint != expected:
            raise ValueError("research schedule fingerprint mismatch")
        object.__setattr__(self, "fingerprint", expected)


def schedule_research(
    history: ResearchHistory,
    result: ResearchCycleResult,
    *,
    health_policy: ResearchHealthPolicy | None = None,
    policy: ResearchSchedulePolicy | None = None,
) -> ResearchSchedule:
    """Select the next bounded research action without granting execution authority."""
    policy = policy or ResearchSchedulePolicy()
    health = assess_research_health(history, result, policy=health_policy)
    reasons = list(health.reasons)

    if health.decision is ResearchHealthDecision.PAUSE_RESEARCH:
        action = ResearchScheduleAction.PAUSE_RESEARCH
    elif health.decision is ResearchHealthDecision.INCREASE_EXPLORATION:
        action = ResearchScheduleAction.INCREASE_EXPLORATION
    elif result.generation + 1 < policy.min_generations_before_review:
        action = ResearchScheduleAction.CONTINUE_RESEARCH
        reasons.append("minimum research generations have not been completed")
    elif policy.require_promotion_candidate and result.metrics.promotion_eligible_count <= 0:
        action = ResearchScheduleAction.CONTINUE_RESEARCH
        reasons.append("no promotion-eligible candidate is available for robustness review")
    else:
        action = ResearchScheduleAction.READY_FOR_ROBUSTNESS_REVIEW
        reasons.append("research health is healthy and review prerequisites are satisfied")

    return ResearchSchedule(
        action=action,
        health=health,
        generation=result.generation,
        execution_authority=False,
        reasons=tuple(reasons),
    )
