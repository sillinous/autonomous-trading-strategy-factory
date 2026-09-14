from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .research_cycle import ResearchCycleResult
from .research_history import ResearchHistory


class ResearchHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class ResearchHealthDecision(str, Enum):
    CONTINUE = "continue"
    INCREASE_EXPLORATION = "increase_exploration"
    PAUSE_RESEARCH = "pause_research"


@dataclass(frozen=True)
class ResearchHealthPolicy:
    """Fail-closed thresholds for deterministic research-control diagnostics."""

    min_unique_strategy_ratio: float = 0.75
    min_novel_strategy_ratio: float = 0.10
    min_mean_genome_distance: float = 0.05
    max_stagnation_generations: int = 3
    min_promotion_eligible_ratio: float = 0.05
    exploration_mutation_saturation: float = 0.90
    exploration_crossover_floor: float = 0.10

    def __post_init__(self) -> None:
        for name, value in (
            ("min_unique_strategy_ratio", self.min_unique_strategy_ratio),
            ("min_novel_strategy_ratio", self.min_novel_strategy_ratio),
            ("min_mean_genome_distance", self.min_mean_genome_distance),
            ("min_promotion_eligible_ratio", self.min_promotion_eligible_ratio),
            ("exploration_mutation_saturation", self.exploration_mutation_saturation),
            ("exploration_crossover_floor", self.exploration_crossover_floor),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if self.max_stagnation_generations < 1:
            raise ValueError("max_stagnation_generations must be positive")


@dataclass(frozen=True)
class ResearchHealth:
    """Machine-readable research-run health; never grants execution authority."""

    status: ResearchHealthStatus
    decision: ResearchHealthDecision
    diversity_collapsed: bool
    novelty_exhausted: bool
    stagnating: bool
    promotion_eligibility_collapsed: bool
    exploration_saturated: bool
    unique_strategy_ratio: float
    novel_strategy_ratio: float
    promotion_eligible_ratio: float
    mean_genome_distance: float
    reasons: tuple[str, ...]


def assess_research_health(
    history: ResearchHistory,
    result: ResearchCycleResult,
    *,
    policy: ResearchHealthPolicy | None = None,
) -> ResearchHealth:
    """Assess deterministic research health from persisted history and latest metrics."""
    policy = policy or ResearchHealthPolicy()
    if not history.records:
        raise ValueError("history must contain the evaluated generation")
    if history.latest.generation != result.generation:
        raise ValueError("history latest generation must match research result")
    if result.metrics.candidate_count <= 0:
        raise ValueError("candidate_count must be positive")

    record = history.latest
    candidate_count = result.metrics.candidate_count
    unique_ratio = record.unique_strategy_count / candidate_count
    novel_ratio = record.new_strategy_count / candidate_count
    promotion_ratio = result.metrics.promotion_eligible_count / candidate_count

    diversity_collapsed = record.mean_genome_distance < policy.min_mean_genome_distance
    novelty_exhausted = novel_ratio < policy.min_novel_strategy_ratio
    stagnating = record.generations_without_improvement >= policy.max_stagnation_generations
    promotion_collapsed = promotion_ratio < policy.min_promotion_eligible_ratio
    exploration_saturated = (
        result.metrics.mutation_rate >= policy.exploration_mutation_saturation
        and result.metrics.crossover_rate <= policy.exploration_crossover_floor
    )

    reasons: list[str] = []
    if unique_ratio < policy.min_unique_strategy_ratio:
        reasons.append("strategy uniqueness is below threshold")
    if diversity_collapsed:
        reasons.append("genome diversity is below threshold")
    if novelty_exhausted:
        reasons.append("strategy novelty is below threshold")
    if stagnating:
        reasons.append("fitness improvement has stagnated")
    if promotion_collapsed:
        reasons.append("promotion eligibility is below threshold")
    if exploration_saturated:
        reasons.append("adaptive exploration pressure is saturated")

    critical = (
        diversity_collapsed and novelty_exhausted
    ) or stagnating or promotion_collapsed
    warning = bool(reasons)

    if critical:
        status = ResearchHealthStatus.CRITICAL
        decision = ResearchHealthDecision.PAUSE_RESEARCH
    elif warning:
        status = ResearchHealthStatus.WARNING
        decision = ResearchHealthDecision.INCREASE_EXPLORATION
    else:
        status = ResearchHealthStatus.HEALTHY
        decision = ResearchHealthDecision.CONTINUE

    return ResearchHealth(
        status=status,
        decision=decision,
        diversity_collapsed=diversity_collapsed,
        novelty_exhausted=novelty_exhausted,
        stagnating=stagnating,
        promotion_eligibility_collapsed=promotion_collapsed,
        exploration_saturated=exploration_saturated,
        unique_strategy_ratio=unique_ratio,
        novel_strategy_ratio=novel_ratio,
        promotion_eligible_ratio=promotion_ratio,
        mean_genome_distance=record.mean_genome_distance,
        reasons=tuple(reasons),
    )
