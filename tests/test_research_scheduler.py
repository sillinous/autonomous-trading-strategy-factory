from types import SimpleNamespace

import pytest

from atsf.research_health import (
    ResearchHealthDecision,
    ResearchHealthPolicy,
    ResearchHealthStatus,
)
from atsf.research_history import GenerationRecord, ResearchHistory
from atsf.research_scheduler import (
    ResearchScheduleAction,
    ResearchSchedulePolicy,
    schedule_research,
)


def make_state(*, promotion=2, stagnation=0, generation=0):
    history = ResearchHistory(
        records=(
            GenerationRecord(
                generation=generation,
                candidate_count=4,
                selected_count=2,
                promoted_count=promotion,
                best_fitness=1.0,
                mean_fitness=0.8,
                mean_genome_distance=0.20,
                min_genome_distance=0.05,
                max_genome_distance=0.40,
                unique_strategy_count=4,
                new_strategy_count=2,
                best_strategy_id="best",
                generations_without_improvement=stagnation,
            ),
        ),
        known_strategy_ids=frozenset({"a", "b", "c", "d"}),
    )
    result = SimpleNamespace(
        generation=generation,
        metrics=SimpleNamespace(
            candidate_count=4,
            promotion_eligible_count=promotion,
            mutation_rate=0.50,
            crossover_rate=0.50,
        ),
    )
    return history, result


def test_healthy_research_can_be_ready_for_robustness_review():
    history, result = make_state()

    schedule = schedule_research(history, result)

    assert schedule.action is ResearchScheduleAction.READY_FOR_ROBUSTNESS_REVIEW
    assert schedule.health.status is ResearchHealthStatus.HEALTHY
    assert schedule.health.decision is ResearchHealthDecision.CONTINUE
    assert schedule.execution_authority is False


def test_scheduler_continues_until_minimum_research_depth():
    history, result = make_state()

    schedule = schedule_research(
        history,
        result,
        policy=ResearchSchedulePolicy(min_generations_before_review=3),
    )

    assert schedule.action is ResearchScheduleAction.CONTINUE_RESEARCH
    assert any("minimum research generations" in reason for reason in schedule.reasons)


def test_scheduler_requires_promotion_candidate_for_review():
    history, result = make_state(promotion=0)

    schedule = schedule_research(
        history,
        result,
        health_policy=ResearchHealthPolicy(min_promotion_eligible_ratio=0.0),
    )

    assert schedule.action is ResearchScheduleAction.CONTINUE_RESEARCH
    assert any("no promotion-eligible" in reason for reason in schedule.reasons)


def test_scheduler_increases_exploration_for_warning_health():
    history, result = make_state()
    history = ResearchHistory(
        records=(
            GenerationRecord(
                generation=0,
                candidate_count=4,
                selected_count=2,
                promoted_count=2,
                best_fitness=1.0,
                mean_fitness=0.8,
                mean_genome_distance=0.01,
                min_genome_distance=0.01,
                max_genome_distance=0.02,
                unique_strategy_count=2,
                new_strategy_count=1,
                best_strategy_id="best",
                generations_without_improvement=0,
            ),
        ),
        known_strategy_ids=frozenset({"a", "b", "c", "d"}),
    )

    schedule = schedule_research(history, result)

    assert schedule.action is ResearchScheduleAction.INCREASE_EXPLORATION
    assert schedule.execution_authority is False


def test_scheduler_pauses_on_critical_health():
    history, result = make_state(stagnation=3)

    schedule = schedule_research(history, result)

    assert schedule.action is ResearchScheduleAction.PAUSE_RESEARCH
    assert schedule.health.status is ResearchHealthStatus.CRITICAL
    assert schedule.execution_authority is False


def test_scheduler_policy_validates_minimum_depth():
    with pytest.raises(ValueError, match="min_generations_before_review"):
        ResearchSchedulePolicy(min_generations_before_review=0)


def test_scheduler_fingerprint_is_deterministic_and_bound():
    history, result = make_state()
    first = schedule_research(history, result)
    second = schedule_research(history, result)

    assert first.fingerprint
    assert first.fingerprint == second.fingerprint

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        type(first)(
            action=first.action,
            health=first.health,
            generation=first.generation,
            execution_authority=False,
            reasons=first.reasons,
            fingerprint="tampered",
        )


def test_scheduler_rejects_execution_authority():
    history, result = make_state()
    schedule = schedule_research(history, result)

    with pytest.raises(ValueError, match="cannot grant execution authority"):
        type(schedule)(
            action=schedule.action,
            health=schedule.health,
            generation=schedule.generation,
            execution_authority=True,
            reasons=schedule.reasons,
        )
