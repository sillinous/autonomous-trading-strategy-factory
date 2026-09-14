from types import SimpleNamespace

import pytest

from atsf.research_health import (
    ResearchHealthDecision,
    ResearchHealthPolicy,
    ResearchHealthStatus,
    assess_research_health,
)
from atsf.research_history import GenerationRecord, ResearchHistory


def make_state(
    *,
    unique=4,
    novel=2,
    distance=0.20,
    stagnation=0,
    promotion=2,
    mutation=0.50,
    crossover=0.50,
):
    history = ResearchHistory(
        records=(
            GenerationRecord(
                generation=0,
                candidate_count=4,
                selected_count=2,
                promoted_count=promotion,
                best_fitness=1.0,
                mean_fitness=0.8,
                mean_genome_distance=distance,
                min_genome_distance=0.05,
                max_genome_distance=0.40,
                unique_strategy_count=unique,
                new_strategy_count=novel,
                best_strategy_id="best",
                generations_without_improvement=stagnation,
            ),
        ),
        known_strategy_ids=frozenset({"a", "b", "c", "d"}),
    )
    result = SimpleNamespace(
        generation=0,
        metrics=SimpleNamespace(
            candidate_count=4,
            promotion_eligible_count=promotion,
            mutation_rate=mutation,
            crossover_rate=crossover,
        ),
    )
    return history, result


def test_healthy_research_continues():
    history, result = make_state()

    health = assess_research_health(history, result)

    assert health.status is ResearchHealthStatus.HEALTHY
    assert health.decision is ResearchHealthDecision.CONTINUE
    assert health.reasons == ()
    assert health.unique_strategy_ratio == 1.0
    assert health.novel_strategy_ratio == 0.5
    assert health.promotion_eligible_ratio == 0.5


def test_warning_recommends_more_exploration_for_diversity_or_novelty_pressure():
    history, result = make_state(unique=2, novel=0)

    health = assess_research_health(history, result)

    assert health.status is ResearchHealthStatus.WARNING
    assert health.decision is ResearchHealthDecision.INCREASE_EXPLORATION
    assert not health.stagnating
    assert health.diversity_collapsed is False
    assert health.novelty_exhausted is True
    assert any("uniqueness" in reason for reason in health.reasons)


def test_critical_stagnation_pauses_research():
    history, result = make_state(stagnation=3)

    health = assess_research_health(history, result)

    assert health.status is ResearchHealthStatus.CRITICAL
    assert health.decision is ResearchHealthDecision.PAUSE_RESEARCH
    assert health.stagnating is True


def test_critical_diversity_and_novelty_collapse_pauses_research():
    history, result = make_state(unique=2, novel=0, distance=0.01)

    health = assess_research_health(history, result)

    assert health.status is ResearchHealthStatus.CRITICAL
    assert health.decision is ResearchHealthDecision.PAUSE_RESEARCH
    assert health.diversity_collapsed is True
    assert health.novelty_exhausted is True


def test_exploration_saturation_is_reported_as_warning():
    history, result = make_state(mutation=0.95, crossover=0.05)
    policy = ResearchHealthPolicy(
        min_unique_strategy_ratio=0.5,
        min_novel_strategy_ratio=0.1,
        min_mean_genome_distance=0.05,
        min_promotion_eligible_ratio=0.1,
    )

    health = assess_research_health(history, result, policy=policy)

    assert health.status is ResearchHealthStatus.WARNING
    assert health.decision is ResearchHealthDecision.INCREASE_EXPLORATION
    assert health.exploration_saturated is True


def test_health_rejects_mismatched_history_generation():
    history, result = make_state()
    result.generation = 1

    with pytest.raises(ValueError, match="match research result"):
        assess_research_health(history, result)


def test_health_policy_validates_thresholds():
    with pytest.raises(ValueError, match="min_novel_strategy_ratio"):
        ResearchHealthPolicy(min_novel_strategy_ratio=1.1)
    with pytest.raises(ValueError, match="max_stagnation_generations"):
        ResearchHealthPolicy(max_stagnation_generations=0)
