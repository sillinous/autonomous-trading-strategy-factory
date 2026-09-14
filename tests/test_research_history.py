from types import SimpleNamespace

import pytest

from atsf.research_history import ResearchHistory
from atsf.research_cycle import ResearchCycleResult


def candidate(candidate_id: str, feature: int):
    return SimpleNamespace(
        strategy_id=candidate_id,
        strategy=SimpleNamespace(model_dump=lambda mode=None, feature=feature: {"feature": feature}),
    )


def result(generation: int, scores: dict[str, float], selected: list):
    evaluations = tuple(
        SimpleNamespace(candidate_id=candidate_id, fitness=SimpleNamespace(score=score))
        for candidate_id, score in scores.items()
    )
    metrics = SimpleNamespace(
        selected_count=len(selected),
        promoted_count=sum(score >= 1.0 for score in scores.values()),
        best_fitness=max(scores.values()),
        mean_fitness=sum(scores.values()) / len(scores),
    )
    return ResearchCycleResult(
        generation=generation,
        evaluations=evaluations,
        selected_parents=tuple(selected),
        next_population=tuple(),
        metrics=metrics,
    )


def test_history_records_diversity_and_new_strategies():
    population = [candidate("a", 1), candidate("b", 2), candidate("c", 3)]
    selected = [population[0]]
    history = ResearchHistory().record_generation(
        result(0, {"a": 1.0, "b": 0.8, "c": 0.6}, selected), population
    )
    record = history.latest
    assert record is not None
    assert record.generation == 0
    assert record.unique_strategy_count == 3
    assert record.new_strategy_count == 2
    assert record.mean_genome_distance == 1.0
    assert record.min_genome_distance == 1.0
    assert record.max_genome_distance == 1.0
    assert record.generations_without_improvement == 0
    assert history.best_fitness == 1.0


def test_history_detects_stagnation_and_resets_on_improvement():
    population = [candidate("a", 1), candidate("b", 2)]
    history = ResearchHistory()
    history = history.record_generation(result(0, {"a": 1.0, "b": 0.5}, population), population)
    history = history.record_generation(result(1, {"a": 1.0, "b": 0.4}, population), population)
    assert history.generations_without_improvement == 1
    assert history.is_stagnating(1)
    history = history.record_generation(result(2, {"a": 1.1, "b": 0.4}, population), population)
    assert history.generations_without_improvement == 0
    assert not history.is_stagnating(1)


def test_history_is_immutable_and_validates_inputs():
    population = [candidate("a", 1)]
    history = ResearchHistory().record_generation(result(0, {"a": 1.0}, population), population)
    assert len(history.records) == 1
    assert len(ResearchHistory().records) == 0
    with pytest.raises(ValueError, match="monotonically"):
        history.record_generation(result(0, {"a": 1.0}, population), population)
    with pytest.raises(ValueError, match="epsilon"):
        history.record_generation(result(1, {"a": 1.0}, population), population, improvement_epsilon=-1)
    with pytest.raises(ValueError, match="threshold"):
        history.is_stagnating(0)
