from __future__ import annotations

from types import SimpleNamespace

import pytest

from atsf.population import seed_population
from atsf.research_budget import ResearchAllocation
from atsf.research_pipeline import evaluate_research_execution, execute_research_plan
from atsf.research_planner import ResearchPlan
from atsf.research_queue import ResearchReason, ResearchRequest
from atsf.strategy import Comparator, Condition, Indicator, PositionSizing, RiskLimits, Signal, StrategySpec


def _strategy(period: int = 20) -> StrategySpec:
    return StrategySpec(
        name="pipeline",
        universe=["TEST"],
        indicators=[Indicator(name="sma", period=period)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=100)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=100)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def _plan(source_id: str) -> ResearchPlan:
    request = ResearchRequest("research:degraded:source", source_id, ResearchReason.DEGRADED, 100)
    return ResearchPlan(
        generation=1,
        requests=(request,),
        allocations=(ResearchAllocation(request.request_id, 3, 1.0),),
    )


def test_execute_research_plan_is_deterministic_and_bounded():
    population = tuple(seed_population([_strategy()]))
    first = execute_research_plan(_plan(population[0].strategy_id), population, seed=12)
    second = execute_research_plan(_plan(population[0].strategy_id), population, seed=12)
    assert first == second
    assert len(first.work_items) == 1
    assert len(first.candidates) <= 3
    assert all(candidate.lineage.parent_ids == (population[0].strategy_id,) for candidate in first.candidates)
    assert all(candidate.strategy_id != population[0].strategy_id for candidate in first.candidates)


def test_execute_research_plan_rejects_unknown_parent():
    population = tuple(seed_population([_strategy()]))
    with pytest.raises(KeyError, match="unknown source strategy"):
        execute_research_plan(_plan("missing"), population)


def test_execute_research_plan_rejects_non_integer_seed():
    population = tuple(seed_population([_strategy()]))
    with pytest.raises(TypeError, match="research seed"):
        execute_research_plan(_plan(population[0].strategy_id), population, seed="12")


def test_evaluate_research_execution_uses_normal_candidate_gate(monkeypatch):
    population = tuple(seed_population([_strategy()]))
    execution = execute_research_plan(_plan(population[0].strategy_id), population, seed=12)
    assert execution.candidates
    calls = []

    def fake_evaluate(candidate, data, dataset_id, dataset_version, **kwargs):
        calls.append((candidate.strategy_id, dataset_id, dataset_version, kwargs["seed"]))
        return SimpleNamespace(
            candidate_id=candidate.strategy_id,
            promotion=SimpleNamespace(eligible=candidate.strategy_id.endswith("0")),
        )

    monkeypatch.setattr("atsf.research_pipeline.evaluate_candidate", fake_evaluate)
    result = evaluate_research_execution(
        execution,
        data=object(),
        dataset_id="dataset",
        dataset_version="v1",
        seed=20,
    )
    assert len(calls) == len(execution.candidates)
    assert [call[3] for call in calls] == list(range(20, 20 + len(calls)))
    assert result.evaluations
    assert result.eligible_strategy_ids == tuple(
        evaluation.candidate_id
        for evaluation in result.evaluations
        if evaluation.promotion.eligible
    )
