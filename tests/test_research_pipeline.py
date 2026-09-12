from __future__ import annotations

import pytest

from atsf.population import seed_population
from atsf.research_budget import ResearchAllocation
from atsf.research_pipeline import execute_research_plan
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
