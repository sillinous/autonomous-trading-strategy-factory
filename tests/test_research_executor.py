from __future__ import annotations

import pytest

from atsf.population import seed_population
from atsf.research_budget import ResearchAllocation
from atsf.research_executor import (
    consume_research_work,
    materialize_research_work,
    spawn_research_candidates,
)
from atsf.research_planner import ResearchPlan
from atsf.research_queue import ResearchQueue, ResearchReason, ResearchRequest
from atsf.strategy import Comparator, Condition, Indicator, PositionSizing, RiskLimits, Signal, StrategySpec


def _strategy(period: int = 20, threshold: int = 100) -> StrategySpec:
    return StrategySpec(
        name="executor",
        universe=["TEST"],
        indicators=[Indicator(name="sma", period=period)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=threshold)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=threshold)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def _plan() -> ResearchPlan:
    requests = (
        ResearchRequest("research:degraded:s1", "s1", ResearchReason.DEGRADED, 900),
        ResearchRequest("research:capacity:s2", "s2", ResearchReason.CAPACITY, 500),
    )
    allocations = (
        ResearchAllocation("research:degraded:s1", 60, 0.9),
        ResearchAllocation("research:capacity:s2", 40, 0.5),
    )
    return ResearchPlan(generation=4, requests=requests, allocations=allocations)


def test_materialization_is_deterministic_and_enqueues_requests() -> None:
    queue = ResearchQueue()
    first = materialize_research_work(_plan(), queue=queue)
    second = materialize_research_work(_plan(), queue=queue)
    assert first == second
    assert [item.request_id for item in first] == [request.request_id for request in _plan().requests]
    assert [item.budget_units for item in first] == [60, 40]
    assert len(queue) == 2


def test_missing_allocation_fails_closed() -> None:
    plan = ResearchPlan(
        generation=1,
        requests=(ResearchRequest("r", None, ResearchReason.CAPACITY, 1),),
        allocations=(),
    )
    with pytest.raises(ValueError, match="missing allocation"):
        materialize_research_work(plan)


def test_consume_completes_matching_request() -> None:
    queue = ResearchQueue()
    work = materialize_research_work(_plan(), queue=queue)
    completed = consume_research_work(queue, work[0])
    assert completed.request_id == work[0].request_id
    assert len(queue) == 1


def test_consume_rejects_unknown_request() -> None:
    queue = ResearchQueue()
    work = materialize_research_work(_plan(), queue=queue)
    consume_research_work(queue, work[0])
    with pytest.raises(KeyError, match="unknown research request"):
        consume_research_work(queue, work[0])


def test_spawn_is_bounded_deterministic_and_records_lineage() -> None:
    population = tuple(seed_population([_strategy()]))
    source = population[0]
    work = materialize_research_work(_plan())[0]
    work = work.__class__(work.request_id, source.strategy_id, work.reason, work.priority, 5, work.constraints)
    first = spawn_research_candidates(work, population, seed=41)
    second = spawn_research_candidates(work, population, seed=41)
    assert first == second
    assert len(first) <= 5
    assert all(candidate.strategy_id != source.strategy_id for candidate in first)
    assert all(candidate.lineage.parent_ids == (source.strategy_id,) for candidate in first)
    assert all(candidate.lineage.operator.startswith("mutate_") for candidate in first)


def test_spawn_requires_known_source_and_positive_budget() -> None:
    population = tuple(seed_population([_strategy()]))
    work = materialize_research_work(_plan())[0]
    unknown = work.__class__(work.request_id, "missing", work.reason, work.priority, 1, work.constraints)
    with pytest.raises(KeyError, match="unknown source strategy"):
        spawn_research_candidates(unknown, population)
    zero = work.__class__(work.request_id, population[0].strategy_id, work.reason, work.priority, 0)
    with pytest.raises(ValueError, match="positive budget"):
        spawn_research_candidates(zero, population)


def test_spawn_rejects_unanchored_research() -> None:
    population = tuple(seed_population([_strategy()]))
    work = materialize_research_work(_plan())[0]
    unanchored = work.__class__(work.request_id, None, "diversification", work.priority, 1)
    with pytest.raises(ValueError, match="explicit source"):
        spawn_research_candidates(unanchored, population)
