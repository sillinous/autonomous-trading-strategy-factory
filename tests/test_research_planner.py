from types import SimpleNamespace

from atsf.research_planner import build_research_plan
from atsf.research_budget import ResearchBudgetPolicy
from atsf.research_director import ResearchDirectorPolicy
from atsf.research_queue import ResearchReason
from atsf.strategy import Comparator, Condition, Indicator, PositionSizing, Signal, StrategySpec


def strategy(period: int) -> StrategySpec:
    return StrategySpec(
        name=f"planner-{period}",
        universe=["TEST"],
        indicators=[Indicator(name="sma", period=period)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.1),
    )


def evaluation(candidate_id: str, eligible: bool, score: float, simulations: int = 100):
    return SimpleNamespace(
        candidate_id=candidate_id,
        validation_passed=eligible,
        fitness=SimpleNamespace(score=score),
        promotion=SimpleNamespace(eligible=eligible),
        robustness=SimpleNamespace(passed=eligible),
        monte_carlo=SimpleNamespace(pass_rate=0.9, simulations=simulations),
    )


def test_plan_is_prioritized_and_budgeted():
    a = SimpleNamespace(strategy_id="a", strategy=strategy(5))
    b = SimpleNamespace(strategy_id="b", strategy=strategy(20))
    result = SimpleNamespace(
        generation=4,
        evaluations=(evaluation("a", False, 0.2), evaluation("b", True, 1.0)),
        survivors=(a, b),
    )
    plan = build_research_plan(
        result,
        director_policy=ResearchDirectorPolicy(max_requests=2),
        budget_policy=ResearchBudgetPolicy(total_units=20, min_units=1, max_units_per_request=20),
    )
    assert plan.generation == 4
    assert len(plan.requests) == 2
    assert len(plan.allocations) == 2
    assert sum(item.units for item in plan.allocations) == 20
    assert {request.reason for request in plan.requests} >= {ResearchReason.DEGRADED}
    assert plan.requests[0].reason in {
        ResearchReason.DEGRADED,
        ResearchReason.CAPACITY,
        ResearchReason.DIVERSIFICATION,
    }


def test_plan_is_deterministic():
    candidate = SimpleNamespace(strategy_id="a", strategy=strategy(5))
    result = SimpleNamespace(
        generation=1,
        evaluations=(evaluation("a", False, 0.3, simulations=0),),
        survivors=(candidate,),
    )
    first = build_research_plan(result)
    second = build_research_plan(result)
    assert first == second
    assert first.requests[0].reason is ResearchReason.DEGRADED


def test_missing_candidate_creates_degraded_signal():
    result = SimpleNamespace(
        generation=2,
        evaluations=(evaluation("missing", False, 0.0),),
        survivors=(),
    )
    plan = build_research_plan(result)
    assert plan.requests[0].reason is ResearchReason.DEGRADED
