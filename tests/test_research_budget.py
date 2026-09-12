import pytest

from atsf.research_budget import ResearchBudgetPolicy, allocate_budget
from atsf.research_director import ResearchReason, ResearchSignal


def signal(strategy_id: str, score: float) -> ResearchSignal:
    return ResearchSignal(
        strategy_id=strategy_id,
        reason=ResearchReason.DIVERSIFICATION,
        novelty=score,
    )


def test_budget_is_bounded_and_deterministic():
    signals = [signal("a", 1.0), signal("b", 2.0), signal("c", 3.0)]
    policy = ResearchBudgetPolicy(total_units=30, min_units=2, max_units_per_request=20)
    first = allocate_budget(signals, policy=policy)
    second = allocate_budget(signals, policy=policy)
    assert first == second
    assert sum(item.units for item in first) == 30
    assert all(2 <= item.units <= 20 for item in first)


def test_zero_scores_receive_equal_budget():
    result = allocate_budget(
        [signal("a", 0), signal("b", 0)],
        policy=ResearchBudgetPolicy(total_units=10, min_units=1, max_units_per_request=10),
    )
    assert [item.units for item in result] == [5, 5]


def test_mismatched_ids_fail_closed():
    with pytest.raises(ValueError, match="request_ids"):
        allocate_budget([signal("a", 1)], request_ids=[])


def test_negative_signal_fails_closed():
    with pytest.raises(ValueError, match="finite and non-negative"):
        allocate_budget([ResearchSignal("a", ResearchReason.CAPACITY, novelty=-1)])
