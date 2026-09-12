import pytest

from atsf.research_director import ResearchDirectorPolicy, ResearchSignal, prioritize
from atsf.research_queue import ResearchReason


def test_prioritize_prefers_high_information_research():
    signals = [
        ResearchSignal("stable", ResearchReason.DIVERSIFICATION, fitness=0.9, robustness=0.9, novelty=0.1, uncertainty=0.1),
        ResearchSignal("unknown", ResearchReason.DEGRADED, fitness=0.2, robustness=0.2, novelty=0.9, uncertainty=0.9),
    ]
    requests = prioritize(signals)
    assert requests[0].source_strategy_id == "unknown"
    assert requests[0].priority < requests[1].priority


def test_prioritize_is_deterministic_and_bounded():
    signals = [
        ResearchSignal("b", ResearchReason.CAPACITY, capacity_gap=1.0),
        ResearchSignal("a", ResearchReason.CAPACITY, capacity_gap=1.0),
        ResearchSignal(None, ResearchReason.DIVERSIFICATION, novelty=1.0),
    ]
    policy = ResearchDirectorPolicy(max_requests=2)
    first = prioritize(signals, policy=policy)
    second = prioritize(signals, policy=policy)
    assert first == second
    assert len(first) == 2
    assert all(0 <= request.priority <= 1000 for request in first)


def test_duplicate_signal_is_deduplicated():
    signals = [
        ResearchSignal("s1", ResearchReason.DEGRADED, uncertainty=0.4),
        ResearchSignal("s1", ResearchReason.DEGRADED, uncertainty=0.9),
    ]
    requests = prioritize(signals)
    assert len(requests) == 1
    assert requests[0].priority < 1000


def test_invalid_policy_and_signal_are_rejected():
    with pytest.raises(ValueError, match="must not all be zero"):
        ResearchDirectorPolicy(0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="signal values must be finite"):
        prioritize([ResearchSignal("s1", ResearchReason.HALTED, uncertainty=float("nan"))])
