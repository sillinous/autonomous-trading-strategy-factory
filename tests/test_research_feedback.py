from types import SimpleNamespace

import pytest

from atsf.research_feedback import build_research_feedback
from atsf.research_queue import ResearchReason


def _evaluation(candidate_id="s1", *, valid=True, eligible=True, simulations=100, pass_rate=0.95, fitness=0.7, robust=True):
    return SimpleNamespace(
        candidate_id=candidate_id,
        validation_passed=valid,
        promotion=SimpleNamespace(eligible=eligible),
        monte_carlo=SimpleNamespace(simulations=simulations, pass_rate=pass_rate),
        fitness=SimpleNamespace(score=fitness),
        robustness=SimpleNamespace(passed=robust),
    )


def test_feedback_maps_failed_candidates_to_degraded():
    result = build_research_feedback([_evaluation(valid=False, eligible=False)], generation=4)
    assert result.generation == 4
    assert result.signals[0].reason == ResearchReason.DEGRADED


def test_feedback_maps_no_monte_carlo_to_halted():
    result = build_research_feedback([_evaluation(simulations=0, pass_rate=0.0)], generation=4)
    assert result.signals[0].reason == ResearchReason.HALTED
    assert result.signals[0].uncertainty == 1.0


def test_feedback_maps_eligible_candidate_to_capacity():
    result = build_research_feedback([_evaluation()], generation=4)
    signal = result.signals[0]
    assert signal.reason == ResearchReason.CAPACITY
    assert signal.capacity_gap == 1.0
    assert signal.robustness == 1.0


def test_feedback_is_deterministic_and_finite():
    evaluations = [_evaluation("b", fitness=float("nan")), _evaluation("a", pass_rate=float("inf"))]
    first = build_research_feedback(evaluations, generation=2)
    second = build_research_feedback(tuple(reversed(evaluations)), generation=2)
    assert first == second
    assert [signal.strategy_id for signal in first.signals] == ["a", "b"]
    for signal in first.signals:
        assert all(value >= 0 and value <= 1 for value in (
            signal.fitness,
            signal.robustness,
            signal.novelty,
            signal.uncertainty,
            signal.capacity_gap,
        ))


def test_feedback_rejects_invalid_generation():
    with pytest.raises(ValueError, match="generation"):
        build_research_feedback([], generation=-1)
