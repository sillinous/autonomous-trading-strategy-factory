import pandas as pd

from atsf.feedback_loop import process_strategy_health
from atsf.lifecycle import StrategyLifecycle, StrategyState
from atsf.monitoring import DegradationPolicy, assess_degradation
from atsf.research_queue import ResearchQueue, ResearchReason


def test_degradation_enqueues_replacement_work():
    equity = pd.Series([100.0] * 10 + [75.0] * 10, index=pd.date_range("2025-01-01", periods=20))
    report = assess_degradation(
        equity,
        DegradationPolicy(max_drawdown=0.10, min_return=-1.0, max_volatility=10.0),
    )
    lifecycle = StrategyLifecycle()
    queue = ResearchQueue()
    action = process_strategy_health("strategy-1", lifecycle, report, queue)
    assert action.state == StrategyState.DEGRADED
    assert action.research_request is not None
    assert action.research_request.reason == ResearchReason.DEGRADED
    assert len(queue) == 1


def test_healthy_strategy_does_not_create_replacement():
    equity = pd.Series([100.0 + i for i in range(20)], index=pd.date_range("2025-01-01", periods=20))
    report = assess_degradation(equity, DegradationPolicy(max_drawdown=0.20, min_return=-1.0, max_volatility=10.0))
    lifecycle = StrategyLifecycle()
    queue = ResearchQueue()
    action = process_strategy_health("strategy-2", lifecycle, report, queue)
    assert action.state == StrategyState.ACTIVE
    assert action.research_request is None
    assert len(queue) == 0
