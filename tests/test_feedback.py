import pandas as pd

from atsf.feedback import decide_feedback
from atsf.lifecycle import StrategyState
from atsf.monitoring import DegradationPolicy, assess_degradation


def test_feedback_retires_degraded_strategy():
    equity = pd.Series(
        [100.0] * 10 + [80.0] * 10,
        index=pd.date_range("2025-01-01", periods=20),
    )
    report = assess_degradation(
        equity,
        DegradationPolicy(max_drawdown=0.10, min_return=-1.0, max_volatility=10.0),
    )
    decision = decide_feedback(report)
    assert decision.state == StrategyState.DEGRADED
    assert decision.action == "retire_from_paper"


def test_feedback_keeps_healthy_strategy_active():
    equity = pd.Series(
        [100.0 + i for i in range(20)],
        index=pd.date_range("2025-01-01", periods=20),
    )
    report = assess_degradation(
        equity,
        DegradationPolicy(max_drawdown=0.20, min_return=-1.0, max_volatility=10.0),
    )
    decision = decide_feedback(report)
    assert decision.state == StrategyState.ACTIVE
    assert decision.action == "remain_in_paper"
