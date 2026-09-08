import pandas as pd

from atsf.lifecycle import StrategyLifecycle, StrategyState
from atsf.monitoring import DegradationPolicy, assess_degradation


def test_lifecycle_transitions_to_degraded():
    lifecycle = StrategyLifecycle()
    assert lifecycle.state == StrategyState.ACTIVE
    equity = pd.Series([100.0] * 10 + [80.0] * 10, index=pd.date_range("2025-01-01", periods=20))
    report = assess_degradation(
        equity,
        DegradationPolicy(max_drawdown=0.10, min_return=-1.0, max_volatility=10.0, min_observations=20),
    )
    assert lifecycle.apply(report) == StrategyState.DEGRADED
    assert not lifecycle.can_trade
    assert lifecycle.events[-1].previous == StrategyState.ACTIVE


def test_lifecycle_halt_is_terminal():
    lifecycle = StrategyLifecycle()
    assert lifecycle.halt("manual safety stop") == StrategyState.HALTED
    assert lifecycle.halt("another reason") == StrategyState.HALTED
    assert len(lifecycle.events) == 1
    assert not lifecycle.can_trade
