import pandas as pd
import pytest

from atsf.lifecycle import (
    StrategyLifecycle,
    StrategyLifecycleStage,
    StrategyState,
    can_promote_stage,
    transition_promotion_stage,
)
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


def test_promotion_lifecycle_allows_only_explicit_forward_or_safety_transitions():
    assert can_promote_stage(StrategyLifecycleStage.RESEARCH, StrategyLifecycleStage.VALIDATED)
    assert can_promote_stage(StrategyLifecycleStage.PROMOTED, StrategyLifecycleStage.PAPER)
    assert can_promote_stage(StrategyLifecycleStage.PAPER, StrategyLifecycleStage.REPLACED)
    assert not can_promote_stage(StrategyLifecycleStage.RESEARCH, StrategyLifecycleStage.PAPER)
    assert not can_promote_stage(StrategyLifecycleStage.RETIRED, StrategyLifecycleStage.RESEARCH)

    event = transition_promotion_stage(
        "strategy-1",
        StrategyLifecycleStage.PROMOTED,
        StrategyLifecycleStage.PAPER,
        reason="passed paper admission gate",
    )
    assert event.current == StrategyLifecycleStage.PAPER


def test_promotion_lifecycle_rejects_invalid_or_empty_transition_metadata():
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        transition_promotion_stage(
            "strategy-1",
            StrategyLifecycleStage.RESEARCH,
            StrategyLifecycleStage.PAPER,
            reason="skip gates",
        )
    with pytest.raises(ValueError, match="strategy_id"):
        transition_promotion_stage(
            "",
            StrategyLifecycleStage.RESEARCH,
            StrategyLifecycleStage.VALIDATED,
            reason="valid",
        )
    with pytest.raises(ValueError, match="reason"):
        transition_promotion_stage(
            "strategy-1",
            StrategyLifecycleStage.RESEARCH,
            StrategyLifecycleStage.VALIDATED,
            reason="",
        )
