import pandas as pd

from atsf.paper_runner import run_paper_strategy
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)


def make_strategy(max_drawdown=None):
    return StrategySpec(
        name="paper-risk-e2e",
        universe=["TEST"],
        indicators=[Indicator(name="sma", source="close", period=2)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5, max_drawdown=max_drawdown),
    )


def test_paper_runner_halts_when_drawdown_limit_breaches():
    data = pd.DataFrame(
        {"close": [100, 100, 120, 70, 60]},
        index=pd.date_range("2025-01-01", periods=5),
    )
    result = run_paper_strategy(data, make_strategy(max_drawdown=0.10))
    assert result.halted
    assert result.halt_reason == "maximum drawdown breached"
