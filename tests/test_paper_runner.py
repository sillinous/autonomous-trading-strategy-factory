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


def strategy() -> StrategySpec:
    return StrategySpec(
        name="paper-e2e",
        universe=["TEST"],
        indicators=[Indicator(name="sma", source="close", period=3)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def test_strategy_runs_end_to_end_in_paper_mode():
    close = [100, 100, 100, 110, 90, 90, 120]
    data = pd.DataFrame({"close": close}, index=pd.date_range("2025-01-01", periods=len(close)))
    result = run_paper_strategy(data, strategy())
    assert result.snapshots
    assert result.final_equity > 0
    assert len(result.fills) >= 2
    assert result.fills[0].side == "buy"
    assert result.fills[-1].side == "sell"
