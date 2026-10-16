import pandas as pd

from atsf.paper import PaperConfig
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


def strategy(max_drawdown: float | None = None) -> StrategySpec:
    return StrategySpec(
        name="paper-e2e",
        universe=["TEST"],
        indicators=[Indicator(name="sma", source="close", period=3)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5, max_drawdown=max_drawdown),
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


def test_paper_entry_fills_on_next_close_not_signal_close():
    index = pd.date_range("2025-01-01", periods=6, freq="D")
    data = pd.DataFrame({"close": [100, 100, 100, 110, 120, 130]}, index=index)
    result = run_paper_strategy(
        data,
        strategy(),
        config=PaperConfig(commission_bps=0, slippage_bps=0),
    )

    # The first true entry decision occurs at index[3], and fills at index[4].
    assert result.fills[0].side == "buy"
    assert result.fills[0].timestamp == index[4]
    assert result.fills[-1].side == "sell"
    assert result.fills[-1].timestamp == index[-1]


def test_final_bar_signal_does_not_create_same_bar_paper_fill():
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    data = pd.DataFrame({"close": [100, 100, 100, 110]}, index=index)
    result = run_paper_strategy(
        data,
        strategy(),
        config=PaperConfig(commission_bps=0, slippage_bps=0),
    )

    assert result.fills == ()


def test_risk_halt_liquidates_open_position_at_current_close():
    index = pd.date_range("2025-01-01", periods=6, freq="D")
    data = pd.DataFrame({"close": [100, 100, 100, 110, 120, 70]}, index=index)
    result = run_paper_strategy(
        data,
        strategy(max_drawdown=0.10),
        config=PaperConfig(commission_bps=0, slippage_bps=0),
    )

    assert result.halted
    assert result.fills[-1].side == "sell"
    assert result.fills[-1].timestamp == index[-1]
    assert result.final_equity == result.snapshots[-1].cash
