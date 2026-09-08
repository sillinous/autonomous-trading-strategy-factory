import pandas as pd
import pytest

from atsf.signals import compute_indicators, evaluate_signal, strategy_signals
from atsf.strategy import Comparator, Condition, Indicator, Signal, StrategySpec


def make_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=30)
    return pd.DataFrame({"close": range(1, 31)}, index=index)


def test_sma_is_deterministic_and_respects_warmup():
    data = make_data()
    values = compute_indicators(data, [Indicator(name="sma", period=5)])
    assert values["sma"].iloc[:4].isna().all()
    assert values["sma"].iloc[4] == 3
    assert values["sma"].iloc[-1] == 28


def test_cross_above_requires_previous_non_crossed_state():
    data = pd.DataFrame({"close": [1, 2, 3, 2, 4]}, index=pd.RangeIndex(5))
    indicators = {}
    signal = Signal(
        all=[Condition(left="close", comparator=Comparator.CROSSES_ABOVE, right=3)]
    )
    result = evaluate_signal(data, signal, indicators)
    assert result.tolist() == [False, False, False, False, True]


def test_strategy_signals_resolve_indicator_names():
    data = make_data()
    strategy = StrategySpec(
        name="demo",
        universe=["TEST"],
        indicators=[Indicator(name="sma", period=5)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing={"method": "fixed_fraction", "value": 0.5, "max_position": 0.5},
        risk={"max_position": 0.5},
    )
    entry, exit_ = strategy_signals(data, strategy)
    assert entry.iloc[4]
    assert not exit_.any()


def test_unknown_indicator_is_rejected():
    data = make_data()
    with pytest.raises(ValueError, match="unsupported indicator"):
        compute_indicators(data, [Indicator(name="macd", period=12)])
