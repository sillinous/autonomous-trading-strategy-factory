import pandas as pd
import pytest

from atsf.compiler import compile_strategy
from atsf.strategy import Comparator, Condition, PositionSizing, RiskLimits, Signal, Side, StrategySpec


def make_strategy(side=Side.LONG):
    return StrategySpec(
        name="compiler_demo",
        universe=["TEST"],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=2)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=1)]),
        side=side,
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def test_compiler_is_deterministic_and_exposes_manifest():
    first = compile_strategy(make_strategy())
    second = compile_strategy(make_strategy())
    assert first.strategy_id == second.strategy_id
    assert first.manifest() == second.manifest()


def test_compiler_uses_trusted_signal_evaluator():
    compiled = compile_strategy(make_strategy())
    data = pd.DataFrame(
        {"close": [1.0, 3.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    entry, exit_ = compiled.signals(data)
    assert entry.tolist() == [False, True, False]
    assert exit_.tolist() == [False, False, False]


def test_compiler_rejects_short_until_supported():
    with pytest.raises(NotImplementedError):
        compile_strategy(make_strategy(Side.SHORT))


def test_compiler_rejects_blank_version():
    with pytest.raises(ValueError):
        compile_strategy(make_strategy(), compiler_version="   ")
