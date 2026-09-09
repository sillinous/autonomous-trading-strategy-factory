import pandas as pd
import pytest

from atsf.portfolio_paper import run_paper_portfolio
from atsf.promotion import PromotionDecision
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)


def make_strategy(name: str) -> StrategySpec:
    return StrategySpec(
        name=name,
        universe=["TEST"],
        indicators=[Indicator(name="sma", source="close", period=2)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def approved() -> PromotionDecision:
    return PromotionDecision(stage="paper", eligible=True, reasons=())


def test_portfolio_paper_runner_requires_promotion_gate():
    data = pd.DataFrame({"close": [100, 100, 110, 90]}, index=pd.date_range("2025-01-01", periods=4))
    with pytest.raises(PermissionError, match="not eligible"):
        run_paper_portfolio(
            {"a": data},
            {"a": make_strategy("a")},
            {"a": 1.0},
            decisions={"a": PromotionDecision(stage="research", eligible=False, reasons=("failed",))},
        )


def test_portfolio_paper_runner_aggregates_isolated_strategy_brokers():
    index = pd.date_range("2025-01-01", periods=6)
    data = {"a": pd.DataFrame({"close": [100, 100, 110, 90, 90, 120]}, index=index)}
    result = run_paper_portfolio(
        data,
        {"a": make_strategy("a")},
        {"a": 0.5},
        decisions={"a": approved()},
    )
    assert result.snapshots
    assert result.final_equity > 0
    assert all(strategy_id == "a" for strategy_id, _ in result.fills)


def test_portfolio_paper_runner_rejects_excess_exposure():
    data = pd.DataFrame({"close": [100, 101, 102]}, index=pd.date_range("2025-01-01", periods=3))
    with pytest.raises(ValueError, match="100% gross exposure"):
        run_paper_portfolio(
            {"a": data, "b": data},
            {"a": make_strategy("a"), "b": make_strategy("b")},
            {"a": 0.75, "b": 0.75},
            decisions={"a": approved(), "b": approved()},
        )


def test_portfolio_paper_runner_rejects_misaligned_timestamps():
    first = pd.date_range("2025-01-01", periods=4)
    second = pd.date_range("2025-01-01", periods=4, freq="2D")
    with pytest.raises(ValueError, match="same timestamps"):
        run_paper_portfolio(
            {
                "a": pd.DataFrame({"close": [100, 101, 102, 103]}, index=first),
                "b": pd.DataFrame({"close": [100, 101, 102, 103]}, index=second),
            },
            {"a": make_strategy("a"), "b": make_strategy("b")},
            {"a": 0.5, "b": 0.5},
            decisions={"a": approved(), "b": approved()},
        )


def test_portfolio_paper_runner_liquidates_on_risk_halt():
    index = pd.date_range("2025-01-01", periods=5)
    data = {
        "a": pd.DataFrame({"close": [100, 100, 100, 50, 50]}, index=index),
    }
    result = run_paper_portfolio(
        data,
        {"a": make_strategy("a")},
        {"a": 1.0},
        decisions={"a": approved()},
        max_drawdown=0.10,
    )
    assert result.halted
    assert result.halt_reason
    assert result.fills
    assert result.fills[-1][0] == "a"
    assert result.fills[-1][1].side == "sell"
    assert result.final_equity > 0
