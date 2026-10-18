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
    index = pd.date_range("2025-01-01", periods=6)
    data = {
        "a": pd.DataFrame({"close": [100, 100, 110, 120, 50, 50]}, index=index),
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


def test_portfolio_signals_fill_at_next_close_and_not_on_first_bar():
    index = pd.date_range("2025-01-01", periods=6)
    data = {"a": pd.DataFrame({"close": [100, 100, 110, 120, 90, 90]}, index=index)}
    result = run_paper_portfolio(
        data, {"a": make_strategy("a")}, {"a": 1.0}, decisions={"a": approved()},
        commission_bps=0, slippage_bps=0,
    )
    buys = [fill for _, fill in result.fills if fill.side == "buy"]
    sells = [fill for _, fill in result.fills if fill.side == "sell"]
    assert buys
    assert buys[0].timestamp == index[3]  # Entry signal at index[2].
    assert sells[0].timestamp == index[5]  # Exit signal at index[4] fills next close.


def test_final_bar_entry_signal_does_not_create_same_window_fill():
    index = pd.date_range("2025-01-01", periods=3)
    data = {"a": pd.DataFrame({"close": [100, 100, 110]}, index=index)}
    result = run_paper_portfolio(
        data, {"a": make_strategy("a")}, {"a": 1.0}, decisions={"a": approved()},
        commission_bps=0, slippage_bps=0,
    )
    assert result.fills == ()


def test_zero_weight_sleeve_is_ignored_without_creating_zero_cash_broker():
    index = pd.date_range("2025-01-01", periods=4)
    data = {
        "a": pd.DataFrame({"close": [100, 101, 102, 103]}, index=index),
        "b": pd.DataFrame({"close": [100, 101, 102, 103]}, index=index),
    }
    result = run_paper_portfolio(
        data,
        {"a": make_strategy("a"), "b": make_strategy("b")},
        {"a": 1.0, "b": 0.0},
        decisions={"a": approved(), "b": approved()},
        commission_bps=0,
        slippage_bps=0,
    )
    assert result.snapshots
    assert result.final_equity > 0
    assert all(strategy_id == "a" for strategy_id, _ in result.fills)


def test_portfolio_risk_breach_cancels_pending_entries_before_fill():
    index = pd.date_range("2025-01-01", periods=6)
    data = {
        # Sleeve a enters first, then its price gap breaches portfolio drawdown.
        "a": pd.DataFrame({"close": [100, 100, 110, 120, 50, 50]}, index=index),
        # Sleeve b queues an entry at index[3], due to fill at index[4].
        "b": pd.DataFrame({"close": [100, 100, 90, 120, 130, 140]}, index=index),
    }
    result = run_paper_portfolio(
        data,
        {"a": make_strategy("a"), "b": make_strategy("b")},
        {"a": 0.5, "b": 0.5},
        decisions={"a": approved(), "b": approved()},
        max_drawdown=0.10,
        commission_bps=0,
        slippage_bps=0,
    )

    assert result.halted
    assert result.halt_reason == "maximum drawdown breached"
    # b's queued entry must be cancelled, not bought and immediately liquidated.
    assert not any(strategy_id == "b" and fill.side == "buy" for strategy_id, fill in result.fills)
    assert all(fill.timestamp <= index[4] for _, fill in result.fills)
