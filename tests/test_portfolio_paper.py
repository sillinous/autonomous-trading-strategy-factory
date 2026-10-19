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


def test_portfolio_paper_runner_rejects_missing_timestamps():
    index = pd.DatetimeIndex(["2025-01-01", None, "2025-01-03"])
    data = pd.DataFrame({"close": [100, 101, 102]}, index=index)
    with pytest.raises(ValueError, match="missing timestamps"):
        run_paper_portfolio(
            {"a": data},
            {"a": make_strategy("a")},
            {"a": 1.0},
            decisions={"a": approved()},
        )


def test_portfolio_paper_runner_normalizes_numeric_string_closes():
    index = pd.date_range("2025-01-01", periods=6)
    data = {
        "a": pd.DataFrame({"close": ["100", "100", "110", "120", "90", "90"]}, index=index)
    }
    result = run_paper_portfolio(
        data,
        {"a": make_strategy("a")},
        {"a": 1.0},
        decisions={"a": approved()},
        commission_bps=0,
        slippage_bps=0,
    )
    assert result.fills
    assert all(isinstance(fill.price, float) for _, fill in result.fills)


def test_portfolio_paper_runner_rejects_non_finite_indicator_source():
    index = pd.date_range("2025-01-01", periods=4)
    data = {
        "a": pd.DataFrame(
            {"close": [100, 101, 102, 103], "volume": [10, 20, float("inf"), 40]},
            index=index,
        )
    }
    strategy = make_strategy("a").model_copy(
        update={"indicators": [Indicator(name="sma", source="volume", period=2)]}
    )
    with pytest.raises(ValueError, match="signal source column.*finite numeric values"):
        run_paper_portfolio(
            data, {"a": strategy}, {"a": 1.0}, decisions={"a": approved()}
        )


def test_portfolio_paper_runner_rejects_missing_indicator_source():
    index = pd.date_range("2025-01-01", periods=4)
    data = {"a": pd.DataFrame({"close": [100, 101, 102, 103]}, index=index)}
    strategy = make_strategy("a").model_copy(
        update={"indicators": [Indicator(name="sma", source="volume", period=2)]}
    )
    with pytest.raises(ValueError, match="missing signal source column.*volume"):
        run_paper_portfolio(
            data, {"a": strategy}, {"a": 1.0}, decisions={"a": approved()}
        )


def test_portfolio_paper_runner_rejects_timezone_aware_timestamps():
    index = pd.date_range("2025-01-01", periods=4, tz="UTC")
    data = pd.DataFrame({"close": [100, 101, 102, 103]}, index=index)
    with pytest.raises(ValueError, match="timezone-naive"):
        run_paper_portfolio(
            {"a": data},
            {"a": make_strategy("a")},
            {"a": 1.0},
            decisions={"a": approved()},
        )



def test_portfolio_paper_runner_rejects_live_stage_for_paper_execution():
    index = pd.date_range("2025-01-01", periods=6)
    data = {"a": pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]}, index=index)}
    with pytest.raises(PermissionError, match="not eligible"):
        run_paper_portfolio(
            data,
            {"a": make_strategy("a")},
            {"a": 1.0},
            decisions={"a": PromotionDecision(stage="live", eligible=True, reasons=())},
        )


@pytest.mark.parametrize(
    "decision",
    [
        PromotionDecision(stage="paper", eligible=True, reasons=("failed robustness",)),
        PromotionDecision(stage="paper", eligible="false", reasons=()),
        PromotionDecision(stage="paper", eligible=True, reasons=["bad-shape"]),
        PromotionDecision(stage="paper", eligible=True, reasons=(False,)),
    ],
)
def test_portfolio_paper_runner_rejects_malformed_eligible_decisions(decision):
    index = pd.date_range("2025-01-01", periods=6)
    data = {"a": pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]}, index=index)}
    with pytest.raises(PermissionError, match="not eligible"):
        run_paper_portfolio(
            data,
            {"a": make_strategy("a")},
            {"a": 1.0},
            decisions={"a": decision},
        )


def test_portfolio_paper_runner_rejects_unhashable_stage_value():
    index = pd.date_range("2025-01-01", periods=6)
    data = {"a": pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]}, index=index)}
    with pytest.raises(PermissionError, match="not eligible"):
        run_paper_portfolio(
            data,
            {"a": make_strategy("a")},
            {"a": 1.0},
            decisions={"a": PromotionDecision(stage=["paper"], eligible=True, reasons=())},
        )


@pytest.mark.parametrize("bad_initial_cash", [True, "100000", float("nan"), float("inf"), 0])
def test_portfolio_paper_runner_rejects_invalid_initial_cash(bad_initial_cash):
    index = pd.date_range("2025-01-01", periods=3)
    data = {"a": pd.DataFrame({"close": [100, 101, 102]}, index=index)}
    with pytest.raises(ValueError, match="initial_cash"):
        run_paper_portfolio(
            data, {"a": make_strategy("a")}, {"a": 1.0},
            decisions={"a": approved()}, initial_cash=bad_initial_cash,
        )


@pytest.mark.parametrize("bad_weight", [True, "0.5", float("nan"), float("inf"), -0.1])
def test_portfolio_paper_runner_rejects_invalid_weight_types_and_values(bad_weight):
    index = pd.date_range("2025-01-01", periods=3)
    data = {"a": pd.DataFrame({"close": [100, 101, 102]}, index=index)}
    with pytest.raises(ValueError, match="weights"):
        run_paper_portfolio(
            data, {"a": make_strategy("a")}, {"a": bad_weight},
            decisions={"a": approved()},
        )


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("commission_bps", True),
        ("commission_bps", "1.0"),
        ("commission_bps", float("nan")),
        ("commission_bps", -1.0),
        ("slippage_bps", True),
        ("slippage_bps", "2.0"),
        ("slippage_bps", float("inf")),
        ("slippage_bps", -1.0),
    ],
)
def test_portfolio_paper_runner_rejects_invalid_execution_costs(field, bad_value):
    index = pd.date_range("2025-01-01", periods=3)
    data = {"a": pd.DataFrame({"close": [100, 101, 102]}, index=index)}
    kwargs = {field: bad_value}
    with pytest.raises(ValueError, match=field):
        run_paper_portfolio(
            data, {"a": make_strategy("a")}, {"a": 1.0},
            decisions={"a": approved()}, **kwargs,
        )


@pytest.mark.parametrize("bad_drawdown", [True, "0.1", float("nan"), float("inf"), 0, 1])
def test_portfolio_paper_runner_rejects_invalid_drawdown_types_and_values(bad_drawdown):
    index = pd.date_range("2025-01-01", periods=3)
    data = {"a": pd.DataFrame({"close": [100, 101, 102]}, index=index)}
    with pytest.raises(ValueError, match="max_drawdown"):
        run_paper_portfolio(
            data, {"a": make_strategy("a")}, {"a": 1.0},
            decisions={"a": approved()}, max_drawdown=bad_drawdown,
        )


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("data", None),
        ("strategies", None),
        ("weights", None),
        ("decisions", None),
    ],
)
def test_portfolio_paper_runner_rejects_non_dictionary_inputs(field, bad_value):
    index = pd.date_range("2025-01-01", periods=3)
    data = {"a": pd.DataFrame({"close": [100, 101, 102]}, index=index)}
    strategies = {"a": make_strategy("a")}
    weights = {"a": 1.0}
    decisions = {"a": approved()}
    values = {
        "data": data,
        "strategies": strategies,
        "weights": weights,
        "decisions": decisions,
    }
    values[field] = bad_value
    with pytest.raises(ValueError, match="must be dictionaries"):
        run_paper_portfolio(**values)


def test_portfolio_paper_runner_rejects_non_dataframe_market_data():
    with pytest.raises(ValueError, match="DataFrames"):
        run_paper_portfolio(
            {"a": [100, 101, 102]},
            {"a": make_strategy("a")},
            {"a": 1.0},
            decisions={"a": approved()},
        )


def test_portfolio_paper_runner_rejects_non_strategy_values():
    index = pd.date_range("2025-01-01", periods=3)
    data = {"a": pd.DataFrame({"close": [100, 101, 102]}, index=index)}
    with pytest.raises(ValueError, match="StrategySpec"):
        run_paper_portfolio(
            data,
            {"a": object()},
            {"a": 1.0},
            decisions={"a": approved()},
        )


def test_portfolio_paper_runner_rejects_blank_strategy_ids():
    index = pd.date_range("2025-01-01", periods=3)
    data = {" ": pd.DataFrame({"close": [100, 101, 102]}, index=index)}
    with pytest.raises(ValueError, match="non-empty strings"):
        run_paper_portfolio(
            data,
            {" ": make_strategy("blank")},
            {" ": 1.0},
            decisions={" ": approved()},
        )
