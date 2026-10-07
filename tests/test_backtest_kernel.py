"""Regression tests for the backtest kernel's execution, sizing, risk, and ledger contract."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from atsf.backtest import BacktestConfig, run_long_signal_backtest, target_fraction
from atsf.evaluation import evaluate_walk_forward
from atsf.signals import position_state, strategy_position
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)

FREE = BacktestConfig(commission_bps=0.0, slippage_bps=0.0)
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "momentum.yaml"


def bars(close, open_=None, low=None) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(close), freq="B")
    close = np.asarray(close, dtype=float)
    frame = pd.DataFrame({"close": close}, index=index)
    frame["open"] = close if open_ is None else np.asarray(open_, dtype=float)
    frame["low"] = np.minimum(frame["open"], close) if low is None else np.asarray(low, dtype=float)
    frame["high"] = np.maximum(frame["open"], close)
    return frame


def strategy(method="fixed_fraction", value=1.0, max_position=1.0, stop=None, kill=None,
             universe=("TEST",)) -> StrategySpec:
    return StrategySpec(
        name="kernel",
        universe=list(universe),
        indicators=[Indicator(name="sma", period=2)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=0)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=0)]),
        position_sizing=PositionSizing(method=method, value=value, max_position=max_position),
        risk=RiskLimits(max_position=max_position, stop_loss=stop, max_drawdown=kill),
    )


def random_walk(n=2520, seed=0, drift=0.0004, vol=0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, vol / 3, n))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    frame = bars(close, open_, low)
    frame.index = pd.bdate_range("2014-01-02", periods=n)
    return frame


# --- execution timing -----------------------------------------------------------


def test_decision_fills_on_next_bar_open_not_signal_bar():
    data = bars([100, 100, 120, 120], open_=[100, 100, 110, 120])
    signal = pd.Series([False, True, True, True], index=data.index)
    result = run_long_signal_backtest(data, signal, strategy(), FREE)
    buy = result.trades.iloc[0]
    assert buy["timestamp"] == data.index[2]
    assert buy["price"] == pytest.approx(110.0)  # next open, not the 100 signal-bar close
    assert result.total_return == pytest.approx(120 / 110 - 1)


def test_signal_on_final_bar_never_fills():
    data = bars([100, 101, 102])
    signal = pd.Series([False, False, True], index=data.index)
    result = run_long_signal_backtest(data, signal, strategy(), FREE)
    assert result.trades.empty and result.total_return == 0.0


def test_falls_back_to_next_close_without_open_column():
    data = bars([100, 100, 120, 130]).drop(columns=["open", "low", "high"])
    signal = pd.Series([False, True, True, True], index=data.index)
    result = run_long_signal_backtest(data, signal, strategy(), FREE)
    assert result.trades.iloc[0]["price"] == pytest.approx(120.0)


def test_slippage_and_commission_are_adverse():
    data = bars([100, 100, 100, 100])
    signal = pd.Series([False, True, False, False], index=data.index)
    cfg = BacktestConfig(commission_bps=10, slippage_bps=10)
    result = run_long_signal_backtest(data, signal, strategy(), cfg)
    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy["price"] > 100 > sell["price"]
    assert result.total_return < 0


# --- ledger reconciliation -------------------------------------------------------


def test_ledger_agrees_with_equity_curve_on_single_trade():
    """Regression: the old kernel reported +5.1% equity but -2.3% for the same trade."""
    data = bars([100, 110, 121, 100, 100, 100, 100, 100])
    signal = pd.Series([True, True, False, False, False, False, False, False], index=data.index)
    result = run_long_signal_backtest(data, signal, strategy(), FREE)
    assert len(result.trade_returns) == 1
    assert result.trade_returns[0] == pytest.approx(result.total_return)


@pytest.mark.parametrize("seed", range(5))
def test_compounded_trade_returns_equal_total_return(seed):
    data = random_walk(800, seed)
    rng = np.random.default_rng(seed + 100)
    signal = pd.Series(rng.random(len(data)) > 0.5, index=data.index)
    spec = strategy("fixed_fraction", 0.6, stop=0.03, kill=0.25)
    result = run_long_signal_backtest(data, signal, spec, BacktestConfig())
    compounded = math.prod(1 + r for r in result.trade_returns)
    assert compounded == pytest.approx(1 + result.total_return, rel=1e-9)
    assert result.round_trips["pnl"].sum() == pytest.approx(
        result.equity.iloc[-1] - 100_000, abs=1e-6
    )


def test_open_position_is_marked_not_charged():
    data = bars([100, 100, 110])
    signal = pd.Series([True, True, True], index=data.index)
    result = run_long_signal_backtest(data, signal, strategy(), FREE)
    assert result.round_trips.iloc[-1]["exit_reason"] == "open"
    assert (result.trades["action"] == "sell").sum() == 0


# --- sizing ------------------------------------------------------------------------


def test_fixed_fraction_respects_value_and_ceiling():
    closes = np.full(30, 100.0)
    assert target_fraction(strategy("fixed_fraction", 0.3), closes, 25, FREE) == 0.3
    capped = strategy("fixed_fraction", 0.9, max_position=0.5)
    assert target_fraction(capped, closes, 25, FREE) == 0.5


def test_equal_weight_divides_by_universe():
    spec = strategy("equal_weight", 1.0, universe=("A", "B", "C", "D"))
    assert target_fraction(spec, np.full(30, 100.0), 25, FREE) == 0.25


def test_volatility_target_scales_inversely_with_realized_vol():
    rng = np.random.default_rng(1)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 60)))
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
    spec = strategy("volatility_target", 0.10)
    f_calm = target_fraction(spec, calm, 59, FREE)
    f_wild = target_fraction(spec, wild, 59, FREE)
    assert 0 < f_wild < f_calm <= 1.0
    assert f_wild == pytest.approx(0.10 / (np.diff(np.log(wild[39:])).std(ddof=1) * math.sqrt(252)))


def test_volatility_target_waits_for_lookback():
    assert target_fraction(strategy("volatility_target", 0.1), np.full(30, 100.0), 5, FREE) == 0.0


def test_volatility_target_uses_only_past_data():
    rng = np.random.default_rng(2)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
    spec = strategy("volatility_target", 0.1)
    before = target_fraction(spec, closes, 50, FREE)
    shocked = closes.copy()
    shocked[51:] *= np.exp(np.cumsum(rng.normal(0, 0.2, 29)))
    assert target_fraction(spec, shocked, 50, FREE) == before


def test_partial_position_scales_pnl():
    data = bars([100, 100, 100, 120])
    signal = pd.Series([True, True, True, True], index=data.index)
    result = run_long_signal_backtest(data, signal, strategy("fixed_fraction", 0.25, 0.25), FREE)
    assert result.total_return == pytest.approx(0.25 * 0.20)


# --- risk controls -------------------------------------------------------------------


def test_stop_loss_exits_at_stop_price_intrabar():
    data = bars([100, 100, 100, 100, 100], open_=[100, 100, 100, 99, 99],
                low=[100, 100, 100, 90, 99])
    signal = pd.Series(True, index=data.index)
    result = run_long_signal_backtest(data, signal, strategy(stop=0.05), FREE)
    stop = result.round_trips.iloc[0]
    assert stop["exit_reason"] == "stop_loss"
    assert stop["exit_price"] == pytest.approx(95.0)


def test_stop_loss_gap_fills_at_open():
    data = bars([100, 100, 100, 80], open_=[100, 100, 100, 85], low=[100, 100, 100, 80])
    signal = pd.Series(True, index=data.index)
    result = run_long_signal_backtest(data, signal, strategy(stop=0.05), FREE)
    assert result.round_trips.iloc[0]["exit_price"] == pytest.approx(85.0)


def test_stop_requires_signal_reset_before_reentry():
    data = bars([100, 100, 100, 100, 100, 100, 100], low=[100, 100, 100, 90, 100, 100, 100])
    signal = pd.Series([True, True, True, True, False, True, True], index=data.index)
    result = run_long_signal_backtest(data, signal, strategy(stop=0.05), FREE)
    buys = result.trades[result.trades["action"] == "buy"]["timestamp"].tolist()
    assert buys == [data.index[1], data.index[6]]


def test_max_drawdown_kill_switch_flattens_and_halts():
    close = [100, 100, 100, 80, 80, 120, 140]
    data = bars(close)
    signal = pd.Series(True, index=data.index)
    result = run_long_signal_backtest(data, signal, strategy(kill=0.10), FREE)
    assert result.halted_at == data.index[3]
    assert result.round_trips.iloc[0]["exit_reason"] == "max_drawdown"
    assert (result.trades["action"] == "buy").sum() == 1
    assert result.equity.iloc[-1] == result.equity.iloc[4]


# --- signal state & walk-forward ----------------------------------------------------------


def test_position_state_applies_exit_before_entry():
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    entry = pd.Series([True, False, False, True], index=index)
    exit_ = pd.Series([False, False, True, False], index=index)
    assert position_state(entry, exit_).tolist() == [True, True, False, True]


def test_example_strategy_trades_out_of_sample():
    """Regression: unwarmed segments starved SMA-200 and produced zero OOS trades."""
    spec = StrategySpec(**yaml.safe_load(EXAMPLE.read_text()))
    data = random_walk()
    wf = evaluate_walk_forward(data, spec, 504, 126, 126, step_size=126)
    test_trades = [len(seg.backtest.trades) for seg in wf.windows[2::3]]
    assert sum(test_trades) > 0
    assert sum(1 for n in test_trades if n > 0) >= len(test_trades) // 2


def test_walk_forward_signals_match_full_history_without_lookahead():
    spec = StrategySpec(**yaml.safe_load(EXAMPLE.read_text()))
    data = random_walk()
    full = strategy_position(data, spec)
    truncated = strategy_position(data.iloc[:1500], spec)
    assert full.iloc[:1500].equals(truncated)


def test_full_cash_entry_never_trips_cash_check():
    """Regression: a 100% entry on six-figure cash overran an absolute 1e-12 cash check."""
    from atsf.paper import PaperBroker, PaperConfig

    cash, price = 909127.3192399365, 130.88147259022466  # found by search; failed before fix
    broker = PaperBroker(PaperConfig(cash, 1.0, 2.0))
    unit_cost = price * (1 + 2 / 1e4) * (1 + 1 / 1e4)
    quantity = min(cash / unit_cost, broker.max_affordable_quantity(price))
    broker.execute(pd.Timestamp("2024-01-02"), "buy", quantity, price)
    assert 0.0 <= broker.cash < 1e-6


def _crash_then_trend() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    crash = np.linspace(0, np.log(0.5), 120)                     # -50% in the train window
    calm = np.cumsum(rng.normal(0.0008, 0.004, 480))              # steady recovery after
    log_close = np.r_[crash, crash[-1] + calm]
    return bars(100 * np.exp(log_close))


def test_train_segment_is_diagnostic_by_default():
    data = _crash_then_trend()
    always_long = strategy()
    default = evaluate_walk_forward(data, always_long, 300, 150, 150)
    strict = evaluate_walk_forward(data, always_long, 300, 150, 150, require_train_pass=True)
    assert default.train_passed is False
    assert default.passed is True
    assert strict.passed is False
