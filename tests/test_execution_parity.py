"""Backtest and every paper path must produce identical fills on identical data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atsf.backtest import BacktestConfig, run_long_signal_backtest
from atsf.execution import ExecutionPolicy, SleeveEngine
from atsf.paper import PaperBroker
from atsf.paper_runner import run_paper_strategy
from atsf.portfolio_paper import run_paper_portfolio
from atsf.promotion import PromotionDecision
from atsf.signals import strategy_position
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)


def market(seed: int, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.007, n)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.007, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                        index=pd.bdate_range("2020-01-01", periods=n))


def spec(sizing: str = "volatility_target", stop: float | None = 0.04,
         kill: float | None = None) -> StrategySpec:
    value = 0.15 if sizing == "volatility_target" else 0.6
    return StrategySpec(
        name="parity",
        universe=["TEST"],
        indicators=[Indicator(name="fast", kind="ema", period=10),
                    Indicator(name="slow", kind="sma", period=40)],
        entry=Signal(all=[Condition(left="fast", comparator=Comparator.CROSSES_ABOVE, right="slow")]),
        exit=Signal(any=[Condition(left="fast", comparator=Comparator.CROSSES_BELOW, right="slow")]),
        position_sizing=PositionSizing(method=sizing, value=value, max_position=0.8),
        risk=RiskLimits(max_position=0.8, stop_loss=stop, max_drawdown=kill),
    )


CASES = [("volatility_target", 0.04, None), ("fixed_fraction", None, None),
         ("fixed_fraction", 0.03, 0.12)]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("sizing,stop,kill", CASES)
def test_paper_runner_matches_backtest_exactly(seed, sizing, stop, kill):
    data, strategy = market(seed), spec(sizing, stop, kill)
    config = BacktestConfig(commission_bps=1, slippage_bps=2)
    backtest = run_long_signal_backtest(data, strategy_position(data, strategy), strategy, config)
    paper = run_paper_strategy(data, strategy, config=config.paper, liquidate_at_end=False)
    paper_equity = pd.Series([s.equity for s in paper.snapshots], index=data.index)
    pd.testing.assert_series_equal(paper_equity, backtest.equity, check_names=False, rtol=0, atol=0)
    assert [(f.timestamp, f.side, f.price) for f in paper.fills] == list(
        zip(backtest.trades["timestamp"], backtest.trades["action"], backtest.trades["price"]))


@pytest.mark.parametrize("seed", range(3))
def test_single_sleeve_portfolio_matches_backtest(seed):
    data, strategy = market(seed), spec()
    config = BacktestConfig(commission_bps=1, slippage_bps=2)
    backtest = run_long_signal_backtest(data, strategy_position(data, strategy), strategy, config)
    portfolio = run_paper_portfolio({"s": data}, {"s": strategy}, {"s": 1.0},
                                    decisions={"s": PromotionDecision("paper", True, ())},
                                    commission_bps=1, slippage_bps=2)
    equity = pd.Series([s.equity for s in portfolio.snapshots], index=data.index)
    pd.testing.assert_series_equal(equity.iloc[:-1], backtest.equity.iloc[:-1], check_names=False)
    buys = [e.timestamp for _, e in portfolio.events if e.action == "buy"]
    assert buys == list(backtest.trades.loc[backtest.trades["action"] == "buy", "timestamp"])


def test_engine_rejects_out_of_order_steps():
    data, strategy = market(0, 50), spec()
    engine = SleeveEngine(strategy, data, strategy_position(data, strategy), PaperBroker(),
                          ExecutionPolicy())
    engine.step(0)
    with pytest.raises(ValueError, match="in order"):
        engine.step(2)


def test_every_fill_has_a_prior_decision():
    data, strategy = market(1), spec()
    result = run_long_signal_backtest(data, strategy_position(data, strategy), strategy)
    signal_fills = result.trades[result.trades["reason"].isin(["entry_signal", "exit_signal"])]
    assert (signal_fills["decision_timestamp"] < signal_fills["timestamp"]).all()
