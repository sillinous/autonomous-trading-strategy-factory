import pandas as pd
import pytest

import atsf.robustness as robustness
from atsf.backtest import BacktestResult
from atsf.generator import StrategyCandidate
from atsf.robustness import (
    monte_carlo_trade_bootstrap,
    regime_returns,
    score_regime_stability,
)
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)


def test_monte_carlo_is_reproducible():
    result_a = monte_carlo_trade_bootstrap([0.1, -0.05, 0.03], simulations=50, seed=7)
    result_b = monte_carlo_trade_bootstrap([0.1, -0.05, 0.03], simulations=50, seed=7)
    assert result_a == result_b
    assert 0 <= result_a.pass_rate <= 1


def test_monte_carlo_rejects_invalid_returns():
    with pytest.raises(ValueError, match="greater than -100"):
        monte_carlo_trade_bootstrap([-1.0])


def test_regime_returns_preserves_index_and_partitions():
    index = pd.date_range("2024-01-01", periods=30)
    equity = pd.Series(range(100, 130), index=index, dtype=float)
    benchmark = pd.Series(range(200, 230), index=index, dtype=float)
    regimes = regime_returns(equity, benchmark, volatility_window=5)
    assert set(regimes) == {"rising", "falling", "high_volatility", "low_volatility"}
    assert regimes["rising"].index.equals(index)
    assert regimes["falling"].empty


def test_regime_stability_uses_weakest_covered_regime():
    regimes = {
        "rising": pd.Series([0.02, 0.01]),
        "falling": pd.Series([-0.01, 0.02]),
        "empty": pd.Series(dtype=float),
    }
    result = score_regime_stability(regimes)
    assert result.covered_regimes == ("falling", "rising")
    assert result.score < 0
    assert result.regime_returns["falling"] < result.regime_returns["rising"]


def _candidate() -> StrategyCandidate:
    strategy = StrategySpec(
        name="robustness-test",
        universe=["TEST"],
        indicators=[Indicator(name="fast", kind="sma", source="close", period=2)],
        entry=Signal(all=[Condition(left="fast", comparator=Comparator.GT, right=100.0)]),
        exit=Signal(all=[Condition(left="fast", comparator=Comparator.LT, right=100.0)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.25, max_position=0.25),
        risk=RiskLimits(max_position=0.25),
    )
    return StrategyCandidate(strategy, "request-1", None, "test", "candidate-1")


def test_position_signal_honors_exit_before_entry_on_each_bar():
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    entry = pd.Series([True, False, False, True], index=index)
    exit_ = pd.Series([False, False, True, False], index=index)

    result = robustness._position_signal(entry, exit_)

    assert result.tolist() == [True, True, False, True]


def test_robustness_uses_complete_entry_exit_lifecycle(monkeypatch):
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    data = pd.DataFrame({"close": [100.0, 101.0, 99.0, 102.0]}, index=index)
    candidate = _candidate()
    observed_positions = []

    def fake_signals(_data, _strategy):
        return (
            pd.Series([True, False, False, True], index=index),
            pd.Series([False, False, True, False], index=index),
        )

    def fake_backtest(_data, position, _strategy, _config):
        observed_positions.append(position.tolist())
        equity = pd.Series([1.0, 1.1, 1.1, 1.2], index=index)
        return BacktestResult(
            equity=equity,
            returns=equity.pct_change().fillna(0.0),
            trades=pd.DataFrame(columns=["timestamp", "action"]),
            total_return=0.2,
            max_drawdown=0.0,
            trade_returns=(0.1, 0.0909),
        )

    monkeypatch.setattr(robustness, "strategy_signals", fake_signals)
    monkeypatch.setattr(robustness, "run_long_signal_backtest", fake_backtest)

    result = robustness.test_robustness(data, candidate)

    assert result.passed
    assert observed_positions
    assert all(position == [True, True, False, True] for position in observed_positions)


def test_default_scenarios_scale_cost_and_slippage():
    scenarios = robustness.default_scenarios()
    names = [scenario.name for scenario in scenarios]

    assert names == ["baseline", "cost_2x", "slippage_2x", "cost_slippage_2x"]
    assert scenarios[1].config.commission_bps == scenarios[0].config.commission_bps * 2
    assert scenarios[2].config.slippage_bps == scenarios[0].config.slippage_bps * 2
    assert scenarios[3].config.commission_bps == scenarios[0].config.commission_bps * 2
    assert scenarios[3].config.slippage_bps == scenarios[0].config.slippage_bps * 2
