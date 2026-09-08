import pandas as pd
import pytest

from atsf.robustness import (
    monte_carlo_trade_bootstrap,
    regime_returns,
    score_regime_stability,
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
