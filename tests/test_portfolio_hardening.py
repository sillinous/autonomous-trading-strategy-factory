import numpy as np
import pandas as pd
import pytest

from atsf.portfolio import PortfolioPolicy, select_diversified_strategies


def test_portfolio_rejects_nonfinite_returns():
    returns = pd.DataFrame({"a": [0.01] * 19 + [np.nan]})
    with pytest.raises(ValueError, match="finite"):
        select_diversified_strategies(returns, ["a"])


def test_portfolio_requires_minimum_history_of_two():
    returns = pd.DataFrame({"a": [0.01]})
    with pytest.raises(ValueError, match="min_history"):
        select_diversified_strategies(returns, ["a"], PortfolioPolicy(min_history=1))


def test_portfolio_rejects_constant_series_that_cannot_produce_correlation():
    index = pd.date_range("2025-01-01", periods=20)
    returns = pd.DataFrame(
        {"constant": [0.01] * 20, "variable": np.linspace(-0.02, 0.02, 20)},
        index=index,
    )
    result = select_diversified_strategies(returns, ["constant", "variable"])
    assert result.selected == ("constant",)
    assert result.rejected == ("variable",)
