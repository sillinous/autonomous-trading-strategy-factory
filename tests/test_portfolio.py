import pandas as pd
import pytest

from atsf.portfolio import PortfolioPolicy, select_diversified_strategies


def test_portfolio_rejects_highly_correlated_strategy():
    index = pd.date_range("2025-01-01", periods=30)
    base = pd.Series([i * 0.01 + (i % 3) * 0.001 for i in range(30)], index=index)
    returns = pd.DataFrame({"a": base, "b": base * 1.01, "c": -base}, index=index)
    result = select_diversified_strategies(
        returns,
        ["a", "b", "c"],
        PortfolioPolicy(max_strategies=3, max_average_correlation=0.75, min_history=20),
    )
    assert result.selected == ("a", "c")
    assert "b" in result.rejected


def test_portfolio_requires_history_and_known_ids():
    index = pd.date_range("2025-01-01", periods=5)
    returns = pd.DataFrame({"a": [0.01] * 5}, index=index)
    with pytest.raises(ValueError, match="insufficient"):
        select_diversified_strategies(returns, ["a"], PortfolioPolicy(min_history=20))
