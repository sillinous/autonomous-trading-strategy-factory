import pandas as pd

from atsf.ranking import diversity_score


def test_inverse_strategy_is_more_diverse_than_correlated_strategy():
    index = pd.date_range("2025-01-01", periods=30)
    base = pd.Series(range(30), index=index, dtype=float)
    returns = pd.DataFrame({"base": base, "similar": base * 2, "inverse": -base}, index=index)
    assert diversity_score(returns, "inverse", ["base"]) > diversity_score(returns, "similar", ["base"])


def test_first_strategy_gets_maximum_diversity_score():
    returns = pd.DataFrame({"candidate": [0.01, -0.01, 0.02]})
    assert diversity_score(returns, "candidate", []) == 1.0
