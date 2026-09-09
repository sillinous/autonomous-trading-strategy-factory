import pandas as pd
import pytest

from atsf.allocation import AllocationPolicy
from atsf.portfolio import PortfolioPolicy
from atsf.portfolio_builder import build_portfolio
from atsf.ranking import RankedCandidate


def candidate(identifier: str, score: float) -> RankedCandidate:
    return RankedCandidate(identifier, score, score, 1.0, score)


def test_build_portfolio_selects_diverse_candidates_and_allocates():
    index = pd.date_range("2025-01-01", periods=30)
    base = pd.Series(range(30), index=index, dtype=float).pct_change().fillna(0.0)
    inverse = -base
    similar = base * 2.0
    returns = pd.DataFrame({"a": base, "b": similar, "c": inverse}, index=index)
    ranked = (candidate("a", 3.0), candidate("b", 2.0), candidate("c", 1.0))
    result = build_portfolio(
        ranked,
        returns,
        portfolio_policy=PortfolioPolicy(max_strategies=2, max_average_correlation=0.75, min_history=20),
        allocation_policy=AllocationPolicy(max_total_weight=1.0),
    )
    assert result.selection.selected == ("a", "c")
    assert result.selection.rejected == ("b",)
    assert abs(result.allocation.total_weight - 1.0) < 1e-12
    assert abs(sum(result.allocation.weights.values()) - 1.0) < 1e-12


def test_build_portfolio_requires_return_columns():
    returns = pd.DataFrame({"a": [0.01] * 20})
    with pytest.raises(ValueError, match="missing strategy returns"):
        build_portfolio((candidate("a", 1.0), candidate("b", 0.5)), returns)
