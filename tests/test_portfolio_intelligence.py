import numpy as np
import pandas as pd
import pytest

from atsf.portfolio_intelligence import (
    PortfolioIntelligencePolicy,
    build_portfolio_intelligence,
)
from atsf.ranking import RankedCandidate


def ranked(*ids: str) -> tuple[RankedCandidate, ...]:
    return tuple(
        RankedCandidate(identifier, 0.9 - index * 0.05, 0.9, 0.8, 0.85 - index * 0.05)
        for index, identifier in enumerate(ids)
    )


def returns_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=40)
    base = np.linspace(-0.01, 0.01, 40)
    return pd.DataFrame(
        {
            "a": base,
            "b": -base,
            "c": np.roll(base, 3),
        },
        index=index,
    )


def test_portfolio_intelligence_is_deterministic_and_risk_aware():
    returns = returns_frame()
    policy = PortfolioIntelligencePolicy(max_portfolio_volatility=0.01)

    first = build_portfolio_intelligence(ranked("a", "b", "c"), returns, policy=policy)
    second = build_portfolio_intelligence(ranked("a", "b", "c"), returns, policy=policy)

    assert first == second
    assert first.selection.selected
    assert first.allocation.total_weight == pytest.approx(1.0)
    assert first.portfolio_volatility <= 0.01
    assert first.admission_passed is True


def test_portfolio_intelligence_rejects_excessive_portfolio_risk():
    index = pd.date_range("2026-01-01", periods=40)
    returns = pd.DataFrame(
        {
            "a": np.tile([0.20, -0.20], 20),
            "b": np.tile([0.20, -0.20], 20),
        },
        index=index,
    )
    policy = PortfolioIntelligencePolicy(max_portfolio_volatility=0.01)

    result = build_portfolio_intelligence(ranked("a", "b"), returns, policy=policy)

    assert result.admission_passed is False
    assert result.rejected_for_risk == result.selection.selected
    assert result.portfolio_volatility > 0.01


def test_portfolio_intelligence_fails_closed_without_ranked_candidates():
    with pytest.raises(ValueError, match="ranked candidates"):
        build_portfolio_intelligence((), returns_frame())
