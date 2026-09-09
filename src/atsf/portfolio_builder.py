from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .allocation import AllocationPolicy, PortfolioAllocation, allocate_inverse_volatility
from .portfolio import PortfolioPolicy, PortfolioSelection, select_diversified_strategies
from .ranking import RankedCandidate


@dataclass(frozen=True)
class PortfolioBuildResult:
    ranked: tuple[RankedCandidate, ...]
    selection: PortfolioSelection
    allocation: PortfolioAllocation


def build_portfolio(
    ranked: tuple[RankedCandidate, ...],
    returns: pd.DataFrame,
    *,
    portfolio_policy: PortfolioPolicy | None = None,
    allocation_policy: AllocationPolicy | None = None,
) -> PortfolioBuildResult:
    """Select complementary ranked strategies and allocate capital deterministically."""
    if not ranked:
        raise ValueError("ranked candidates cannot be empty")
    ids = [candidate.candidate_id for candidate in ranked]
    selection = select_diversified_strategies(returns, ids, portfolio_policy)
    allocation = allocate_inverse_volatility(
        returns,
        selection.selected,
        allocation_policy,
    )
    return PortfolioBuildResult(ranked, selection, allocation)
