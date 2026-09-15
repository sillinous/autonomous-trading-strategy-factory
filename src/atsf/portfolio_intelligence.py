from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import pandas as pd

from .allocation import AllocationPolicy, PortfolioAllocation
from .portfolio import PortfolioPolicy, PortfolioSelection
from .portfolio_risk import PortfolioRiskPolicy, build_risk_aware_portfolio
from .ranking import RankedCandidate


@dataclass(frozen=True)
class PortfolioIntelligencePolicy:
    portfolio: PortfolioPolicy = field(default_factory=PortfolioPolicy)
    allocation: AllocationPolicy = field(default_factory=AllocationPolicy)
    max_portfolio_volatility: float = 0.25

    def __post_init__(self) -> None:
        if not isfinite(self.max_portfolio_volatility) or self.max_portfolio_volatility <= 0:
            raise ValueError("max_portfolio_volatility must be finite and positive")


@dataclass(frozen=True)
class PortfolioIntelligenceResult:
    ranked: tuple[RankedCandidate, ...]
    selection: PortfolioSelection
    allocation: PortfolioAllocation
    portfolio_volatility: float
    admission_passed: bool
    rejected_for_risk: tuple[str, ...]


def build_portfolio_intelligence(
    ranked: tuple[RankedCandidate, ...],
    returns: pd.DataFrame,
    *,
    policy: PortfolioIntelligencePolicy | None = None,
) -> PortfolioIntelligenceResult:
    """Build a deterministic portfolio with adaptive portfolio-level risk admission."""
    policy = policy or PortfolioIntelligencePolicy()
    result = build_risk_aware_portfolio(
        ranked,
        returns,
        policy=PortfolioRiskPolicy(
            portfolio=policy.portfolio,
            allocation=policy.allocation,
            max_portfolio_volatility=policy.max_portfolio_volatility,
        ),
    )
    return PortfolioIntelligenceResult(
        ranked=ranked,
        selection=result.selection,
        allocation=result.allocation,
        portfolio_volatility=result.portfolio_volatility,
        admission_passed=result.admission_passed,
        rejected_for_risk=result.risk_rejected_strategy_ids,
    )
