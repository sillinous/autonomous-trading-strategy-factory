from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import pandas as pd

from .allocation import AllocationPolicy, PortfolioAllocation, allocate_inverse_volatility
from .portfolio import PortfolioPolicy, PortfolioSelection, select_diversified_strategies
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


def _portfolio_volatility(returns: pd.DataFrame, allocation: PortfolioAllocation) -> float:
    ids = tuple(allocation.weights)
    if not ids:
        return 0.0
    selected = returns[list(ids)].astype(float).replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(selected) < 2:
        raise ValueError("at least two finite return observations are required")
    weights = pd.Series(allocation.weights, index=ids, dtype=float)
    covariance = selected[list(ids)].cov().to_numpy(dtype=float)
    variance = float(weights.to_numpy() @ covariance @ weights.to_numpy())
    if variance < 0 and variance > -1e-12:
        variance = 0.0
    if variance < 0 or not isfinite(variance):
        raise ValueError("portfolio covariance produced an invalid variance")
    volatility = variance ** 0.5
    if not isfinite(volatility):
        raise ValueError("portfolio volatility is not finite")
    return volatility


def build_portfolio_intelligence(
    ranked: tuple[RankedCandidate, ...],
    returns: pd.DataFrame,
    *,
    policy: PortfolioIntelligencePolicy | None = None,
) -> PortfolioIntelligenceResult:
    """Turn strategy rankings into a deterministic risk-aware portfolio admission decision."""
    policy = policy or PortfolioIntelligencePolicy()
    if not ranked:
        raise ValueError("ranked candidates cannot be empty")
    if returns.empty:
        raise ValueError("returns cannot be empty")

    ids = [candidate.candidate_id for candidate in ranked]
    selection = select_diversified_strategies(returns, ids, policy.portfolio)
    if not selection.selected:
        raise ValueError("portfolio selection produced no strategies")

    allocation = allocate_inverse_volatility(returns, selection.selected, policy.allocation)
    volatility = _portfolio_volatility(returns, allocation)
    passed = volatility <= policy.max_portfolio_volatility
    rejected_for_risk = selection.selected if not passed else ()

    return PortfolioIntelligenceResult(
        ranked=ranked,
        selection=selection,
        allocation=allocation,
        portfolio_volatility=volatility,
        admission_passed=passed,
        rejected_for_risk=tuple(rejected_for_risk),
    )
