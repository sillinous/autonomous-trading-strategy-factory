from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import pandas as pd

from .allocation import AllocationPolicy, PortfolioAllocation, allocate_inverse_volatility
from .portfolio import PortfolioPolicy, PortfolioSelection, select_diversified_strategies
from .ranking import RankedCandidate


@dataclass(frozen=True)
class PortfolioRiskPolicy:
    """Fail-closed portfolio construction limits applied after strategy ranking."""

    portfolio: PortfolioPolicy = field(default_factory=PortfolioPolicy)
    allocation: AllocationPolicy = field(default_factory=AllocationPolicy)
    max_portfolio_volatility: float = 0.25

    def __post_init__(self) -> None:
        if not isfinite(self.max_portfolio_volatility) or self.max_portfolio_volatility <= 0:
            raise ValueError("max_portfolio_volatility must be finite and positive")


@dataclass(frozen=True)
class PortfolioRiskResult:
    ranked: tuple[RankedCandidate, ...]
    selection: PortfolioSelection
    allocation: PortfolioAllocation
    portfolio_volatility: float
    admitted_strategy_ids: tuple[str, ...]
    risk_rejected_strategy_ids: tuple[str, ...]
    admission_passed: bool


def _volatility(returns: pd.DataFrame, allocation: PortfolioAllocation) -> float:
    ids = tuple(allocation.weights)
    selected = returns[list(ids)].astype(float)
    if len(selected) < 2:
        raise ValueError("at least two return observations are required")
    covariance = selected.cov().to_numpy(dtype=float)
    weights = pd.Series(allocation.weights, index=ids, dtype=float).to_numpy()
    variance = float(weights @ covariance @ weights)
    if variance < 0 and variance > -1e-12:
        variance = 0.0
    if variance < 0 or not isfinite(variance):
        raise ValueError("portfolio variance is invalid")
    result = variance**0.5
    if not isfinite(result):
        raise ValueError("portfolio volatility is non-finite")
    return result


def build_risk_aware_portfolio(
    ranked: tuple[RankedCandidate, ...],
    returns: pd.DataFrame,
    *,
    policy: PortfolioRiskPolicy | None = None,
) -> PortfolioRiskResult:
    """Deterministically admit the largest ranked prefix that satisfies portfolio risk."""
    policy = policy or PortfolioRiskPolicy()
    if not ranked:
        raise ValueError("ranked candidates cannot be empty")
    if returns.empty:
        raise ValueError("returns cannot be empty")

    ids = [candidate.candidate_id for candidate in ranked]
    selection = select_diversified_strategies(returns, ids, policy.portfolio)
    candidates = list(selection.selected)
    risk_rejected: list[str] = []
    last_volatility = float("nan")

    while candidates:
        allocation = allocate_inverse_volatility(returns, tuple(candidates), policy.allocation)
        last_volatility = _volatility(returns, allocation)
        if last_volatility <= policy.max_portfolio_volatility:
            return PortfolioRiskResult(
                ranked=ranked,
                selection=selection,
                allocation=allocation,
                portfolio_volatility=last_volatility,
                admitted_strategy_ids=tuple(candidates),
                risk_rejected_strategy_ids=tuple(risk_rejected),
                admission_passed=True,
            )
        removed = candidates.pop()
        risk_rejected.append(removed)

    # No feasible portfolio exists. Preserve the final measured risk rather than
    # replacing it with zero, which would incorrectly imply a successful outcome.
    return PortfolioRiskResult(
        ranked=ranked,
        selection=selection,
        allocation=PortfolioAllocation({}, {}, 0.0),
        portfolio_volatility=last_volatility,
        admitted_strategy_ids=(),
        risk_rejected_strategy_ids=tuple(risk_rejected),
        admission_passed=False,
    )
