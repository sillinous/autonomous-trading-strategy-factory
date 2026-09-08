from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyAttribution:
    strategy_id: str
    return_contribution: float
    risk_contribution: float


@dataclass(frozen=True)
class PortfolioAttribution:
    total_return: float
    volatility: float
    contributions: tuple[StrategyAttribution, ...]


def attribute_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> PortfolioAttribution:
    """Attribute portfolio return and volatility to fixed-weight strategy sleeves."""
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if not weights:
        raise ValueError("weights cannot be empty")
    if set(weights) != set(returns.columns):
        raise ValueError("weights and returns must contain the same strategy IDs")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("weights must be finite and non-negative")
    if sum(weights.values()) > 1.0 + 1e-12:
        raise ValueError("weights exceed 100% gross exposure")
    selected = returns[list(weights)].astype(float)
    if not selected.map(math.isfinite).all().all():
        raise ValueError("returns must contain only finite values")
    weight_vector = np.array([weights[sid] for sid in weights], dtype=float)
    daily = selected.to_numpy() @ weight_vector
    portfolio_returns = pd.Series(daily, index=returns.index)
    equity = (1.0 + portfolio_returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    volatility = float(portfolio_returns.std(ddof=1)) if len(portfolio_returns) > 1 else 0.0

    prior_equity = equity.shift(1, fill_value=1.0).to_numpy()
    contribution_values = selected.to_numpy() * weight_vector * prior_equity[:, None]
    contributions = contribution_values.sum(axis=0)
    covariance = selected.cov().to_numpy()
    portfolio_variance = float(weight_vector @ covariance @ weight_vector)
    portfolio_vol = math.sqrt(max(portfolio_variance, 0.0))
    if portfolio_vol > 0:
        risk_values = weight_vector * (covariance @ weight_vector) / portfolio_vol
    else:
        risk_values = np.zeros(len(weight_vector))
    result = tuple(
        StrategyAttribution(sid, float(contribution), float(risk))
        for sid, contribution, risk in zip(weights, contributions, risk_values, strict=True)
    )
    return PortfolioAttribution(total_return, volatility, result)
