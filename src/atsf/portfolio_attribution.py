from __future__ import annotations

from dataclasses import dataclass
import math

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


def attribute_portfolio(returns: pd.DataFrame, weights: dict[str, float]) -> PortfolioAttribution:
    if returns.empty or not weights: raise ValueError("returns and weights cannot be empty")
    if set(weights) != set(returns.columns): raise ValueError("weights and returns must contain the same strategy IDs")
    if any(not math.isfinite(w) or w < 0 for w in weights.values()): raise ValueError("weights must be finite and non-negative")
    if sum(weights.values()) > 1.0 + 1e-12: raise ValueError("weights exceed 100% gross exposure")
    selected = returns[list(weights)].astype(float)
    if not np.isfinite(selected.to_numpy()).all(): raise ValueError("returns must contain only finite values")
    w = np.array([weights[sid] for sid in weights], dtype=float)
    daily = selected.to_numpy() @ w
    series = pd.Series(daily, index=returns.index)
    contributions = (selected.to_numpy() * w).sum(axis=0)
    total_return = float(contributions.sum())
    volatility = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    covariance = selected.cov().to_numpy()
    variance = float(w @ covariance @ w)
    portfolio_vol = float(np.sqrt(max(variance, 0.0)))
    risk = w * (covariance @ w) / portfolio_vol if portfolio_vol else np.zeros(len(w))
    return PortfolioAttribution(total_return, volatility, tuple(StrategyAttribution(sid, float(ret), float(rc)) for sid, ret, rc in zip(weights, contributions, risk, strict=True)))
