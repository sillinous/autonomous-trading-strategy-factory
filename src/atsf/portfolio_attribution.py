from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

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
    if not isinstance(returns, pd.DataFrame) or returns.empty or not isinstance(weights, dict) or not weights:
        raise ValueError("returns and weights cannot be empty")
    if not returns.columns.is_unique:
        raise ValueError("returns columns must be unique")
    if any(not isinstance(strategy_id, str) or not strategy_id for strategy_id in weights):
        raise ValueError("strategy IDs must be non-empty strings")
    if set(weights) != set(returns.columns):
        raise ValueError("weights and returns must contain the same strategy IDs")
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not math.isfinite(float(weight))
        or weight < 0
        for weight in weights.values()
    ):
        raise ValueError("weights must be finite non-negative numbers")
    if sum(weights.values()) > 1.0 + 1e-12:
        raise ValueError("weights exceed 100% gross exposure")
    selected = returns[list(weights)]
    if any(
        not pd.api.types.is_numeric_dtype(selected[column].dtype)
        or pd.api.types.is_bool_dtype(selected[column].dtype)
        for column in selected.columns
    ):
        raise ValueError("returns must contain numeric, non-boolean values")
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")
    w = np.array([weights[sid] for sid in weights], dtype=float)
    daily = values @ w
    series = pd.Series(daily, index=returns.index)
    contributions = (values * w).sum(axis=0)
    total_return = float(contributions.sum())
    volatility = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    # Sample covariance is undefined for a single observation. Consistent
    # with the reported zero volatility, attribute no measured risk in that case.
    covariance = selected.cov().to_numpy() if len(selected) > 1 else np.zeros((len(w), len(w)))
    variance = float(w @ covariance @ w)
    portfolio_vol = float(np.sqrt(max(variance, 0.0)))
    risk = w * (covariance @ w) / portfolio_vol if portfolio_vol else np.zeros(len(w))
    return PortfolioAttribution(
        total_return,
        volatility,
        tuple(
            StrategyAttribution(sid, float(ret), float(rc))
            for sid, ret, rc in zip(weights, contributions, risk, strict=True)
        ),
    )
