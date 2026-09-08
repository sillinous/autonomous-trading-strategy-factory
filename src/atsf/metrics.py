from __future__ import annotations

import math

import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    """Return maximum peak-to-trough drawdown as a negative fraction."""
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def sharpe_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    returns = equity.pct_change().dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(math.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=1))


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
