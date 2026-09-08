from __future__ import annotations

from dataclasses import dataclass

import math
import pandas as pd


@dataclass(frozen=True)
class DegradationPolicy:
    max_drawdown: float = 0.20
    min_return: float = -0.10
    max_volatility: float = 0.10
    min_observations: int = 20


@dataclass(frozen=True)
class DegradationReport:
    degraded: bool
    observations: int
    total_return: float
    max_drawdown: float
    volatility: float
    reasons: tuple[str, ...]


def assess_degradation(
    equity: pd.Series,
    policy: DegradationPolicy | None = None,
) -> DegradationReport:
    policy = policy or DegradationPolicy()
    if policy.min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    if not 0 < policy.max_drawdown < 1:
        raise ValueError("max_drawdown must be between 0 and 1")
    if policy.max_volatility < 0:
        raise ValueError("max_volatility cannot be negative")
    if len(equity) < policy.min_observations:
        raise ValueError("insufficient observations")
    values = equity.astype(float)
    if (values <= 0).any() or not values.index.is_monotonic_increasing:
        raise ValueError("equity must be positive and chronologically ordered")
    returns = values.pct_change().dropna()
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    drawdown = float((values / values.cummax() - 1.0).min())
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    reasons: list[str] = []
    if drawdown < -policy.max_drawdown:
        reasons.append("maximum drawdown breached")
    if total_return < policy.min_return:
        reasons.append("minimum return breached")
    if volatility > policy.max_volatility:
        reasons.append("volatility ceiling breached")
    return DegradationReport(
        bool(reasons), len(values), total_return, drawdown, volatility, tuple(reasons)
    )


def is_finite_report(report: DegradationReport) -> bool:
    return all(
        math.isfinite(value)
        for value in (report.total_return, report.max_drawdown, report.volatility)
    )
