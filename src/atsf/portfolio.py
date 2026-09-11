from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class PortfolioPolicy:
    max_strategies: int = 10
    max_average_correlation: float = 0.75
    min_history: int = 20

@dataclass(frozen=True)
class PortfolioSelection:
    selected: tuple[str, ...]
    rejected: tuple[str, ...]
    average_correlation: float


def select_diversified_strategies(returns: pd.DataFrame, ranked_strategy_ids: list[str], policy: PortfolioPolicy | None = None) -> PortfolioSelection:
    policy = policy or PortfolioPolicy()
    if policy.max_strategies <= 0: raise ValueError("max_strategies must be positive")
    if not 0 <= policy.max_average_correlation <= 1: raise ValueError("max_average_correlation must be between 0 and 1")
    if policy.min_history < 2: raise ValueError("min_history must be at least 2")
    if len(returns) < policy.min_history: raise ValueError("insufficient return history")
    if len(set(ranked_strategy_ids)) != len(ranked_strategy_ids): raise ValueError("ranked strategy IDs must be unique")
    missing = [identifier for identifier in ranked_strategy_ids if identifier not in returns.columns]
    if missing: raise ValueError(f"missing strategy returns: {missing}")
    selected_returns = returns[ranked_strategy_ids].astype(float).replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(selected_returns) < policy.min_history: raise ValueError("insufficient finite return history")

    selected: list[str] = []
    rejected: list[str] = []
    for identifier in ranked_strategy_ids:
        if len(selected) >= policy.max_strategies:
            rejected.append(identifier); continue
        if not selected:
            selected.append(identifier); continue
        correlations = selected_returns[selected + [identifier]].corr()[identifier].drop(identifier).dropna()
        if correlations.empty:
            rejected.append(identifier); continue
        positive_average = float(correlations.clip(lower=0).mean())
        if math.isfinite(positive_average) and positive_average <= policy.max_average_correlation:
            selected.append(identifier)
        else:
            rejected.append(identifier)

    average_correlation = 0.0
    if len(selected) > 1:
        matrix = selected_returns[selected].corr()
        upper = [max(0.0, float(matrix.iloc[i, j])) for i in range(len(selected)) for j in range(i + 1, len(selected)) if math.isfinite(float(matrix.iloc[i, j]))]
        average_correlation = sum(upper) / len(upper) if upper else 0.0
    return PortfolioSelection(tuple(selected), tuple(rejected), average_correlation)
