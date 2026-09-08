from __future__ import annotations

from dataclasses import dataclass

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


def select_diversified_strategies(
    returns: pd.DataFrame,
    ranked_strategy_ids: list[str],
    policy: PortfolioPolicy | None = None,
) -> PortfolioSelection:
    """Greedily select ranked strategies while enforcing correlation concentration limits."""
    policy = policy or PortfolioPolicy()
    if policy.max_strategies <= 0:
        raise ValueError("max_strategies must be positive")
    if not 0 <= policy.max_average_correlation <= 1:
        raise ValueError("max_average_correlation must be between 0 and 1")
    if len(returns) < policy.min_history:
        raise ValueError("insufficient return history")
    missing = [identifier for identifier in ranked_strategy_ids if identifier not in returns.columns]
    if missing:
        raise ValueError(f"missing strategy returns: {missing}")

    selected: list[str] = []
    rejected: list[str] = []
    for identifier in ranked_strategy_ids:
        if len(selected) >= policy.max_strategies:
            rejected.append(identifier)
            continue
        if not selected:
            selected.append(identifier)
            continue
        correlations = returns[selected + [identifier]].corr()[identifier].drop(identifier)
        average = float(correlations.abs().mean())
        if average <= policy.max_average_correlation:
            selected.append(identifier)
        else:
            rejected.append(identifier)

    average_correlation = 0.0
    if len(selected) > 1:
        matrix = returns[selected].corr().abs()
        upper = [
            float(matrix.iloc[i, j])
            for i in range(len(selected))
            for j in range(i + 1, len(selected))
        ]
        average_correlation = sum(upper) / len(upper) if upper else 0.0
    return PortfolioSelection(tuple(selected), tuple(rejected), average_correlation)
