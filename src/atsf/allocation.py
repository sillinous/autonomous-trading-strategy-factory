from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AllocationPolicy:
    max_total_weight: float = 1.0
    min_weight: float = 0.0
    volatility_floor: float = 1e-8


@dataclass(frozen=True)
class PortfolioAllocation:
    weights: dict[str, float]
    estimated_volatility: dict[str, float]
    total_weight: float


def allocate_inverse_volatility(
    returns: pd.DataFrame,
    strategy_ids: tuple[str, ...],
    policy: AllocationPolicy | None = None,
) -> PortfolioAllocation:
    """Allocate capital inversely to realized volatility, capped by total exposure."""
    policy = policy or AllocationPolicy()
    if not 0 < policy.max_total_weight <= 1:
        raise ValueError("max_total_weight must be in (0, 1]")
    if policy.min_weight < 0:
        raise ValueError("min_weight cannot be negative")
    if policy.min_weight * len(strategy_ids) > policy.max_total_weight:
        raise ValueError("minimum weights exceed total exposure")
    if not strategy_ids:
        return PortfolioAllocation({}, {}, 0.0)
    missing = [sid for sid in strategy_ids if sid not in returns.columns]
    if missing:
        raise ValueError(f"missing strategy returns: {missing}")
    volatility = returns[list(strategy_ids)].astype(float).std(ddof=1)
    if not volatility.notna().all():
        raise ValueError("strategy volatility is non-finite")
    vol = {sid: max(float(volatility[sid]), policy.volatility_floor) for sid in strategy_ids}
    inverse = {sid: 1.0 / value for sid, value in vol.items()}
    total_inverse = sum(inverse.values())
    weights = {
        sid: max(policy.min_weight, policy.max_total_weight * inverse[sid] / total_inverse)
        for sid in strategy_ids
    }
    if sum(weights.values()) > policy.max_total_weight + 1e-12:
        scale = policy.max_total_weight / sum(weights.values())
        weights = {sid: weight * scale for sid, weight in weights.items()}
    return PortfolioAllocation(weights, vol, sum(weights.values()))
