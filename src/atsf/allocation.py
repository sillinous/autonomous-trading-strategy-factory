from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class AllocationPolicy:
    max_total_weight: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 1.0
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
    """Allocate inverse-volatility weights with explicit minimum and maximum caps."""
    policy = policy or AllocationPolicy()
    if not 0 < policy.max_total_weight <= 1:
        raise ValueError("max_total_weight must be in (0, 1]")
    if policy.min_weight < 0:
        raise ValueError("min_weight cannot be negative")
    if policy.max_weight < policy.min_weight or policy.max_weight <= 0:
        raise ValueError("max_weight must be positive and at least min_weight")
    if policy.min_weight * len(strategy_ids) > policy.max_total_weight:
        raise ValueError("minimum weights exceed total exposure")
    if policy.volatility_floor <= 0 or not math.isfinite(policy.volatility_floor):
        raise ValueError("volatility_floor must be positive and finite")
    if len(set(strategy_ids)) != len(strategy_ids):
        raise ValueError("strategy IDs must be unique")
    if not strategy_ids:
        return PortfolioAllocation({}, {}, 0.0)
    if returns.empty:
        raise ValueError("returns cannot be empty")
    missing = [sid for sid in strategy_ids if sid not in returns.columns]
    if missing:
        raise ValueError(f"missing strategy returns: {missing}")
    selected = returns[list(strategy_ids)].astype(float)
    if not selected.map(math.isfinite).all().all():
        raise ValueError("strategy returns must contain only finite values")
    volatility = selected.std(ddof=1)
    if not volatility.map(math.isfinite).all():
        raise ValueError("strategy volatility is non-finite")
    vol = {sid: max(float(volatility[sid]), policy.volatility_floor) for sid in strategy_ids}
    inverse = {sid: 1.0 / vol[sid] for sid in strategy_ids}
    weights = {sid: policy.min_weight for sid in strategy_ids}
    remaining = policy.max_total_weight - sum(weights.values())
    active = list(strategy_ids)
    while active and remaining > 1e-12:
        total_inverse = sum(inverse[sid] for sid in active)
        proposed = {sid: remaining * inverse[sid] / total_inverse for sid in active}
        capped = [sid for sid in active if weights[sid] + proposed[sid] > policy.max_weight + 1e-12]
        if not capped:
            for sid in active:
                weights[sid] += proposed[sid]
            remaining = 0.0
            break
        for sid in capped:
            increment = policy.max_weight - weights[sid]
            weights[sid] = policy.max_weight
            remaining -= increment
            active.remove(sid)
    total_weight = sum(weights.values())
    return PortfolioAllocation(weights, vol, total_weight)
