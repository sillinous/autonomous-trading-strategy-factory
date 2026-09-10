from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .portfolio_attribution import PortfolioAttribution, attribute_portfolio


@dataclass(frozen=True)
class PortfolioRunIdentity:
    """Deterministic identity for a paper portfolio execution."""

    run_id: str
    portfolio_id: str
    dataset_version: str


def portfolio_run_id(
    portfolio_id: str,
    dataset_version: str,
    strategy_ids: tuple[str, ...],
    weights: dict[str, float],
) -> str:
    if not portfolio_id or not dataset_version:
        raise ValueError("portfolio_id and dataset_version are required")
    if not strategy_ids or len(set(strategy_ids)) != len(strategy_ids):
        raise ValueError("strategy_ids must be non-empty and unique")
    if set(weights) != set(strategy_ids):
        raise ValueError("weights must match strategy_ids")
    payload = {
        "portfolio_id": portfolio_id,
        "dataset_version": dataset_version,
        "strategy_ids": list(strategy_ids),
        "weights": {key: weights[key] for key in sorted(weights)},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:16]


def build_portfolio_run_identity(
    portfolio_id: str,
    dataset_version: str,
    weights: dict[str, float],
) -> PortfolioRunIdentity:
    strategy_ids = tuple(sorted(weights))
    return PortfolioRunIdentity(
        run_id=portfolio_run_id(portfolio_id, dataset_version, strategy_ids, weights),
        portfolio_id=portfolio_id,
        dataset_version=dataset_version,
    )


def attribute_run(
    returns,
    weights: dict[str, float],
) -> PortfolioAttribution:
    """Compute deterministic strategy-level return and risk attribution."""
    return attribute_portfolio(returns, weights)
