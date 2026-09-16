from __future__ import annotations

import pandas as pd

from .portfolio_attribution import attribute_portfolio
from .portfolio_health import PortfolioHealthPolicy, PortfolioHealthResult, assess_portfolio_health
from .portfolio_intelligence import PortfolioIntelligenceResult
from .portfolio_lifecycle_store import PortfolioLifecycleRecord, PortfolioLifecycleStore


def persist_portfolio_health(
    store: PortfolioLifecycleStore,
    *,
    portfolio_id: str,
    generation: int,
    portfolio: PortfolioIntelligenceResult,
    health: PortfolioHealthResult,
    created_at: str | None = None,
) -> PortfolioLifecycleRecord:
    """Persist an auditable portfolio health decision without granting authority."""
    if not portfolio_id.strip():
        raise ValueError("portfolio_id must be non-empty")
    if set(portfolio.selection.selected) != set(portfolio.allocation.weights):
        raise ValueError("portfolio selection and allocation are inconsistent")
    record = store.new_record(
        portfolio_id=portfolio_id,
        generation=generation,
        strategy_ids=tuple(portfolio.selection.selected),
        weights=portfolio.allocation.weights,
        health_status=health.status,
        decision=health.decision,
        total_return=health.total_return,
        volatility=health.volatility,
        max_risk_fraction=health.max_risk_fraction,
        breached_limits=health.breached_limits,
        replace_strategy_ids=health.replace_strategy_ids,
        created_at=created_at,
    )
    return store.append(record)


def assess_and_persist_portfolio_health(
    store: PortfolioLifecycleStore,
    *,
    portfolio_id: str,
    generation: int,
    portfolio: PortfolioIntelligenceResult,
    returns: pd.DataFrame,
    policy: PortfolioHealthPolicy | None = None,
    created_at: str | None = None,
) -> PortfolioLifecycleRecord:
    """Derive health from the selected portfolio and persist the resulting decision."""
    if set(portfolio.allocation.weights) != set(portfolio.selection.selected):
        raise ValueError("portfolio selection and allocation are inconsistent")
    selected_returns = returns[list(portfolio.selection.selected)]
    attribution = attribute_portfolio(selected_returns, portfolio.allocation.weights)
    health = assess_portfolio_health(
        attribution,
        observation_count=len(selected_returns),
        policy=policy,
    )
    return persist_portfolio_health(
        store,
        portfolio_id=portfolio_id,
        generation=generation,
        portfolio=portfolio,
        health=health,
        created_at=created_at,
    )
