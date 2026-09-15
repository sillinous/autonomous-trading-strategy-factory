from __future__ import annotations

from .portfolio_health import PortfolioHealthResult
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
    if portfolio.selection.selected != tuple(portfolio.allocation.weights):
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
