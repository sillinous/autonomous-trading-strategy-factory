import sqlite3

import numpy as np
import pandas as pd
import pytest

from atsf.portfolio_attribution import attribute_portfolio
from atsf.portfolio_health import assess_portfolio_health
from atsf.portfolio_intelligence import PortfolioIntelligencePolicy, build_portfolio_intelligence
from atsf.portfolio_lifecycle import assess_and_persist_portfolio_health, persist_portfolio_health
from atsf.portfolio_lifecycle_store import PortfolioLifecycleStore
from atsf.ranking import RankedCandidate


def _portfolio():
    ranked = tuple(
        RankedCandidate(strategy_id, 1.0 - i * 0.1, 1.0, 1.0, 1.0 - i * 0.1)
        for i, strategy_id in enumerate(("a", "b"))
    )
    index = pd.date_range("2026-01-01", periods=20)
    base = np.linspace(-0.005, 0.005, 20)
    returns = pd.DataFrame({"a": base, "b": -base}, index=index)
    return build_portfolio_intelligence(
        ranked,
        returns,
        policy=PortfolioIntelligencePolicy(max_portfolio_volatility=0.01),
    ), returns


def test_persist_portfolio_health_round_trips_and_binds_decision():
    portfolio, returns = _portfolio()
    attribution = attribute_portfolio(returns, portfolio.allocation.weights)
    health = assess_portfolio_health(attribution, observation_count=len(returns))
    store = PortfolioLifecycleStore(sqlite3.connect(":memory:"))

    record = persist_portfolio_health(
        store,
        portfolio_id="portfolio-1",
        generation=3,
        portfolio=portfolio,
        health=health,
        created_at="2026-09-15T00:00:00+00:00",
    )

    assert record.strategy_ids == tuple(sorted(portfolio.selection.selected))
    assert dict(record.weights) == portfolio.allocation.weights
    assert record.decision == health.decision
    assert record.execution_authority is False
    assert store.latest("portfolio-1") == record
    store.verify("portfolio-1")


def test_assess_and_persist_derives_health_from_returns():
    portfolio, returns = _portfolio()
    store = PortfolioLifecycleStore(sqlite3.connect(":memory:"))

    record = assess_and_persist_portfolio_health(
        store,
        portfolio_id="portfolio-1",
        generation=1,
        portfolio=portfolio,
        returns=returns,
        created_at="2026-09-15T00:00:00+00:00",
    )

    assert record.health_status == "HEALTHY"
    assert record.decision == "CONTINUE_PORTFOLIO"
    assert record.total_return == pytest.approx(0.0)
    assert record.execution_authority is False


def test_persist_rejects_selection_allocation_mismatch():
    portfolio, returns = _portfolio()
    attribution = attribute_portfolio(returns, portfolio.allocation.weights)
    health = assess_portfolio_health(attribution, observation_count=len(returns))
    tampered = type(portfolio)(
        ranked=portfolio.ranked,
        selection=portfolio.selection,
        allocation=type(portfolio.allocation)(weights={"a": 1.0}, estimated_volatility={"a": 0.1}, total_weight=1.0),
        portfolio_volatility=portfolio.portfolio_volatility,
        admission_passed=portfolio.admission_passed,
        rejected_for_risk=portfolio.rejected_for_risk,
    )
    store = PortfolioLifecycleStore(sqlite3.connect(":memory:"))

    with pytest.raises(ValueError, match="selection and allocation"):
        persist_portfolio_health(
            store,
            portfolio_id="portfolio-1",
            generation=1,
            portfolio=tampered,
            health=health,
        )
