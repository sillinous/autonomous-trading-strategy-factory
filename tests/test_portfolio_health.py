import pandas as pd
import pytest

from atsf.portfolio_attribution import attribute_portfolio
from atsf.portfolio_health import PortfolioHealthPolicy, assess_portfolio_health


def test_healthy_portfolio():
    returns = pd.DataFrame({"a": [0.01, 0.00, 0.02], "b": [0.00, 0.02, 0.00]})
    attribution = attribute_portfolio(returns, {"a": 0.5, "b": 0.5})
    result = assess_portfolio_health(
        attribution,
        observation_count=3,
        policy=PortfolioHealthPolicy(max_single_strategy_risk_fraction=1.0),
    )
    assert result.healthy
    assert result.breached_limits == ()
    assert result.replace_strategy_ids == ()


def test_concentration_identifies_risk_contributor_for_replacement():
    returns = pd.DataFrame({"a": [0.10, 0.10, 0.10], "b": [0.00, 0.01, -0.01]})
    attribution = attribute_portfolio(returns, {"a": 0.9, "b": 0.1})
    result = assess_portfolio_health(
        attribution,
        observation_count=3,
        policy=PortfolioHealthPolicy(max_single_strategy_risk_fraction=0.60),
    )
    assert result.status == "REVIEW_REQUIRED"
    assert "concentration" in result.breached_limits
    assert result.replace_strategy_ids == ("b",)


def test_insufficient_observations_fail_closed():
    returns = pd.DataFrame({"a": [0.01, 0.02], "b": [0.00, 0.01]})
    attribution = attribute_portfolio(returns, {"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError, match="insufficient observations"):
        assess_portfolio_health(attribution, observation_count=1)


def test_nonfinite_attribution_fails_closed():
    returns = pd.DataFrame({"a": [0.01, 0.02], "b": [0.00, 0.01]})
    attribution = attribute_portfolio(returns, {"a": 0.5, "b": 0.5})
    bad = type(attribution)(float("nan"), attribution.volatility, attribution.contributions)
    with pytest.raises(ValueError, match="non-finite"):
        assess_portfolio_health(bad, observation_count=2)
