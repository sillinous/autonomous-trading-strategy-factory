import pandas as pd
import pytest

from atsf.portfolio_run import attribute_run, build_portfolio_run_identity, portfolio_run_id


def test_portfolio_run_id_is_deterministic_and_order_independent_for_weights() -> None:
    weights = {"b": 0.4, "a": 0.6}
    first = portfolio_run_id("portfolio-1", "dataset-1", ("a", "b"), weights)
    second = portfolio_run_id("portfolio-1", "dataset-1", ("a", "b"), {"a": 0.6, "b": 0.4})
    assert first == second


def test_build_identity_sorts_strategy_ids() -> None:
    identity = build_portfolio_run_identity("portfolio-1", "v1", {"b": 0.4, "a": 0.6})
    assert identity.portfolio_id == "portfolio-1"
    assert identity.dataset_version == "v1"
    assert len(identity.run_id) == 16


def test_run_id_requires_matching_strategy_weights() -> None:
    with pytest.raises(ValueError, match="weights must match"):
        portfolio_run_id("p", "v", ("a", "b"), {"a": 1.0})


def test_attribute_run_delegates_to_portfolio_attribution() -> None:
    returns = pd.DataFrame({"a": [0.01, 0.0, 0.02], "b": [0.0, 0.01, -0.01]})
    result = attribute_run(returns, {"a": 0.5, "b": 0.5})
    assert result.total_return != 0.0
    assert len(result.contributions) == 2
