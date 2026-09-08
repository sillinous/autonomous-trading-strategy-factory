import numpy as np
import pandas as pd
import pytest

from atsf.allocation import AllocationPolicy, allocate_inverse_volatility


def test_inverse_volatility_allocation_favors_lower_volatility():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        {
            "low": rng.normal(0, 0.005, 40),
            "high": rng.normal(0, 0.02, 40),
        }
    )
    result = allocate_inverse_volatility(returns, ("low", "high"))
    assert result.weights["low"] > result.weights["high"]
    assert result.total_weight == pytest.approx(1.0)


def test_allocation_respects_total_exposure_and_missing_ids():
    returns = pd.DataFrame({"a": [0.01] * 20})
    result = allocate_inverse_volatility(
        returns, ("a",), AllocationPolicy(max_total_weight=0.5)
    )
    assert result.total_weight == pytest.approx(0.5)
    with pytest.raises(ValueError, match="missing strategy returns"):
        allocate_inverse_volatility(returns, ("missing",))


def test_allocation_minimum_is_preserved_without_scaling_it_away():
    returns = pd.DataFrame(
        {"low": [0.001, -0.001] * 20, "high": [0.02, -0.02] * 20}
    )
    result = allocate_inverse_volatility(
        returns,
        ("low", "high"),
        AllocationPolicy(min_weight=0.2, max_weight=0.7),
    )
    assert result.weights["low"] >= 0.2
    assert result.weights["high"] >= 0.2
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_allocation_respects_per_strategy_cap():
    returns = pd.DataFrame(
        {"stable": [0.001, -0.001] * 20, "volatile": [0.03, -0.03] * 20}
    )
    result = allocate_inverse_volatility(
        returns,
        ("stable", "volatile"),
        AllocationPolicy(max_weight=0.6),
    )
    assert result.weights["stable"] == pytest.approx(0.6)
    assert result.weights["volatile"] == pytest.approx(0.4)


def test_allocation_rejects_nonfinite_and_duplicate_ids():
    returns = pd.DataFrame({"a": [0.01] * 20})
    with pytest.raises(ValueError, match="finite"):
        allocate_inverse_volatility(returns.assign(a=np.nan), ("a",))
    with pytest.raises(ValueError, match="unique"):
        allocate_inverse_volatility(returns, ("a", "a"))
