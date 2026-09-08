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
