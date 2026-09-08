import numpy as np
import pandas as pd
import pytest

from atsf.portfolio_attribution import attribute_portfolio


def test_portfolio_return_contributions_reconcile():
    returns = pd.DataFrame(
        {"trend": [0.01, 0.02, -0.01], "hedge": [-0.005, 0.01, 0.005]}
    )
    result = attribute_portfolio(returns, {"trend": 0.6, "hedge": 0.4})
    contributions = sum(item.return_contribution for item in result.contributions)
    assert contributions == pytest.approx(result.total_return)
    assert result.total_return > 0


def test_risk_contributions_reconcile_to_portfolio_volatility():
    returns = pd.DataFrame(
        {"a": [0.01, -0.01] * 20, "b": [0.02, -0.02] * 20}
    )
    result = attribute_portfolio(returns, {"a": 0.5, "b": 0.5})
    risk = sum(item.risk_contribution for item in result.contributions)
    assert risk == pytest.approx(result.volatility)


def test_attribution_rejects_invalid_inputs():
    returns = pd.DataFrame({"a": [0.01] * 20})
    with pytest.raises(ValueError, match="finite"):
        attribute_portfolio(returns.assign(a=np.inf), {"a": 1.0})
    with pytest.raises(ValueError, match="gross exposure"):
        attribute_portfolio(returns, {"a": 1.1})
