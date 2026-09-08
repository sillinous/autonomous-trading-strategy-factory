import pandas as pd
import pytest

from atsf.accounting import reconcile_equity


def test_reconcile_equity_accepts_consistent_curve():
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    returns = pd.Series([0.0, 0.10, -0.05], index=index)
    equity = (1 + returns).cumprod() * 100_000
    result = reconcile_equity(equity, returns)
    assert result.passed
    assert result.max_error == 0.0


def test_reconcile_equity_rejects_inconsistent_curve():
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    returns = pd.Series([0.0, 0.10, -0.05], index=index)
    equity = pd.Series([100_000.0, 110_000.0, 120_000.0], index=index)
    result = reconcile_equity(equity, returns)
    assert not result.passed
    assert result.reasons


def test_reconcile_equity_rejects_mismatched_indexes():
    equity = pd.Series([100.0, 101.0], index=pd.date_range("2025-01-01", periods=2))
    returns = pd.Series([0.0, 0.01], index=pd.date_range("2025-01-02", periods=2))
    with pytest.raises(ValueError, match="indexes must match"):
        reconcile_equity(equity, returns)
