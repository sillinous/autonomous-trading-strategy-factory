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


@pytest.mark.parametrize("bad_tolerance", [True, "0.1", float("nan"), float("inf"), -1.0])
def test_reconcile_equity_rejects_invalid_tolerance(bad_tolerance):
    index = pd.date_range("2025-01-01", periods=2, freq="D")
    values = pd.Series([100.0, 101.0], index=index)
    with pytest.raises(ValueError, match="tolerance"):
        reconcile_equity(values, values.pct_change().fillna(0.0), tolerance=bad_tolerance)


@pytest.mark.parametrize(
    "bad_values",
    [
        [True, False],
        ["100", "101"],
        [100 + 1j, 101 + 1j],
    ],
)
def test_reconcile_equity_rejects_non_real_numeric_series(bad_values):
    index = pd.date_range("2025-01-01", periods=2, freq="D")
    equity = pd.Series(bad_values, index=index)
    returns = pd.Series([0.0, 0.01], index=index)
    with pytest.raises(ValueError, match="real numeric"):
        reconcile_equity(equity, returns)


def test_reconcile_equity_rejects_duplicate_indexes():
    index = pd.to_datetime(["2025-01-01", "2025-01-01"])
    equity = pd.Series([100.0, 101.0], index=index)
    returns = pd.Series([0.0, 0.01], index=index)
    with pytest.raises(ValueError, match="indexes must be unique"):
        reconcile_equity(equity, returns)


def test_reconcile_equity_rejects_unordered_indexes():
    index = pd.to_datetime(["2025-01-02", "2025-01-01"])
    equity = pd.Series([100.0, 101.0], index=index)
    returns = pd.Series([0.0, 0.01], index=index)
    with pytest.raises(ValueError, match="indexes must be ordered"):
        reconcile_equity(equity, returns)


def test_reconcile_equity_requires_series_inputs():
    with pytest.raises(ValueError, match="pandas Series"):
        reconcile_equity([100.0, 101.0], pd.Series([0.0, 0.01]))
