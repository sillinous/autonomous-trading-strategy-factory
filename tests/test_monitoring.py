import pandas as pd
import pytest

from atsf.monitoring import DegradationPolicy, assess_degradation, is_finite_report


def test_monitoring_detects_degradation():
    equity = pd.Series(
        [100.0] * 10 + [90.0, 80.0] + [79.0] * 10,
        index=pd.date_range("2025-01-01", periods=22),
    )
    report = assess_degradation(
        equity,
        DegradationPolicy(max_drawdown=0.10, min_return=-0.50, max_volatility=1.0, min_observations=20),
    )
    assert report.degraded
    assert "maximum drawdown breached" in report.reasons
    assert is_finite_report(report)


def test_monitoring_rejects_short_history():
    equity = pd.Series([100.0, 101.0], index=pd.date_range("2025-01-01", periods=2))
    with pytest.raises(ValueError, match="insufficient"):
        assess_degradation(equity)


def test_monitoring_rejects_non_chronological_equity():
    equity = pd.Series([100.0, 101.0, 99.0], index=pd.to_datetime(["2025-01-02", "2025-01-01", "2025-01-03"]))
    with pytest.raises(ValueError, match="chronologically"):
        assess_degradation(equity)
