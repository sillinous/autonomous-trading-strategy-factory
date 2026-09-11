import pandas as pd
import pytest

from atsf.execution_lineage import build_fill_lineage, decision_id, verify_fill_lineage
from atsf.portfolio_audit import PortfolioAuditEvent


def _signals():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    return {"s1": (pd.Series([True, False, False], index=index), pd.Series([False, True, False], index=index))}


def test_decision_id_is_deterministic():
    timestamp = pd.Timestamp("2026-01-01")
    assert decision_id("s1", timestamp, "buy", "entry_signal") == decision_id("s1", timestamp, "buy", "entry_signal")
    assert decision_id("s1", timestamp, "buy", "entry_signal") != decision_id("s1", timestamp, "sell", "exit_signal")


def test_fill_lineage_connects_signals_and_end_of_sample():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    events = (
        PortfolioAuditEvent(0, "s1", "buy", index[0].isoformat(), 1.0, 100.0, 0.1),
        PortfolioAuditEvent(1, "s1", "sell", index[1].isoformat(), 1.0, 101.0, 0.1),
        PortfolioAuditEvent(2, "s1", "sell", index[2].isoformat(), 1.0, 102.0, 0.1),
    )
    lineage = build_fill_lineage(events, _signals(), halted=False)
    assert [item.reason for item in lineage] == ["entry_signal", "exit_signal", "end_of_sample"]
    assert verify_fill_lineage(events, lineage, _signals(), halted=False)


def test_risk_halt_lineage_uses_actual_liquidation_timestamp():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    events = (
        PortfolioAuditEvent(0, "s1", "buy", index[0].isoformat(), 1.0, 100.0, 0.1),
        PortfolioAuditEvent(1, "s1", "sell", index[1].isoformat(), 1.0, 101.0, 0.1),
    )
    lineage = build_fill_lineage(events, _signals(), halted=True, liquidation_timestamp=index[1])
    assert [item.reason for item in lineage] == ["entry_signal", "risk_halt"]
    assert verify_fill_lineage(events, lineage, _signals(), halted=True, liquidation_timestamp=index[1])


def test_risk_halt_without_matching_timestamp_fails_closed():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    events = (PortfolioAuditEvent(0, "s1", "sell", index[1].isoformat(), 1.0, 100.0, 0.1),)
    with pytest.raises(ValueError, match="no deterministic authorizing decision"):
        build_fill_lineage(events, _signals(), halted=True, liquidation_timestamp=index[2])


def test_fill_without_authorizing_signal_fails_closed():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    events = (PortfolioAuditEvent(0, "s1", "buy", index[1].isoformat(), 1.0, 100.0, 0.1),)
    with pytest.raises(ValueError, match="no deterministic authorizing decision"):
        build_fill_lineage(events, _signals(), halted=False)
