import pandas as pd
import pytest

from atsf.execution_lineage import build_fill_lineage, decision_id, verify_fill_lineage
from atsf.portfolio_audit import PortfolioAuditEvent

INDEX = pd.date_range("2026-01-01", periods=4, freq="D")


def _signals():
    # position state: [True, True, False, False]
    return {"s1": (pd.Series([True, False, False, False], index=INDEX),
                   pd.Series([False, False, True, False], index=INDEX))}


def _event(sequence, action, day, price=100.0):
    return PortfolioAuditEvent(sequence, "s1", action, INDEX[day].isoformat(), 1.0, price, 0.1)


def test_decision_id_is_deterministic():
    timestamp = pd.Timestamp("2026-01-01")
    assert decision_id("s1", timestamp, "buy", "entry_signal") == decision_id("s1", timestamp, "buy", "entry_signal")
    assert decision_id("s1", timestamp, "buy", "entry_signal") != decision_id("s1", timestamp, "sell", "exit_signal")


def test_signal_fills_are_authorized_by_the_previous_bar():
    events = (_event(0, "buy", 1), _event(1, "sell", 3))
    lineage = build_fill_lineage(events, _signals(), halted=False)
    assert [item.reason for item in lineage] == ["entry_signal", "exit_signal"]
    assert verify_fill_lineage(events, lineage, _signals(), halted=False)


def test_same_bar_fill_has_no_authorizing_decision():
    """A buy on the signal bar itself would be look-ahead and must fail closed."""
    with pytest.raises(ValueError, match="no deterministic authorizing decision"):
        build_fill_lineage((_event(0, "buy", 0),), _signals(), halted=False)


def test_end_of_sample_liquidation():
    signals = {"s1": (pd.Series([True, False, False, False], index=INDEX),
                      pd.Series(False, index=INDEX))}
    events = (_event(0, "buy", 1), _event(1, "sell", 3))
    assert [item.reason for item in build_fill_lineage(events, signals, halted=False)] == [
        "entry_signal", "end_of_sample"]


def test_risk_halt_lineage_uses_actual_liquidation_timestamp():
    events = (_event(0, "buy", 1), _event(1, "sell", 2))
    lineage = build_fill_lineage(events, _signals(), halted=True, liquidation_timestamp=INDEX[2])
    assert [item.reason for item in lineage] == ["entry_signal", "risk_halt"]
    assert verify_fill_lineage(events, lineage, _signals(), halted=True, liquidation_timestamp=INDEX[2])


def test_risk_halt_without_matching_timestamp_fails_closed():
    with pytest.raises(ValueError, match="no deterministic authorizing decision"):
        build_fill_lineage((_event(0, "sell", 2),), _signals(), halted=True, liquidation_timestamp=INDEX[3])


def test_declared_stop_exit_is_authorized_and_undeclared_one_fails():
    events = (_event(0, "buy", 1), _event(1, "sell", 2))
    lineage = build_fill_lineage(events, _signals(), halted=False,
                                 risk_exits={"s1": {INDEX[2]: "stop_loss"}})
    assert [item.reason for item in lineage] == ["entry_signal", "stop_loss"]
    with pytest.raises(ValueError, match="no deterministic authorizing decision"):
        build_fill_lineage(events, _signals(), halted=False)


def test_risk_exits_reject_unknown_reasons():
    with pytest.raises(ValueError, match="risk_exits"):
        build_fill_lineage((), _signals(), halted=False, risk_exits={"s1": {INDEX[2]: "whim"}})


def test_fill_without_authorizing_signal_fails_closed():
    with pytest.raises(ValueError, match="no deterministic authorizing decision"):
        build_fill_lineage((_event(0, "buy", 3),), _signals(), halted=False)
