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
        PortfolioAuditEvent(0, "s1", "buy", index[1].isoformat(), 1.0, 100.0, 0.1),
        PortfolioAuditEvent(1, "s1", "sell", index[2].isoformat(), 1.0, 101.0, 0.1),
    )
    lineage = build_fill_lineage(events, _signals(), halted=False)
    assert [item.reason for item in lineage] == ["entry_signal", "exit_signal"]
    assert verify_fill_lineage(events, lineage, _signals(), halted=False)


def test_risk_halt_lineage_uses_actual_liquidation_timestamp():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    events = (
        PortfolioAuditEvent(0, "s1", "buy", index[1].isoformat(), 1.0, 100.0, 0.1),
        PortfolioAuditEvent(1, "s1", "sell", index[2].isoformat(), 1.0, 101.0, 0.1),
    )
    lineage = build_fill_lineage(events, _signals(), halted=True, liquidation_timestamp=index[2])
    assert [item.reason for item in lineage] == ["entry_signal", "risk_halt"]
    assert verify_fill_lineage(events, lineage, _signals(), halted=True, liquidation_timestamp=index[2])


def test_risk_halt_without_matching_timestamp_fails_closed():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    events = (PortfolioAuditEvent(0, "s1", "sell", index[1].isoformat(), 1.0, 100.0, 0.1),)
    with pytest.raises(ValueError, match="no deterministic authorizing decision"):
        build_fill_lineage(events, _signals(), halted=True, liquidation_timestamp=index[2])


def test_fill_without_authorizing_signal_fails_closed():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    events = (PortfolioAuditEvent(0, "s1", "buy", index[0].isoformat(), 1.0, 100.0, 0.1),)
    with pytest.raises(ValueError, match="no deterministic authorizing decision"):
        build_fill_lineage(events, _signals(), halted=False)


def test_fill_lineage_authorizes_end_of_sample_liquidation_from_prior_bar_state():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    signals = {"s1": (
        pd.Series([True, False, False], index=index),
        pd.Series([False, False, False], index=index),
    )}
    events = (
        PortfolioAuditEvent(0, "s1", "buy", index[1].isoformat(), 1.0, 100.0, 0.1),
        PortfolioAuditEvent(1, "s1", "sell", index[2].isoformat(), 1.0, 102.0, 0.1),
    )
    lineage = build_fill_lineage(events, signals, halted=False)
    assert [item.reason for item in lineage] == ["entry_signal", "end_of_sample"]


@pytest.mark.parametrize(
    "entry_values",
    [["False", "True", "False"], [False, None, True], [0, 1, 0]],
)
def test_fill_lineage_rejects_non_boolean_or_missing_signals(entry_values):
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    signals = {"s1": (pd.Series(entry_values, index=index), pd.Series([False, True, False], index=index))}
    with pytest.raises(ValueError, match="non-missing boolean"):
        build_fill_lineage((), signals, halted=False)


def test_fill_lineage_rejects_mismatched_entry_exit_indexes():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    signals = {"s1": (
        pd.Series([True, False, False], index=index),
        pd.Series([False, True], index=index[:2]),
    )}
    with pytest.raises(ValueError, match="indexes must match"):
        build_fill_lineage((), signals, halted=False)


def test_fill_lineage_rejects_duplicate_event_sequences():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    events = (
        PortfolioAuditEvent(0, "s1", "buy", index[1].isoformat(), 1.0, 100.0, 0.1),
        PortfolioAuditEvent(0, "s1", "sell", index[2].isoformat(), 1.0, 101.0, 0.1),
    )
    with pytest.raises(ValueError, match="sequence values must be unique"):
        build_fill_lineage(events, _signals(), halted=False)


def test_fill_lineage_rejects_unknown_execution_action():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    events = (PortfolioAuditEvent(0, "s1", "short", index[1].isoformat(), 1.0, 100.0, 0.1),)
    with pytest.raises(ValueError, match="unsupported execution action"):
        build_fill_lineage(events, _signals(), halted=False)


def test_fill_lineage_rejects_fill_timestamp_missing_from_signal_index():
    signals = _signals()
    event = PortfolioAuditEvent(0, "s1", "buy", "2026-01-10T00:00:00", 1.0, 100.0, 0.1)
    with pytest.raises(ValueError, match="absent from signals"):
        build_fill_lineage((event,), signals, halted=False)


from atsf.portfolio_audit import audit_event_id


@pytest.mark.parametrize(
    "event, message",
    [
        (PortfolioAuditEvent(True, "s1", "buy", "2026-01-01T00:00:00", 1.0, 100.0, 0.0), "sequence must be"),
        (PortfolioAuditEvent(0, "s1", "short", "2026-01-01T00:00:00", 1.0, 100.0, 0.0), "supported action"),
        (PortfolioAuditEvent(0, "s1", "buy", "not-a-date", 1.0, 100.0, 0.0), "valid ISO 8601"),
        (PortfolioAuditEvent(0, "s1", "buy", "2026-01-01T00:00:00+00:00", 1.0, 100.0, 0.0), "timezone-naive"),
        (PortfolioAuditEvent(0, "s1", "buy", "2026-01-01T00:00:00", True, 100.0, 0.0), "must be numeric"),
        (PortfolioAuditEvent(0, "s1", "buy", "2026-01-01T00:00:00", 1.0, "100", 0.0), "must be numeric"),
    ],
)
def test_audit_event_id_rejects_malformed_event_fields(event, message):
    with pytest.raises(ValueError, match=message):
        audit_event_id(event)


def test_audit_event_id_is_deterministic_for_valid_event():
    event = PortfolioAuditEvent(3, "s1", "buy", "2026-01-01T00:00:00", 1.0, 100.0, 0.25)
    assert audit_event_id(event) == audit_event_id(event)
