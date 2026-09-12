from types import SimpleNamespace

import pytest

from atsf.portfolio_feedback import build_portfolio_feedback
from atsf.research_queue import ResearchReason


def _portfolio(*, admitted=True, correlation=0.1, volatility=0.05):
    return SimpleNamespace(
        admission_passed=admitted,
        portfolio_volatility=volatility,
        selection=SimpleNamespace(
            selected=("strategy-a", "strategy-b"),
            average_correlation=correlation,
        ),
    )


def test_admitted_low_correlation_portfolio_creates_capacity_feedback():
    result = build_portfolio_feedback(_portfolio())
    assert result.admission_passed
    assert {signal.reason for signal in result.signals} == {ResearchReason.CAPACITY}
    assert all(signal.capacity_gap == 1.0 for signal in result.signals)
    assert result.signals == build_portfolio_feedback(_portfolio()).signals


def test_high_correlation_portfolio_requests_diversification():
    result = build_portfolio_feedback(_portfolio(correlation=0.9))
    assert {signal.reason for signal in result.signals} == {ResearchReason.DIVERSIFICATION}
    assert all(signal.novelty <= 0.1 for signal in result.signals)


def test_risk_failed_portfolio_degrades_selected_strategies():
    result = build_portfolio_feedback(_portfolio(admitted=False, volatility=1.5))
    assert not result.admission_passed
    assert {signal.reason for signal in result.signals} == {ResearchReason.DEGRADED}
    assert all(signal.capacity_gap == 0.0 for signal in result.signals)
    assert all(signal.uncertainty == 1.0 for signal in result.signals)


def test_invalid_portfolio_feedback_fails_closed():
    with pytest.raises(ValueError, match="volatility"):
        build_portfolio_feedback(_portfolio(volatility=float("nan")))
    with pytest.raises(ValueError, match="correlation"):
        build_portfolio_feedback(_portfolio(correlation=float("inf")))
    with pytest.raises(ValueError, match="selected"):
        build_portfolio_feedback(SimpleNamespace(
            admission_passed=True,
            portfolio_volatility=0.1,
            selection=SimpleNamespace(selected=(), average_correlation=0.1),
        ))
