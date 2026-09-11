import pytest

from atsf.feedback_loop import process_strategy_health
from atsf.feedback_provenance import build_feedback_provenance
from atsf.feedback_registry import FeedbackEventStore
from atsf.lifecycle import StrategyLifecycle
from atsf.monitoring import DegradationReport
from atsf.registry import ExperimentRegistry
from atsf.research_queue import ResearchQueue


def make_event():
    report = DegradationReport(
        degraded=True,
        observations=30,
        total_return=-0.12,
        max_drawdown=-0.22,
        volatility=0.08,
        reasons=("minimum return breached",),
    )
    lifecycle = StrategyLifecycle()
    queue = ResearchQueue()
    action = process_strategy_health("strategy-1", lifecycle, report, queue)
    return build_feedback_provenance("strategy-1", action, report, previous_state="active")


def test_feedback_event_is_write_once_and_idempotent():
    registry = ExperimentRegistry()
    store = FeedbackEventStore(registry)
    event = make_event()
    store.save(event)
    store.save(event)
    assert store.get(event.event_id)["fingerprint"] == event.fingerprint
    registry.close()


def test_feedback_event_cannot_be_replaced():
    registry = ExperimentRegistry()
    store = FeedbackEventStore(registry)
    event = make_event()
    store.save(event)
    registry._connection.execute(
        "UPDATE strategy_feedback_events SET event_json = ? WHERE event_id = ?",
        ('{"fingerprint":"tampered"}', event.event_id),
    )
    registry._connection.commit()
    with pytest.raises(ValueError, match="immutable"):
        store.save(event)
    registry.close()


def test_invalid_feedback_event_is_rejected():
    registry = ExperimentRegistry()
    store = FeedbackEventStore(registry)
    event = make_event()
    invalid = event.__class__(
        event_id=event.event_id,
        strategy_id=event.strategy_id,
        previous_state=event.previous_state,
        resulting_state="halted",
        report_fingerprint=event.report_fingerprint,
        research_request_id=event.research_request_id,
        research_fingerprint=event.research_fingerprint,
        fingerprint=event.fingerprint,
    )
    with pytest.raises(ValueError, match="invalid"):
        store.save(invalid)
    registry.close()
