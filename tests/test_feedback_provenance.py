from atsf.feedback_loop import process_strategy_health
from atsf.feedback_provenance import build_feedback_provenance, verify_feedback_provenance
from atsf.lifecycle import StrategyLifecycle
from atsf.monitoring import DegradationReport
from atsf.research_queue import ResearchQueue


def report(*reasons: str) -> DegradationReport:
    return DegradationReport(
        total_return=-0.12,
        max_drawdown=0.22,
        volatility=0.08,
        observations=30,
        reasons=tuple(reasons),
    )


def test_feedback_provenance_binds_health_and_research_request():
    lifecycle = StrategyLifecycle()
    queue = ResearchQueue()
    action = process_strategy_health("strategy-1", lifecycle, report("return below minimum"), queue)
    event = build_feedback_provenance(
        "strategy-1", action, report("return below minimum"), previous_state="active"
    )
    assert event.research_request_id is not None
    assert event.research_fingerprint is not None
    assert verify_feedback_provenance(event)


def test_feedback_provenance_detects_tampering():
    lifecycle = StrategyLifecycle()
    queue = ResearchQueue()
    action = process_strategy_health("strategy-1", lifecycle, report("return below minimum"), queue)
    event = build_feedback_provenance("strategy-1", action, report("return below minimum"), previous_state="active")
    tampered = event.__class__(
        event_id=event.event_id,
        strategy_id=event.strategy_id,
        previous_state=event.previous_state,
        resulting_state="halted",
        report_fingerprint=event.report_fingerprint,
        research_request_id=event.research_request_id,
        research_fingerprint=event.research_fingerprint,
        fingerprint=event.fingerprint,
    )
    assert not verify_feedback_provenance(tampered)
