from atsf.feedback_loop import process_strategy_health
from atsf.lifecycle import StrategyLifecycle, StrategyState
from atsf.paper_replay_feedback import process_paper_replay_health
from atsf.research_queue import ResearchQueue, ResearchReason
from atsf.research_registry import ResearchRequestStore
from atsf.registry import ExperimentRegistry


def test_failed_paper_replay_creates_replacement_research():
    registry = ExperimentRegistry()
    queue = ResearchQueue()
    lifecycle = StrategyLifecycle()
    action = process_paper_replay_health(registry, "missing-run", lifecycle, queue)
    assert action is None


def test_failed_paper_replay_with_identity_degrades_and_enqueues():
    registry = ExperimentRegistry()
    registry._connection.execute(
        "INSERT INTO portfolios(portfolio_id, definition_json) VALUES (?, ?)",
        ("portfolio-1", "{}"),
    )
    registry._connection.execute(
        "INSERT INTO portfolio_runs(run_id, portfolio_id, final_equity, halted, halt_reason) VALUES (?, ?, ?, ?, ?)",
        ("run-1", "portfolio-1", 100000.0, 0, None),
    )
    registry._connection.commit()
    from atsf.paper_admission_record import build_admission_record
    from atsf.paper_admission_store import PaperAdmissionStore

    PaperAdmissionStore(registry).save(build_admission_record("strategy-1", "run-1", "cert-1"))
    queue = ResearchQueue()
    lifecycle = StrategyLifecycle()
    request_store = ResearchRequestStore(registry)
    action = process_paper_replay_health(registry, "run-1", lifecycle, queue, request_store)
    assert action is not None
    assert action.state == StrategyState.DEGRADED
    assert action.research_request is not None
    assert action.research_request.reason == ResearchReason.DEGRADED
    assert len(queue) == 1
    assert request_store.get(action.research_request.request_id) == action.research_request
    registry.close()
