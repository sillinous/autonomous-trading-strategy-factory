from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_store import LifecycleStore
from atsf.replacement_cycle import ReplacementCycleStore
from atsf.replacement_supervisor import discover_replacement_work
from atsf.research_queue import ResearchReason, ResearchRequest
from atsf.research_registry import ResearchRequestStore
from atsf.registry import ExperimentRegistry


def test_discover_replacement_work_is_deterministic_and_restart_safe():
    registry = ExperimentRegistry()
    lifecycle = LifecycleStore(registry._connection)
    lifecycle.save("degraded-a", StrategyLifecycleStage.DEGRADED, reason="health failure")
    lifecycle.save("paper-b", StrategyLifecycleStage.PAPER, reason="still healthy")

    requests = ResearchRequestStore(registry)
    requests.save(ResearchRequest("request-a", "degraded-a", ResearchReason.DEGRADED, 10, ()))
    requests.save(ResearchRequest("request-b", "paper-b", ResearchReason.DEGRADED, 1, ()))

    first = discover_replacement_work(registry)
    second = discover_replacement_work(registry)
    assert first == second
    assert [(item.request_id, item.status) for item in first] == [("request-a", "ready")]
    registry.close()


def test_discover_replacement_work_marks_completed_cycles_without_reexecution():
    registry = ExperimentRegistry()
    lifecycle = LifecycleStore(registry._connection)
    lifecycle.save("degraded-a", StrategyLifecycleStage.DEGRADED, reason="health failure")
    request = ResearchRequest("request-a", "degraded-a", ResearchReason.DEGRADED, 10, ())
    ResearchRequestStore(registry).save(request)
    cycle_store = ReplacementCycleStore(registry)
    cycle_store._connection.execute(
        "INSERT INTO replacement_cycles(cycle_id, request_id, source_strategy_id, cycle_json) VALUES (?, ?, ?, ?)",
        ("cycle-a", "request-a", "degraded-a", "{}"),
    )
    cycle_store._connection.commit()

    work = discover_replacement_work(registry)
    assert work[0].status == "completed"
    registry.close()
