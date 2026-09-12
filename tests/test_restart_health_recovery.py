from atsf.control_plane import StrategyControlPlane
from atsf.lifecycle import StrategyLifecycle, StrategyState
from atsf.monitoring import DegradationReport
from atsf.registry import ExperimentRegistry
from atsf.research_queue import ResearchQueue
from atsf.research_registry import ResearchRequestStore
from atsf.feedback_loop import process_strategy_health


def test_degraded_health_restores_after_sqlite_restart(tmp_path):
    path = tmp_path / "atsf.sqlite3"
    first_registry = ExperimentRegistry(path)
    first_plane = StrategyControlPlane(first_registry)
    first_lifecycle = StrategyLifecycle()
    first_queue = ResearchQueue()
    first_store = ResearchRequestStore(first_registry)

    first = process_strategy_health(
        "strategy-1",
        first_lifecycle,
        DegradationReport(True, 0, ("replay failed",)),
        first_queue,
        first_store,
        first_plane,
    )
    assert first.state == StrategyState.DEGRADED
    assert first.research_request is not None
    first_registry.close()

    second_registry = ExperimentRegistry(path)
    second_plane = StrategyControlPlane(second_registry)
    second_lifecycle = StrategyLifecycle()
    restored = second_plane.restore("strategy-1", second_lifecycle)

    assert restored is not None
    assert restored.state == StrategyState.DEGRADED
    assert second_lifecycle.state == StrategyState.DEGRADED

    second_queue = ResearchQueue()
    second_store = ResearchRequestStore(second_registry)
    second = process_strategy_health(
        "strategy-1",
        second_lifecycle,
        DegradationReport(True, 0, ("replay failed",)),
        second_queue,
        second_store,
        second_plane,
    )
    assert second.state == StrategyState.DEGRADED
    assert second.research_request is None
    assert len(second_store.list_for_strategy("strategy-1")) == 1
    assert len(second_plane.events("strategy-1")) == 1
    second_registry.close()
