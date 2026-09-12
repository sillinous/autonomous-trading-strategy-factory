from __future__ import annotations

from .control_plane import StrategyControlPlane
from .feedback_loop import FeedbackAction, process_strategy_health
from .lifecycle import StrategyLifecycle
from .monitoring import DegradationReport
from .paper_replay import verify_paper_replay
from .registry import ExperimentRegistry
from .research_queue import ResearchQueue
from .research_registry import ResearchRequestStore


def process_paper_replay_health(
    registry: ExperimentRegistry,
    run_id: str,
    lifecycle: StrategyLifecycle,
    queue: ResearchQueue,
    request_store: ResearchRequestStore | None = None,
    control_plane: StrategyControlPlane | None = None,
) -> FeedbackAction | None:
    """Convert a failed durable PAPER replay into deterministic replacement research."""
    result = verify_paper_replay(registry, run_id)
    if result.replayable:
        return None
    if not result.strategy_id:
        raise ValueError("paper replay failure has no strategy identity")
    report = DegradationReport(
        degraded=True,
        observations=0,
        reasons=("paper replay verification failed", *result.reasons),
    )
    return process_strategy_health(
        result.strategy_id,
        lifecycle,
        report,
        queue,
        request_store,
        control_plane,
    )
