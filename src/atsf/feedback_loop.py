from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .lifecycle import StrategyLifecycle, StrategyState
from .monitoring import DegradationReport
from .research_queue import ResearchQueue, ResearchReason, ResearchRequest
from .research_registry import ResearchRequestStore


@dataclass(frozen=True)
class FeedbackAction:
    strategy_id: str
    state: StrategyState
    research_request: ResearchRequest | None


def process_strategy_health(
    strategy_id: str,
    lifecycle: StrategyLifecycle,
    report: DegradationReport,
    queue: ResearchQueue,
    request_store: ResearchRequestStore | None = None,
) -> FeedbackAction:
    """Transition paper health and persist/enqueue deterministic replacement work."""
    if not strategy_id:
        raise ValueError("strategy_id cannot be empty")

    previous = lifecycle.state
    state = lifecycle.apply(report)
    if previous == StrategyState.ACTIVE and state == StrategyState.DEGRADED:
        request_id = sha256(
            f"{strategy_id}:degraded:{report.observations}:{','.join(report.reasons)}".encode()
        ).hexdigest()[:16]
        request = ResearchRequest(
            request_id=request_id,
            source_strategy_id=strategy_id,
            reason=ResearchReason.DEGRADED,
            priority=0,
            constraints=("avoid_known_failure",),
        )
        if request_store is not None:
            request_store.save(request)
        return FeedbackAction(strategy_id, state, queue.enqueue(request))
    return FeedbackAction(strategy_id, state, None)
