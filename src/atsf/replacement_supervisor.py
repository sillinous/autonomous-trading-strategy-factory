from __future__ import annotations

from dataclasses import dataclass

from .lifecycle import StrategyLifecycleStage
from .lifecycle_store import LifecycleStore
from .replacement_cycle import ReplacementCycleStore
from .research_queue import ResearchReason, ResearchQueue
from .research_registry import ResearchRequestStore
from .registry import ExperimentRegistry


@dataclass(frozen=True)
class ReplacementWork:
    request_id: str
    strategy_id: str
    reason: ResearchReason
    status: str


def discover_replacement_work(
    registry: ExperimentRegistry,
    *,
    limit: int = 10,
) -> tuple[ReplacementWork, ...]:
    """Discover durable replacement work without mutating the research system.

    Completed cycles are reported as ``completed`` so a restarted worker cannot
    accidentally execute the same request twice. Only durably persisted,
    DEGRADED-source requests are returned as actionable work.
    """
    if limit < 1:
        raise ValueError("limit must be positive")

    requests = ResearchRequestStore(registry)
    cycles = ReplacementCycleStore(registry)
    lifecycle = LifecycleStore(registry._connection)
    discovered: list[ReplacementWork] = []

    for request in requests.list_all():
        if len(discovered) >= limit:
            break
        if request.reason is not ResearchReason.DEGRADED:
            continue
        source = lifecycle.get(request.source_strategy_id)
        if source is None:
            continue
        if cycles.get(request.request_id) is not None:
            discovered.append(
                ReplacementWork(
                    request_id=request.request_id,
                    strategy_id=request.source_strategy_id,
                    reason=request.reason,
                    status="completed",
                )
            )
            continue
        if source.stage is StrategyLifecycleStage.DEGRADED:
            discovered.append(
                ReplacementWork(
                    request_id=request.request_id,
                    strategy_id=request.source_strategy_id,
                    reason=request.reason,
                    status="ready",
                )
            )
    return tuple(discovered)
