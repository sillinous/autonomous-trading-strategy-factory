from __future__ import annotations

from dataclasses import dataclass

from .research_planner import ResearchPlan
from .research_queue import ResearchQueue, ResearchRequest


@dataclass(frozen=True)
class ResearchWorkItem:
    """A durable execution unit derived from a planned research request."""

    request_id: str
    source_strategy_id: str | None
    reason: str
    priority: int
    budget_units: int
    constraints: tuple[str, ...] = ()


def materialize_research_work(
    plan: ResearchPlan,
    *,
    queue: ResearchQueue | None = None,
) -> tuple[ResearchWorkItem, ...]:
    """Materialize a deterministic research plan into queue-backed work items.

    The planner remains pure; this boundary is responsible for turning decisions
    into executable research capacity. No brokerage or live-execution behavior is
    introduced here.
    """
    allocation_by_id = {item.request_id: item for item in plan.allocations}
    if len(allocation_by_id) != len(plan.allocations):
        raise ValueError("research allocations must have unique request IDs")

    work: list[ResearchWorkItem] = []
    for request in plan.requests:
        allocation = allocation_by_id.get(request.request_id)
        if allocation is None:
            raise ValueError(f"missing allocation for research request: {request.request_id}")
        if allocation.units <= 0:
            raise ValueError("research work must have positive budget")
        if allocation.request_id != request.request_id:
            raise ValueError("allocation/request identity mismatch")

        queued = request
        if queue is not None:
            queued = queue.enqueue(request)

        work.append(
            ResearchWorkItem(
                request_id=queued.request_id,
                source_strategy_id=queued.source_strategy_id,
                reason=queued.reason.value,
                priority=queued.priority,
                budget_units=allocation.units,
                constraints=queued.constraints,
            )
        )

    return tuple(work)


def consume_research_work(
    queue: ResearchQueue,
    work_item: ResearchWorkItem,
) -> ResearchRequest:
    """Atomically mark a materialized request complete after execution."""
    if work_item.budget_units <= 0:
        raise ValueError("research work must have positive budget")
    request = queue.complete(work_item.request_id)
    if request.priority != work_item.priority or request.source_strategy_id != work_item.source_strategy_id:
        raise ValueError("queued research request does not match work item")
    return request
