from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from random import Random

from .population import Candidate, mutate_candidate
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
    """Materialize a deterministic research plan into queue-backed work."""
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

        queued = request if queue is None else queue.enqueue(request)
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


def spawn_research_candidates(
    work_item: ResearchWorkItem,
    population: tuple[Candidate, ...],
    *,
    seed: int = 0,
) -> tuple[Candidate, ...]:
    """Spawn at most ``budget_units`` deterministic replacement candidates.

    This is deliberately limited to candidate generation. Backtesting, robustness,
    promotion, and paper execution remain downstream gates. A missing source is
    fail-closed rather than silently selecting an arbitrary parent.
    """
    if work_item.budget_units <= 0:
        raise ValueError("research work must have positive budget")
    if not isinstance(seed, int):
        raise TypeError("research seed must be an integer")
    if not population:
        raise ValueError("research execution requires a non-empty population")

    source = None
    if work_item.source_strategy_id is not None:
        source = next(
            (candidate for candidate in population if candidate.strategy_id == work_item.source_strategy_id),
            None,
        )
        if source is None:
            raise KeyError(f"unknown source strategy: {work_item.source_strategy_id}")
    elif work_item.reason == "diversification":
        raise ValueError("diversification work requires an explicit source strategy")
    else:
        raise ValueError("research work without a source strategy cannot spawn replacements")

    rng = Random(seed)
    seen = {candidate.strategy_id for candidate in population}
    spawned: list[Candidate] = []
    attempts = 0
    max_attempts = work_item.budget_units * 4
    while len(spawned) < work_item.budget_units and attempts < max_attempts:
        attempts += 1
        candidate = mutate_candidate(source, rng)
        if candidate.strategy_id in seen:
            continue
        seen.add(candidate.strategy_id)
        spawned.append(candidate)

    return tuple(spawned)


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
