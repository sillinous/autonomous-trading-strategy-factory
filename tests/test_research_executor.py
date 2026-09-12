from __future__ import annotations

from types import SimpleNamespace

import pytest

from atsf.research_budget import ResearchAllocation
from atsf.research_executor import consume_research_work, materialize_research_work
from atsf.research_planner import ResearchPlan
from atsf.research_queue import ResearchQueue, ResearchReason, ResearchRequest


def _plan() -> ResearchPlan:
    requests = (
        ResearchRequest("research:degraded:s1", "s1", ResearchReason.DEGRADED, 900),
        ResearchRequest("research:capacity:s2", "s2", ResearchReason.CAPACITY, 500),
    )
    allocations = (
        ResearchAllocation("research:degraded:s1", 60, 0.9),
        ResearchAllocation("research:capacity:s2", 40, 0.5),
    )
    return ResearchPlan(generation=4, requests=requests, allocations=allocations)


def test_materialization_is_deterministic_and_enqueues_requests() -> None:
    queue = ResearchQueue()
    first = materialize_research_work(_plan(), queue=queue)
    second = materialize_research_work(_plan(), queue=queue)

    assert first == second
    assert [item.request_id for item in first] == list(
        request.request_id for request in _plan().requests
    )
    assert [item.budget_units for item in first] == [60, 40]
    assert len(queue) == 2


def test_missing_allocation_fails_closed() -> None:
    plan = ResearchPlan(
        generation=1,
        requests=(ResearchRequest("r", None, ResearchReason.CAPACITY, 1),),
        allocations=(),
    )

    with pytest.raises(ValueError, match="missing allocation"):
        materialize_research_work(plan)


def test_consume_completes_matching_request() -> None:
    queue = ResearchQueue()
    work = materialize_research_work(_plan(), queue=queue)

    completed = consume_research_work(queue, work[0])

    assert completed.request_id == work[0].request_id
    assert len(queue) == 1


def test_consume_rejects_unknown_request() -> None:
    queue = ResearchQueue()
    work = materialize_research_work(_plan(), queue=queue)
    consume_research_work(queue, work[0])

    with pytest.raises(KeyError, match="unknown research request"):
        consume_research_work(queue, work[0])
