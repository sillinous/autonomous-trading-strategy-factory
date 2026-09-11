import pytest

from atsf.research_queue import ResearchReason, ResearchRequest
from atsf.research_registry import ResearchRequestStore
from atsf.registry import ExperimentRegistry


def make_request() -> ResearchRequest:
    return ResearchRequest(
        request_id="request-1",
        source_strategy_id="strategy-1",
        reason=ResearchReason.DEGRADED,
        priority=0,
        constraints=("avoid_known_failure",),
    )


def test_research_request_store_is_idempotent_and_round_trips() -> None:
    registry = ExperimentRegistry()
    store = ResearchRequestStore(registry)
    request = make_request()

    assert store.save(request) == request
    assert store.save(request) == request
    assert store.get(request.request_id) == request
    assert store.list_for_strategy(request.source_strategy_id) == (request,)
    registry.close()


def test_research_request_store_rejects_immutable_replacement() -> None:
    registry = ExperimentRegistry()
    store = ResearchRequestStore(registry)
    request = make_request()
    store.save(request)
    replacement = ResearchRequest(
        request_id=request.request_id,
        source_strategy_id=request.source_strategy_id,
        reason=request.reason,
        priority=1,
        constraints=request.constraints,
    )
    with pytest.raises(ValueError, match="research request is immutable"):
        store.save(replacement)
    registry.close()


def test_research_request_store_rejects_empty_lookup() -> None:
    registry = ExperimentRegistry()
    store = ResearchRequestStore(registry)
    with pytest.raises(ValueError, match="request_id cannot be empty"):
        store.get("")
    registry.close()
