import pytest

from atsf.generator import StrategyGenerator
from atsf.replacement_research import generate_replacements
from atsf.research_queue import ResearchReason, ResearchRequest
from atsf.research_registry import ResearchRequestStore
from atsf.registry import ExperimentRegistry


def request() -> ResearchRequest:
    return ResearchRequest(
        request_id="replacement-1",
        source_strategy_id="failed-strategy",
        reason=ResearchReason.DEGRADED,
        priority=0,
        constraints=("avoid_known_failure",),
    )


def test_replacement_research_requires_persisted_request_when_store_is_supplied():
    registry = ExperimentRegistry()
    store = ResearchRequestStore(registry)
    with pytest.raises(ValueError, match="not persisted"):
        generate_replacements(request(), ["TEST"], request_store=store)
    registry.close()


def test_replacement_research_generates_typed_successors_with_lineage():
    registry = ExperimentRegistry()
    store = ResearchRequestStore(registry)
    original = request()
    store.save(original)
    result = generate_replacements(original, ["TEST"], generator=StrategyGenerator(), request_store=store)
    assert len(result.candidates) == 3
    assert all(candidate.request_id == original.request_id for candidate in result.candidates)
    assert all(candidate.parent_strategy_id == original.source_strategy_id for candidate in result.candidates)
    assert len({candidate.candidate_id for candidate in result.candidates}) == 3
    assert all(candidate.strategy.universe == ["TEST"] for candidate in result.candidates)
    registry.close()


def test_replacement_constraints_are_honored():
    registry = ExperimentRegistry()
    store = ResearchRequestStore(registry)
    constrained = ResearchRequest(
        request_id="replacement-2",
        source_strategy_id="failed-strategy",
        reason=ResearchReason.DEGRADED,
        priority=0,
        constraints=("trend_only",),
    )
    store.save(constrained)
    result = generate_replacements(constrained, ["TEST"], request_store=store)
    assert len(result.candidates) == 2
    assert all(candidate.mutation != "sma_slow" for candidate in result.candidates)
    registry.close()
