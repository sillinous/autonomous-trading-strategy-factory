import pytest

from atsf.generator import StrategyGenerator
from atsf.research_queue import ResearchReason, ResearchRequest
from atsf.research_registry import ResearchRequestStore
from atsf.registry import ExperimentRegistry
from atsf.replacement_research import generate_replacements
from atsf.successor_registry import SuccessorCandidateStore


def test_successor_candidates_round_trip_and_are_immutable():
    registry = ExperimentRegistry()
    requests = ResearchRequestStore(registry)
    candidates_store = SuccessorCandidateStore(registry)
    request = ResearchRequest("request-1", "failed-1", ResearchReason.DEGRADED, 0, ("avoid_known_failure",))
    requests.save(request)
    candidate = generate_replacements(request, ["TEST"], generator=StrategyGenerator(), request_store=requests).candidates[0]
    strategy_id = registry.save_strategy(candidate.strategy)
    candidates_store.save(candidate, strategy_id)
    assert candidates_store.list_for_request(request.request_id)[0]["candidate_id"] == candidate.candidate_id
    assert candidates_store.get(candidate.candidate_id)["strategy_id"] == strategy_id
    with pytest.raises(ValueError, match="successor candidate is immutable"):
        candidates_store.save(candidate, "different-strategy")
    registry.close()
