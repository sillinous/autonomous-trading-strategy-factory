from atsf.generator import StrategyGenerator
from atsf.research_queue import ResearchReason, ResearchRequest


def test_generator_emits_provenance_and_deterministic_ids():
    request = ResearchRequest("req-1", "old", ResearchReason.DEGRADED, 1)
    first = StrategyGenerator().generate(request, ["TEST"])
    second = StrategyGenerator().generate(request, ["TEST"])
    assert first
    assert [candidate.candidate_id for candidate in first] == [candidate.candidate_id for candidate in second]
    assert all(candidate.request_id == "req-1" for candidate in first)
    assert all(candidate.parent_strategy_id == "old" for candidate in first)


def test_generator_honors_constraints():
    request = ResearchRequest("req-2", None, ResearchReason.DIVERSIFICATION, 1, ("trend_only",))
    candidates = StrategyGenerator().generate(request, ["TEST"])
    assert len(candidates) == 2
    assert all(candidate.mutation != "sma_slow" for candidate in candidates)
