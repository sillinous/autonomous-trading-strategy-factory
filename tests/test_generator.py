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


def test_generated_candidate_compiles_with_named_indicators():
    import pandas as pd

    from atsf.signals import strategy_signals

    request = ResearchRequest("req-compile", None, ResearchReason.DEGRADED, 1)
    candidate = StrategyGenerator().generate(request, ["TEST"])[0]
    data = pd.DataFrame({"close": range(1, 101)})
    entry, exit_ = strategy_signals(data, candidate.strategy)
    assert len(entry) == len(data)
    assert len(exit_) == len(data)
    assert candidate.strategy.indicators[0].kind in {"sma", "ema", "rsi"}


def test_custom_indicator_names_require_explicit_kind():
    import pytest

    from atsf.strategy import Indicator

    with pytest.raises(ValueError, match="indicator kind is required"):
        Indicator(name="custom_indicator", period=10)
