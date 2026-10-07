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
    names = {candidate.mutation for candidate in candidates}
    assert "sma_slow" not in names
    assert not names & {"rsi_reversion", "bollinger_reversion"}
    assert {"sma_fast", "ema_fast", "donchian_breakout", "roc_momentum", "macd_cross"} <= names


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


def test_every_archetype_compiles_and_trades_on_realistic_data():
    import numpy as np
    import pandas as pd

    from atsf.backtest import run_long_signal_backtest
    from atsf.generator import ARCHETYPES, archetype_strategy
    from atsf.signals import strategy_position

    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 2000)))
    data = pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996, "close": close},
                        index=pd.bdate_range("2015-01-01", periods=2000))
    for name in ARCHETYPES:
        spec = archetype_strategy(name, ["TEST"])
        result = run_long_signal_backtest(data, strategy_position(data, spec), spec)
        assert len(result.trades) > 0, name


def test_comparator_mutation_preserves_direction():
    from random import Random

    from atsf.generator import mutate_signal_comparator
    from atsf.strategy import Comparator

    request = ResearchRequest("req-3", None, ResearchReason.DIVERSIFICATION, 1)
    base = StrategyGenerator().generate(request, ["TEST"])[0].strategy
    bullish = {Comparator.GT, Comparator.GTE, Comparator.CROSSES_ABOVE}
    for seed in range(50):
        mutated = mutate_signal_comparator(base, Random(seed))
        assert mutated.entry.all[0].comparator in bullish - {base.entry.all[0].comparator}
