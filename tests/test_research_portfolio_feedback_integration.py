import json
from types import SimpleNamespace

from atsf.portfolio_feedback import build_portfolio_feedback
from atsf.research_cycle_registry import ResearchCycleRegistry
from atsf.research_queue import ResearchReason


def test_portfolio_feedback_is_deterministic_and_cycle_serializable():
    portfolio = SimpleNamespace(
        admission_passed=True,
        portfolio_volatility=0.10,
        selection=SimpleNamespace(selected=("s1", "s2"), average_correlation=0.90),
    )
    first = build_portfolio_feedback(portfolio, generation=2)
    second = build_portfolio_feedback(portfolio, generation=2)
    assert first == second
    assert {signal.reason for signal in first.signals} == {ResearchReason.DIVERSIFICATION}
    payload = {
        "generation": first.generation,
        "signals": [signal.__dict__ for signal in first.signals],
    }
    assert json.dumps(payload, sort_keys=True, allow_nan=False)


def test_research_cycle_registry_persists_portfolio_feedback():
    import sqlite3

    registry = ResearchCycleRegistry(sqlite3.connect(":memory:"))
    record = registry.save_cycle(
        "cycle-1",
        1,
        plan={"requests": []},
        feedback={"signals": []},
        admissions=[],
        portfolio_feedback={"admission_passed": False, "signals": [{"reason": "degraded"}]},
    )
    restored = registry.get_cycle("cycle-1")
    assert restored == record
    assert json.loads(restored.portfolio_feedback_json)["admission_passed"] is False
