import pandas as pd

from atsf.discovery import evaluate_candidates
from atsf.generator import StrategyGenerator
from atsf.research_queue import ResearchReason, ResearchRequest
from atsf.validation import ValidationPolicy


def test_generated_candidates_flow_through_backtest_and_validation():
    index = pd.date_range("2025-01-01", periods=80)
    data = pd.DataFrame(
        {"close": [100.0 + i * 0.25 for i in range(80)]},
        index=index,
    )
    request = ResearchRequest("discovery-1", None, ResearchReason.DEGRADED, 1)
    candidates = StrategyGenerator().generate(request, ["TEST"])
    result = evaluate_candidates(
        data,
        candidates,
        validation_policy=ValidationPolicy(min_sharpe=-100.0, max_drawdown=1.0),
    )
    assert len(result.evaluations) == len(candidates)
    assert all(item.backtest.equity.iloc[-1] > 0 for item in result.evaluations)
    assert len(result.accepted) > 0
