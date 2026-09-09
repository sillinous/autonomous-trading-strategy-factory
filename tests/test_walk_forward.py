import pandas as pd

from atsf.generator import StrategyGenerator
from atsf.research_queue import ResearchReason, ResearchRequest
from atsf.validation import ValidationPolicy
from atsf.walk_forward import walk_forward_validate


def test_walk_forward_produces_multiple_out_of_sample_folds():
    index = pd.date_range("2025-01-01", periods=100)
    data = pd.DataFrame({"close": [100.0 + i * 0.2 for i in range(100)]}, index=index)
    request = ResearchRequest("wf-1", None, ResearchReason.DEGRADED, 1)
    candidate = StrategyGenerator().generate(request, ["TEST"])[0]

    result = walk_forward_validate(
        data,
        candidate,
        train_size=30,
        test_size=20,
        validation_policy=ValidationPolicy(min_sharpe=-100.0, max_drawdown=1.0),
    )

    assert len(result.folds) == 4
    assert all(fold.test_start > fold.train_start for fold in result.folds)
    assert result.passed


def test_walk_forward_rejects_insufficient_data():
    index = pd.date_range("2025-01-01", periods=10)
    data = pd.DataFrame({"close": [100.0] * 10}, index=index)
    candidate = StrategyGenerator().generate(
        ResearchRequest("wf-2", None, ResearchReason.DEGRADED, 1), ["TEST"]
    )[0]

    try:
        walk_forward_validate(data, candidate, 8, 5)
    except ValueError as exc:
        assert "insufficient data" in str(exc)
    else:
        raise AssertionError("expected insufficient-data failure")
