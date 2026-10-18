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



def _candidate_for_validation():
    return StrategyGenerator().generate(
        ResearchRequest("wf-input-validation", None, ResearchReason.DEGRADED, 1), ["TEST"]
    )[0]


def test_walk_forward_rejects_explicit_zero_step():
    index = pd.date_range("2025-01-01", periods=30)
    data = pd.DataFrame({"close": [100.0 + i for i in range(30)]}, index=index)

    try:
        walk_forward_validate(data, _candidate_for_validation(), 10, 5, step=0)
    except ValueError as exc:
        assert "step must be positive" in str(exc)
    else:
        raise AssertionError("expected explicit zero step to be rejected")


def test_walk_forward_rejects_unsorted_timestamps():
    index = pd.DatetimeIndex(["2025-01-02", "2025-01-01"] + list(pd.date_range("2025-01-03", periods=28)))
    data = pd.DataFrame({"close": [100.0 + i for i in range(30)]}, index=index)

    try:
        walk_forward_validate(data, _candidate_for_validation(), 10, 5)
    except ValueError as exc:
        assert "timestamps must be sorted and unique" in str(exc)
    else:
        raise AssertionError("expected unsorted timestamps to be rejected")


def test_walk_forward_rejects_non_finite_close_prices():
    index = pd.date_range("2025-01-01", periods=30)
    prices = [100.0 + i for i in range(30)]
    prices[12] = float("inf")
    data = pd.DataFrame({"close": prices}, index=index)

    try:
        walk_forward_validate(data, _candidate_for_validation(), 10, 5)
    except ValueError as exc:
        assert "close prices must be finite and positive" in str(exc)
    else:
        raise AssertionError("expected non-finite close to be rejected")
