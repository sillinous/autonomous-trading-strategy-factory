import pandas as pd
import pytest

from atsf.generator import StrategyGenerator
from atsf.research_queue import ResearchReason, ResearchRequest
from atsf.validation import ValidationPolicy
from atsf.walk_forward import walk_forward_validate


def _candidate():
    return StrategyGenerator().generate(
        ResearchRequest("wf-test", None, ResearchReason.DEGRADED, 1), ["TEST"]
    )[0]


def _data(periods=100):
    index = pd.date_range("2025-01-01", periods=periods)
    return pd.DataFrame({"close": [100.0 + i * 0.2 for i in range(periods)]}, index=index)


def test_walk_forward_produces_multiple_out_of_sample_folds():
    result = walk_forward_validate(
        _data(),
        _candidate(),
        train_size=30,
        test_size=20,
        validation_policy=ValidationPolicy(min_sharpe=-100.0, max_drawdown=1.0),
    )

    assert len(result.folds) == 4
    assert all(fold.test_start > fold.train_start for fold in result.folds)
    assert all(fold.train_end < fold.test_start for fold in result.folds)
    assert result.passed


def test_walk_forward_rejects_insufficient_data():
    with pytest.raises(ValueError, match="insufficient data"):
        walk_forward_validate(_data(10), _candidate(), 8, 5)


def test_walk_forward_rejects_explicit_zero_step():
    with pytest.raises(ValueError, match="step must be positive"):
        walk_forward_validate(_data(), _candidate(), 30, 20, step=0)


def test_walk_forward_rejects_unsorted_timestamps():
    data = _data()
    data = data.iloc[::-1]

    with pytest.raises(ValueError, match="sorted in increasing chronological order"):
        walk_forward_validate(data, _candidate(), 30, 20)


def test_walk_forward_rejects_duplicate_timestamps():
    data = _data()
    data.index = data.index.where(data.index != data.index[1], data.index[0])

    with pytest.raises(ValueError, match="unique timestamps"):
        walk_forward_validate(data, _candidate(), 30, 20)
