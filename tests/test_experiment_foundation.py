import pandas as pd
import pytest

from atsf.experiment import ExperimentSpec
from atsf.fitness import FitnessPolicy, score_strategy
from atsf.lineage import LineageRecord
from atsf.splits import chronological_split, walk_forward_windows


def test_chronological_split_has_no_overlap():
    data = pd.DataFrame({"close": range(10)}, index=pd.date_range("2024-01-01", periods=10))
    result = chronological_split(data, train_fraction=0.6, validation_fraction=0.2)
    assert len(result.train) == 6
    assert len(result.validation) == 2
    assert len(result.test) == 2
    assert result.train.index[-1] < result.validation.index[0] < result.test.index[0]


def test_walk_forward_windows_are_ordered():
    data = pd.DataFrame({"close": range(12)}, index=pd.date_range("2024-01-01", periods=12))
    windows = walk_forward_windows(data, train_size=5, validation_size=2, test_size=2)
    assert len(windows) == 3
    for window in windows:
        assert window.train.index[-1] < window.validation.index[0]
        assert window.validation.index[-1] < window.test.index[0]


def test_split_rejects_unsorted_data():
    data = pd.DataFrame({"close": [1, 2]}, index=[2, 1])
    with pytest.raises(ValueError, match="monotonically"):
        chronological_split(data)


def test_fitness_hard_gate_rejects_weak_candidate():
    result = score_strategy(0.1, 0.05)
    assert not result.eligible
    assert result.reasons


def test_fitness_accepts_robust_candidate():
    result = score_strategy(1.5, 0.1, FitnessPolicy())
    assert result.eligible
    assert result.score > 1


def test_experiment_id_is_reproducible():
    from atsf.strategy import Condition, Comparator, Signal
    from atsf.strategy import PositionSizing, RiskLimits, StrategySpec

    strategy = StrategySpec(
        name="demo",
        universe=["TEST"],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=1)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=1)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )
    a = ExperimentSpec(strategy, "prices", "v1", seed=42)
    b = ExperimentSpec(strategy, "prices", "v1", seed=42)
    assert a.experiment_id == b.experiment_id


def test_lineage_requires_parent_for_derived_strategy():
    with pytest.raises(ValueError):
        LineageRecord("child", generation=1, operator="mutation")
