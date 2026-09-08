import math

import pytest

from atsf.perturbation import evaluate_parameter_perturbations
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)


def make_strategy() -> StrategySpec:
    return StrategySpec(
        name="perturbation-test",
        universe=["TEST"],
        indicators=[Indicator(name="sma", source="close", period=10)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=1.0)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=0.5)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def test_parameter_perturbations_are_reproducible():
    strategy = make_strategy()

    def evaluator(candidate: StrategySpec) -> float:
        return float(candidate.indicators[0].period or 0)

    first = evaluate_parameter_perturbations(strategy, evaluator, samples=10, seed=7)
    second = evaluate_parameter_perturbations(strategy, evaluator, samples=10, seed=7)
    assert first == second
    assert first.samples == 10
    assert len(first.strategy_ids) == 10


def test_non_finite_perturbation_scores_are_rejected_when_all_fail():
    with pytest.raises(ValueError, match="no finite"):
        evaluate_parameter_perturbations(make_strategy(), lambda _: math.nan, samples=5)
