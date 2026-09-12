from random import Random

import pytest

from atsf.generator import (
    mutate_indicator_period,
    mutate_position_fraction,
    mutate_signal_comparator,
    mutate_threshold,
)
from atsf.population import crossover_candidate, mutate_candidate, seed_population, strategy_id
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
        name="seed",
        universe=["TEST"],
        indicators=[Indicator(name="sma", period=20)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=100)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=100)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def test_seed_population_deduplicates():
    strategy = make_strategy()
    population = seed_population([strategy, strategy])
    assert len(population) == 1
    assert population[0].lineage.generation == 0
    assert population[0].lineage.parent_ids == ()


def test_strategy_id_is_stable_and_compact():
    strategy = make_strategy()
    assert strategy_id(strategy) == strategy_id(strategy.model_copy(deep=True))
    assert len(strategy_id(strategy)) == 16


def test_mutation_advances_generation_and_parentage():
    parent = seed_population([make_strategy()])[0]
    child = mutate_candidate(parent, Random(7))
    assert child.lineage.generation == 1
    assert child.lineage.parent_ids == (parent.strategy_id,)
    assert child.strategy_id != parent.strategy_id


def test_crossover_records_two_parents_and_is_reproducible():
    first = make_strategy()
    second = make_strategy().model_copy(
        update={
            "name": "alternate",
            "indicators": [Indicator(name="ema", period=10)],
            "entry": Signal(all=[Condition(left="close", comparator=Comparator.GT, right=110)]),
        }
    )
    parent_a, parent_b = seed_population([first, second])
    child_a = crossover_candidate(parent_a, parent_b, Random(21))
    child_b = crossover_candidate(parent_a, parent_b, Random(21))
    assert child_a.strategy == child_b.strategy
    assert child_a.strategy_id == child_b.strategy_id
    assert child_a.strategy_id not in {parent_a.strategy_id, parent_b.strategy_id}
    assert child_a.lineage.generation == 1
    assert child_a.lineage.parent_ids == (parent_a.strategy_id, parent_b.strategy_id)
    assert child_a.lineage.operator == "crossover"


def test_crossover_requires_distinct_parents():
    parent = seed_population([make_strategy()])[0]
    with pytest.raises(ValueError, match="distinct parents"):
        crossover_candidate(parent, parent, Random(1))


def test_each_mutation_operator_changes_the_strategy():
    strategy = make_strategy()
    mutated = [
        mutate_indicator_period(strategy, Random(1)),
        mutate_threshold(strategy, Random(2)),
        mutate_signal_comparator(strategy, Random(3)),
        mutate_position_fraction(strategy, Random(4)),
    ]
    assert all(candidate != strategy for candidate in mutated)
    assert len({candidate.model_dump_json() for candidate in mutated}) == len(mutated)


def test_mutation_operators_are_reproducible():
    strategy = make_strategy()
    operators = (
        mutate_indicator_period,
        mutate_threshold,
        mutate_signal_comparator,
        mutate_position_fraction,
    )
    for index, operator in enumerate(operators, start=1):
        first = operator(strategy, Random(index))
        second = operator(strategy, Random(index))
        assert first == second


def test_position_mutation_respects_risk_ceiling():
    strategy = make_strategy().model_copy(
        update={
            "position_sizing": PositionSizing(
                method="fixed_fraction", value=0.5, max_position=0.25
            ),
            "risk": RiskLimits(max_position=0.25),
        }
    )
    mutated = mutate_position_fraction(strategy, Random(9))
    assert mutated.position_sizing.value <= 0.25
