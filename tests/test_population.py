from random import Random

from atsf.generator import (
    mutate_indicator_period,
    mutate_position_fraction,
    mutate_signal_comparator,
    mutate_threshold,
)
from atsf.population import mutate_candidate, seed_population, strategy_id
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
