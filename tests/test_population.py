from random import Random

from atsf.population import mutate_candidate, seed_population
from atsf.strategy import Condition, Comparator, Indicator, PositionSizing, RiskLimits, Signal, StrategySpec


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


def test_mutation_advances_generation_and_parentage():
    parent = seed_population([make_strategy()])[0]
    child = mutate_candidate(parent, Random(7))
    assert child.lineage.generation == 1
    assert child.lineage.parent_ids == (parent.strategy_id,)
    assert child.strategy_id != parent.strategy_id
