from random import Random

from atsf.genome import StrategyGenome, crossover, genome_distance
from atsf.population import seed_population
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)


def make_strategy(period: int = 20, threshold: int = 100) -> StrategySpec:
    return StrategySpec(
        name="genome",
        universe=["TEST"],
        indicators=[Indicator(name="sma", period=period)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=threshold)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=threshold)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def test_genome_is_canonical_and_stable():
    strategy = make_strategy()
    first = StrategyGenome.from_strategy(strategy)
    second = StrategyGenome.from_strategy(strategy.model_copy(deep=True))
    assert first == second
    assert len(first.strategy_id) == 16
    assert first.as_dict() == second.as_dict()


def test_genome_distance_is_zero_for_identical_and_positive_for_changes():
    base = make_strategy()
    changed = make_strategy(period=50)
    assert genome_distance(base, base) == 0.0
    assert 0.0 < genome_distance(base, changed) <= 1.0


def test_crossover_is_reproducible_and_produces_valid_child():
    first = crossover(make_strategy(10, 90), make_strategy(50, 110), Random(17))
    second = crossover(make_strategy(10, 90), make_strategy(50, 110), Random(17))
    assert first == second
    assert first.version == 2
    assert first.metadata["evolution_operator"] == "crossover"
    assert seed_population([first])


def test_crossover_changes_lineage_identity():
    parent_a = make_strategy(10, 90)
    parent_b = make_strategy(50, 110)
    child = crossover(parent_a, parent_b, Random(3))
    assert StrategyGenome.from_strategy(child).strategy_id not in {
        StrategyGenome.from_strategy(parent_a).strategy_id,
        StrategyGenome.from_strategy(parent_b).strategy_id,
    }
