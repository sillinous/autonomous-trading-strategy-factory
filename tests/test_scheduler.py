import pandas as pd

from atsf.population import seed_population
from atsf.scheduler import evolve_generation
from atsf.strategy import Comparator, Condition, Indicator, PositionSizing, RiskLimits, Signal, StrategySpec


def make_strategy() -> StrategySpec:
    return StrategySpec(
        name="seed",
        universe=["TEST"],
        indicators=[Indicator(name="sma", period=5)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def make_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=30)
    return pd.DataFrame({"close": [100 + (i % 7) for i in range(30)]}, index=index)


def test_evolve_generation_preserves_population_size_and_lineage():
    population = seed_population([make_strategy()])
    result = evolve_generation(
        population,
        make_data(),
        "fixture",
        "v1",
        target_size=5,
        survivor_count=1,
        seed=11,
    )
    assert len(result.next_population) == 5
    assert len(result.survivors) == 1
    assert result.generation == 1
    assert all(candidate.lineage.generation == 1 for candidate in result.next_population[1:])


def test_evolve_generation_is_reproducible():
    population = seed_population([make_strategy()])
    first = evolve_generation(population, make_data(), "fixture", "v1", 4, 1, seed=23)
    second = evolve_generation(population, make_data(), "fixture", "v1", 4, 1, seed=23)
    assert [candidate.strategy_id for candidate in first.next_population] == [
        candidate.strategy_id for candidate in second.next_population
    ]
