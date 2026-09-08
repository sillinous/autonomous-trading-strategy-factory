import pandas as pd

from atsf.fitness import FitnessPolicy
from atsf.population import seed_population
from atsf.registry import ExperimentRegistry
from atsf.research import run_research
from atsf.strategy import Comparator, Condition, Indicator, PositionSizing, RiskLimits, Signal, StrategySpec


def make_strategy() -> StrategySpec:
    return StrategySpec(
        name="research-seed",
        universe=["TEST"],
        indicators=[Indicator(name="sma", source="close", period=5)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def make_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=40)
    close = [100 + (i % 9) for i in range(40)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "volume": [1000] * 40,
        },
        index=index,
    )


def test_research_run_is_reproducible():
    kwargs = dict(
        seeds=[make_strategy()],
        data=make_data(),
        generations=2,
        population_size=4,
        survivor_count=2,
        seed=17,
        fitness_policy=FitnessPolicy(min_sharpe=-1.0, max_drawdown=1.0),
    )
    first = run_research(**kwargs)
    second = run_research(**kwargs)
    assert first.dataset_version == second.dataset_version
    assert [candidate.strategy_id for candidate in first.final_population] == [
        candidate.strategy_id for candidate in second.final_population
    ]
    assert len(first.generations) == 2
    assert all(len(result.next_population) == 4 for result in first.generations)


def test_research_persists_experiment_evidence():
    registry = ExperimentRegistry()
    result = run_research(
        [make_strategy()],
        make_data(),
        generations=1,
        population_size=2,
        survivor_count=1,
        seed=9,
        dataset_id="fixture-prices",
        registry=registry,
        fitness_policy=FitnessPolicy(min_sharpe=-1.0, max_drawdown=1.0),
    )
    experiments = registry.list_experiments("fixture-prices")
    assert experiments
    evidence = registry.get_evaluation_evidence(experiments[0]["experiment_id"])
    assert evidence is not None
    assert "walk_forward" in evidence
    assert "monte_carlo" in evidence
    assert "perturbation" in evidence
    assert "regime" in evidence
    assert "promotion" in evidence
    assert result.dataset_id == "fixture-prices"
    registry.close()


def test_seed_population_starts_at_generation_zero():
    population = seed_population([make_strategy()])
    assert population[0].lineage.generation == 0
