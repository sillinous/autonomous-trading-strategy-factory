from random import Random

import pytest

from atsf.generator import (
    mutate_indicator_period,
    mutate_position_fraction,
    mutate_signal_comparator,
    mutate_threshold,
)
from atsf.population import (
    crossover_candidate,
    evolve_population,
    mutate_candidate,
    seed_population,
    strategy_id,
)
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


def make_parent_population() -> list:
    first = make_strategy()
    second = make_strategy().model_copy(
        update={
            "name": "alternate",
            "indicators": [Indicator(name="ema", period=10)],
            "entry": Signal(all=[Condition(left="close", comparator=Comparator.GT, right=110)]),
        }
    )
    third = make_strategy().model_copy(
        update={
            "name": "third",
            "indicators": [Indicator(name="rsi", period=14)],
            "entry": Signal(all=[Condition(left="close", comparator=Comparator.GT, right=120)]),
        }
    )
    return seed_population([first, second, third])


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


def test_mutation_fails_closed_when_no_operator_is_applicable(monkeypatch):
    parent = seed_population([make_strategy()])[0]
    monkeypatch.setattr("atsf.population._applicable_mutations", lambda _: ())
    with pytest.raises(ValueError, match="no applicable mutation operators"):
        mutate_candidate(parent, Random(3))


def test_evolve_population_is_reproducible_and_unique():
    parents = make_parent_population()
    first = evolve_population(parents, 10, Random(42), crossover_rate=0.5, mutation_rate=0.5)
    second = evolve_population(parents, 10, Random(42), crossover_rate=0.5, mutation_rate=0.5)
    assert [candidate.strategy_id for candidate in first] == [candidate.strategy_id for candidate in second]
    assert len(first) == 10
    assert len({candidate.strategy_id for candidate in first}) == 10


def test_evolve_population_preserves_requested_elites():
    parents = make_parent_population()
    population = evolve_population(parents, 6, Random(8), elite_count=2)
    assert population[:2] == parents[:2]
    assert all(candidate.lineage.generation == 0 for candidate in population[:2])
    assert all(candidate.lineage.generation == 1 for candidate in population[2:])


def test_evolve_population_records_variation_lineage():
    parents = make_parent_population()
    population = evolve_population(parents, 8, Random(12), crossover_rate=1.0, mutation_rate=0.0)
    children = population
    assert all(child.lineage.parent_ids for child in children)
    assert all(child.lineage.operator == "crossover" for child in children)
    assert all(child.lineage.generation == 1 for child in children)


def test_evolve_population_validates_configuration():
    parents = make_parent_population()
    with pytest.raises(ValueError, match="target_size"):
        evolve_population(parents, 0, Random(1))
    with pytest.raises(ValueError, match="crossover_rate"):
        evolve_population(parents, 4, Random(1), crossover_rate=1.1)
    with pytest.raises(ValueError, match="mutation_rate"):
        evolve_population(parents, 4, Random(1), mutation_rate=-0.1)
    with pytest.raises(ValueError, match="variation rate"):
        evolve_population(parents, 4, Random(1), crossover_rate=0.0, mutation_rate=0.0)


def test_evolve_population_fails_closed_when_crossover_has_one_parent():
    parent = seed_population([make_strategy()])
    with pytest.raises(ValueError, match="at least two parents"):
        evolve_population(parent, 2, Random(2), crossover_rate=1.0, mutation_rate=0.0)
