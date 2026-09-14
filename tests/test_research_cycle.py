from types import SimpleNamespace

import pandas as pd
import pytest

from atsf.adaptive_evolution import AdaptiveEvolutionPolicy
from atsf.research_cycle import ResearchCyclePolicy, run_research_cycle
from atsf.research_history import GenerationRecord, ResearchHistory
from atsf.selection import SelectionPolicy
from tests.test_population import make_parent_population


def fake_evaluation(candidate):
    baseline = pd.Series([1.0, 100.0])
    return SimpleNamespace(
        candidate_id=candidate.strategy_id,
        fitness=SimpleNamespace(score=1.0),
        backtest=SimpleNamespace(max_drawdown=0.10),
        monte_carlo=SimpleNamespace(pass_rate=0.90),
        regime=SimpleNamespace(score=-0.01),
        robustness=SimpleNamespace(
            baseline=SimpleNamespace(equity=baseline),
            scenarios=(("stress", SimpleNamespace(equity=pd.Series([1.0, 95.0]))),),
        ),
        promotion=SimpleNamespace(eligible=True),
        validation_passed=True,
    )


def policy():
    return ResearchCyclePolicy(
        selection=SelectionPolicy(population_size=2, preserve_diversity=False),
        crossover_rate=0.5,
        mutation_rate=0.5,
        elite_count=1,
    )


def test_research_cycle_evaluates_selects_and_evolves():
    population = make_parent_population()
    result = run_research_cycle(population, fake_evaluation, generation=0, seed=7, policy=policy())

    assert len(result.evaluations) == len(population)
    assert len(result.selected_parents) == 2
    assert len(result.next_population) == 2
    assert result.metrics.generation == 0
    assert result.metrics.candidate_count == 3
    assert result.metrics.selected_count == 2
    assert result.metrics.promoted_count == 3
    assert result.metrics.crossover_rate == 0.5
    assert result.metrics.mutation_rate == 0.5
    assert not result.metrics.stagnating
    assert result.next_population[0].strategy_id == result.selected_parents[0].strategy_id


def test_research_cycle_is_reproducible():
    population = make_parent_population()
    first = run_research_cycle(population, fake_evaluation, generation=3, seed=11, policy=policy())
    second = run_research_cycle(population, fake_evaluation, generation=3, seed=11, policy=policy())

    assert [item.strategy_id for item in first.next_population] == [
        item.strategy_id for item in second.next_population
    ]
    assert first.metrics == second.metrics


def test_research_cycle_adapts_rates_from_stagnating_history():
    history = ResearchHistory(
        records=tuple(
            GenerationRecord(
                generation=index,
                candidate_count=2,
                selected_count=2,
                promoted_count=2,
                best_fitness=1.0,
                mean_fitness=0.8,
                mean_genome_distance=0.5,
                min_genome_distance=0.2,
                max_genome_distance=0.8,
                unique_strategy_count=2,
                new_strategy_count=1,
                best_strategy_id=f"strategy-{index}",
                generations_without_improvement=index,
            )
            for index in range(4)
        )
    )
    result = run_research_cycle(
        make_parent_population(),
        fake_evaluation,
        generation=4,
        seed=7,
        policy=policy(),
        history=history,
        adaptive_policy=AdaptiveEvolutionPolicy(stagnation_threshold=3),
    )
    assert result.metrics.stagnating
    assert result.metrics.crossover_rate == pytest.approx(0.45)
    assert result.metrics.mutation_rate == pytest.approx(0.60)


def test_research_cycle_rejects_duplicate_population_ids():
    population = make_parent_population()
    duplicate = [population[0], population[0]]
    with pytest.raises(ValueError, match="unique"):
        run_research_cycle(duplicate, fake_evaluation, generation=0, seed=1, policy=policy())


def test_research_cycle_validates_generation_and_policy():
    with pytest.raises(ValueError, match="variation rate"):
        ResearchCyclePolicy(
            selection=SelectionPolicy(population_size=2),
            crossover_rate=0.0,
            mutation_rate=0.0,
        )
    with pytest.raises(ValueError, match="generation"):
        run_research_cycle(
            make_parent_population(), fake_evaluation, generation=-1, seed=1, policy=policy()
        )
