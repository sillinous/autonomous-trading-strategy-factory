from types import SimpleNamespace

import pytest

from atsf.adaptive_evolution import AdaptiveEvolutionPolicy, adapt_evolution
from atsf.research_history import GenerationRecord, ResearchHistory


def history_with_stagnation(count: int) -> ResearchHistory:
    return ResearchHistory(
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
            for index in range(count + 1)
        )
    )


def test_adaptation_is_inactive_without_stagnation():
    rates = adapt_evolution(0.5, 0.5, ResearchHistory())
    assert rates.crossover_rate == 0.5
    assert rates.mutation_rate == 0.5
    assert not rates.stagnating


def test_adaptation_increases_exploration_after_stagnation():
    policy = AdaptiveEvolutionPolicy(stagnation_threshold=3)
    rates = adapt_evolution(0.5, 0.5, history_with_stagnation(3), policy)
    assert rates.crossover_rate == pytest.approx(0.45)
    assert rates.mutation_rate == pytest.approx(0.60)
    assert rates.stagnating


def test_adaptation_respects_hard_bounds():
    policy = AdaptiveEvolutionPolicy(
        mutation_step=0.4,
        crossover_step=0.4,
        max_mutation_rate=0.6,
        min_crossover_rate=0.3,
    )
    rates = adapt_evolution(0.4, 0.5, history_with_stagnation(3), policy)
    assert rates.crossover_rate == pytest.approx(0.3)
    assert rates.mutation_rate == pytest.approx(0.6)


def test_adaptation_is_deterministic_and_validates_rates():
    history = history_with_stagnation(5)
    first = adapt_evolution(0.7, 0.2, history)
    second = adapt_evolution(0.7, 0.2, history)
    assert first == second
    with pytest.raises(ValueError, match="crossover_rate"):
        adapt_evolution(1.1, 0.2, history)
    with pytest.raises(ValueError, match="mutation_rate"):
        adapt_evolution(0.2, -0.1, history)


def test_adaptive_policy_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="threshold"):
        AdaptiveEvolutionPolicy(stagnation_threshold=0)
    with pytest.raises(ValueError, match="between"):
        AdaptiveEvolutionPolicy(mutation_step=1.1)
