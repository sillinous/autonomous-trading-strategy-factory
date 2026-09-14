from types import SimpleNamespace

import pandas as pd
import pytest

from atsf.research_cycle import ResearchCyclePolicy
from atsf.research_runner import ResearchRunPolicy, run_research
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


def cycle_policy():
    return ResearchCyclePolicy(
        selection=SelectionPolicy(population_size=2, preserve_diversity=False),
        crossover_rate=0.5,
        mutation_rate=0.5,
        elite_count=1,
    )


def test_research_runner_is_bounded_and_reproducible():
    population = make_parent_population()
    first = run_research(
        population,
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=3),
        seed=17,
    )
    second = run_research(
        population,
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=3),
        seed=17,
    )

    assert len(first.generations) == 3
    assert len(first.history.records) == 3
    assert [c.strategy_id for c in first.final_population] == [c.strategy_id for c in second.final_population]
    assert first.history == second.history
    assert not first.stopped_on_stagnation


def test_research_runner_can_stop_on_stagnation():
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=10, stop_on_stagnation=2),
        seed=3,
    )

    assert len(result.generations) == 3
    assert result.stopped_on_stagnation
    assert result.history.generations_without_improvement == 2


def test_research_run_policy_validates_bounds():
    with pytest.raises(ValueError, match="max_generations"):
        ResearchRunPolicy(max_generations=0)
    with pytest.raises(ValueError, match="epsilon"):
        ResearchRunPolicy(improvement_epsilon=-1)
    with pytest.raises(ValueError, match="stagnation"):
        ResearchRunPolicy(stop_on_stagnation=0)
