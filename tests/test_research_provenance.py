from types import SimpleNamespace

import pandas as pd

from atsf.research_cycle import ResearchCyclePolicy, run_research_cycle
from atsf.research_provenance import build_generation_provenance
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


def test_generation_provenance_is_deterministic():
    population = make_parent_population()
    policy = ResearchCyclePolicy(
        selection=SelectionPolicy(population_size=2, preserve_diversity=False),
        elite_count=1,
    )
    result = run_research_cycle(
        population,
        fake_evaluation,
        generation=4,
        seed=91,
        policy=policy,
    )

    first = build_generation_provenance(result, population, seed=91)
    second = build_generation_provenance(result, population, seed=91)

    assert first == second
    assert first.generation == 4
    assert len(first.candidate_records) == len(population)
    assert first.metrics_digest
    assert all(record.genome_digest for record in first.candidate_records)
    assert all(record.evaluation_digest for record in first.candidate_records)
