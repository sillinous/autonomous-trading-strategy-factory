from types import SimpleNamespace

import pandas as pd
import pytest

from atsf.research_cycle import ResearchCyclePolicy, run_research_cycle
from atsf.research_provenance import build_generation_provenance
from atsf.selection import SelectionPolicy
from tests.test_population import make_parent_population


def fake_evaluation(candidate):
    baseline = pd.Series([1.0, 100.0])
    return SimpleNamespace(
        candidate_id=candidate.strategy_id,
        experiment=SimpleNamespace(dataset_id="dataset", dataset_version="1"),
        fitness=SimpleNamespace(score=1.0),
        backtest=SimpleNamespace(max_drawdown=0.10),
        walk_forward=SimpleNamespace(oos_sharpe=1.2),
        monte_carlo=SimpleNamespace(pass_rate=0.90),
        perturbation=SimpleNamespace(stable=True),
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
        elite_count=1,
    )


def test_generation_provenance_is_deterministic():
    population = make_parent_population()
    result = run_research_cycle(
        population,
        fake_evaluation,
        generation=4,
        seed=91,
        policy=cycle_policy(),
    )

    first = build_generation_provenance(result, population, seed=91)
    second = build_generation_provenance(result, population, seed=91)

    assert first == second
    assert first.generation == 4
    assert len(first.candidate_records) == len(population)
    assert first.metrics_digest
    assert all(record.genome_digest for record in first.candidate_records)
    assert all(record.evaluation_digest for record in first.candidate_records)


def test_provenance_digest_covers_evaluation_evidence():
    population = make_parent_population()
    result = run_research_cycle(
        population,
        fake_evaluation,
        generation=4,
        seed=91,
        policy=cycle_policy(),
    )
    original = build_generation_provenance(result, population, seed=91)

    changed_evaluation = fake_evaluation(population[0])
    changed_evaluation.backtest.max_drawdown = 0.20
    changed = list(result.evaluations)
    changed[0] = changed_evaluation
    altered_result = SimpleNamespace(
        generation=result.generation,
        evaluations=tuple(changed),
        metrics=result.metrics,
        selected_parents=result.selected_parents,
        next_population=result.next_population,
    )

    altered = build_generation_provenance(altered_result, population, seed=91)
    assert altered.candidate_records[0].evaluation_digest != original.candidate_records[0].evaluation_digest


def test_provenance_rejects_evaluation_candidate_mismatch():
    population = make_parent_population()
    result = run_research_cycle(
        population,
        fake_evaluation,
        generation=4,
        seed=91,
        policy=cycle_policy(),
    )
    mismatched = list(result.evaluations)
    mismatched[0] = SimpleNamespace(
        **{**vars(mismatched[0]), "candidate_id": "not-the-candidate"}
    )
    altered_result = SimpleNamespace(
        generation=result.generation,
        evaluations=tuple(mismatched),
        metrics=result.metrics,
        selected_parents=result.selected_parents,
        next_population=result.next_population,
    )

    with pytest.raises(ValueError, match="candidate_id"):
        build_generation_provenance(altered_result, population, seed=91)
