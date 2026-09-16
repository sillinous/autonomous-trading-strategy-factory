from types import SimpleNamespace
import sqlite3

import pandas as pd
import pytest

from atsf.research_cycle import ResearchCyclePolicy
from atsf.research_cycle_registry import ResearchCycleRegistry
from atsf.research_runner import ResearchRunPolicy, resume_research, run_research
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


def test_research_runner_resume_matches_uninterrupted_final_state():
    population = make_parent_population()
    uninterrupted = run_research(
        population,
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=5),
        seed=17,
    )
    first_segment = run_research(
        population,
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=3),
        seed=17,
    )

    checkpoint = first_segment.final_checkpoint
    assert checkpoint is not None
    assert checkpoint.next_generation == 3
    assert checkpoint.next_seed == 20

    resumed = resume_research(
        checkpoint,
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=5),
    )

    assert [c.strategy_id for c in resumed.final_population] == [
        c.strategy_id for c in uninterrupted.final_population
    ]
    assert resumed.history == uninterrupted.history
    assert resumed.provenance == uninterrupted.provenance
    assert resumed.final_checkpoint == uninterrupted.final_checkpoint
    assert len(resumed.generations) == 2
    assert resumed.generations[0].generation == 3
    assert resumed.generations[1].generation == 4


def test_research_runner_resume_rejects_stopped_checkpoint():
    stopped = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=10, stop_on_stagnation=2),
        seed=3,
    )
    checkpoint = stopped.final_checkpoint
    assert checkpoint is not None and checkpoint.stopped

    with pytest.raises(ValueError, match="stopped checkpoint"):
        resume_research(
            checkpoint,
            fake_evaluation,
            cycle_policy=cycle_policy(),
            run_policy=ResearchRunPolicy(max_generations=10),
        )


def test_research_runner_resume_validates_generation_budget():
    first = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=3),
        seed=17,
    )
    checkpoint = first.final_checkpoint
    assert checkpoint is not None

    with pytest.raises(ValueError, match="exceeds max_generations"):
        resume_research(
            checkpoint,
            fake_evaluation,
            cycle_policy=cycle_policy(),
            run_policy=ResearchRunPolicy(max_generations=2),
        )


def test_research_run_policy_validates_bounds():
    with pytest.raises(ValueError, match="max_generations"):
        ResearchRunPolicy(max_generations=0)
    with pytest.raises(ValueError, match="epsilon"):
        ResearchRunPolicy(improvement_epsilon=-1)
    with pytest.raises(ValueError, match="stagnation"):
        ResearchRunPolicy(stop_on_stagnation=0)


def test_research_runner_persists_and_verifies_cycles_before_advancing():
    connection = sqlite3.connect(":memory:")
    registry = ResearchCycleRegistry(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=2),
        seed=17,
        cycle_registry=registry,
    )

    assert [record.generation for record in registry.list_cycles()] == [0, 1]
    registry.verify()
    resumed = resume_research(
        result.final_checkpoint,
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=3),
        cycle_registry=registry,
    )
    assert resumed.generations[0].generation == 2
    assert [record.generation for record in registry.list_cycles()] == [0, 1, 2]
    registry.verify()


def test_research_runner_blocks_advancement_after_cycle_tampering():
    connection = sqlite3.connect(":memory:")
    registry = ResearchCycleRegistry(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=2),
        seed=17,
        cycle_registry=registry,
    )
    connection.execute("UPDATE research_cycles SET feedback_json = '{}' WHERE cycle_id = 'generation-0'")
    connection.commit()

    with pytest.raises(ValueError, match="integrity"):
        resume_research(
            result.final_checkpoint,
            fake_evaluation,
            cycle_policy=cycle_policy(),
            run_policy=ResearchRunPolicy(max_generations=3),
            cycle_registry=registry,
        )

    assert [record.generation for record in registry.list_cycles()] == [0, 1]


def test_research_runner_blocks_advancement_after_audit_tampering():
    connection = sqlite3.connect(":memory:")
    registry = ResearchCycleRegistry(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=2),
        seed=17,
        cycle_registry=registry,
    )
    connection.execute("UPDATE research_cycle_audit SET previous_digest = 'tampered' WHERE sequence = 2")
    connection.commit()

    with pytest.raises(ValueError, match="integrity"):
        resume_research(
            result.final_checkpoint,
            fake_evaluation,
            cycle_policy=cycle_policy(),
            run_policy=ResearchRunPolicy(max_generations=3),
            cycle_registry=registry,
        )

    assert [record.generation for record in registry.list_cycles()] == [0, 1]
