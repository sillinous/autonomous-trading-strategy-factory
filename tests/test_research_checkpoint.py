from types import SimpleNamespace

import pandas as pd
import pytest

from atsf.research_checkpoint import ResearchCheckpoint
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


def test_checkpoint_round_trip_preserves_restart_state():
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=3),
        seed=17,
    )
    checkpoint = result.checkpoints[-1]
    restored = ResearchCheckpoint.from_json(checkpoint.to_json())

    assert restored == checkpoint
    assert restored.state_digest == checkpoint.state_digest
    assert restored.compute_state_digest() == restored.state_digest
    assert [c.strategy_id for c in restored.population] == [c.strategy_id for c in result.final_population]
    assert restored.next_generation == 3
    assert restored.next_seed == 20
    assert restored.history.known_strategy_ids == checkpoint.history.known_strategy_ids
    assert restored.to_dict()["known_strategy_ids"] == sorted(checkpoint.history.known_strategy_ids)


def test_checkpoint_detects_tampering():
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=4,
    )
    payload = result.checkpoints[-1].to_dict()
    payload["next_seed"] += 1
    with pytest.raises(ValueError, match="digest"):
        ResearchCheckpoint.from_dict(payload)


def test_checkpoint_rejects_strategy_id_mismatch():
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=4,
    )
    payload = result.checkpoints[-1].to_dict()
    payload.pop("state_digest")
    payload["population"][0]["strategy_id"] = "tampered"
    with pytest.raises(ValueError, match="strategy ID"):
        ResearchCheckpoint.from_dict(payload)


def test_checkpoint_rejects_previous_schema_version():
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=4,
    )
    payload = result.checkpoints[-1].to_dict()
    payload["schema_version"] = 1
    payload.pop("state_digest")
    with pytest.raises(ValueError, match="unsupported checkpoint schema version"):
        ResearchCheckpoint.from_dict(payload)
