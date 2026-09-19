import sqlite3

import pytest

from atsf.research_checkpoint import ResearchCheckpoint
from atsf.research_checkpoint_store import ResearchCheckpointStore
from atsf.research_control_plane import ResearchControlPlane
from atsf.research_cycle_registry import ResearchCycleRegistry
from atsf.research_runner import ResearchRunPolicy, run_research
from tests.test_population import make_parent_population
from tests.test_research_runner import cycle_policy, fake_evaluation


def test_control_plane_persists_and_cross_verifies_generation():
    connection = sqlite3.connect(":memory:")
    control = ResearchControlPlane(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=17,
    )
    checkpoint = result.final_checkpoint
    assert checkpoint is not None

    control.persist_generation(result.generations[0], seed=17, checkpoint=checkpoint)

    assert control.latest_checkpoint() == checkpoint
    control.verify()


def test_control_plane_detects_cross_store_checkpoint_tampering():
    connection = sqlite3.connect(":memory:")
    registry = ResearchCycleRegistry(connection)
    checkpoints = ResearchCheckpointStore(connection)
    control = ResearchControlPlane(
        connection,
        cycle_registry=registry,
        checkpoint_store=checkpoints,
    )
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=17,
    )
    checkpoint = result.final_checkpoint
    assert checkpoint is not None
    control.persist_generation(result.generations[0], seed=17, checkpoint=checkpoint)

    connection.execute(
        "UPDATE research_cycles SET plan_json = ? WHERE cycle_id = ?",
        ('{"checkpoint_digest":"tampered"}', "generation-0"),
    )
    connection.commit()

    with pytest.raises(ValueError, match="does not match research-cycle evidence"):
        control.verify()


def test_control_plane_requires_shared_connection():
    connection_a = sqlite3.connect(":memory:")
    connection_b = sqlite3.connect(":memory:")

    with pytest.raises(ValueError, match="share a connection"):
        ResearchControlPlane(
            connection_a,
            cycle_registry=ResearchCycleRegistry(connection_b),
        )


def test_control_plane_rollback_is_atomic():
    connection = sqlite3.connect(":memory:")
    control = ResearchControlPlane(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=17,
    )
    checkpoint = result.final_checkpoint
    assert checkpoint is not None

    original = control.checkpoint_store.save_in_transaction

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("forced rollback")

    control.checkpoint_store.save_in_transaction = fail

    with pytest.raises(RuntimeError, match="forced rollback"):
        control.persist_generation(result.generations[0], seed=17, checkpoint=checkpoint)

    assert control.cycle_registry.list_cycles() == []
    assert control.checkpoint_store.latest() is None


def test_control_plane_verifies_generation_pair():
    connection = sqlite3.connect(":memory:")
    control = ResearchControlPlane(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=17,
    )
    checkpoint = result.final_checkpoint
    assert checkpoint is not None
    control.persist_generation(result.generations[0], seed=17, checkpoint=checkpoint)

    pair = control.verify_generation(0)
    assert pair.cycle.generation == 0
    assert pair.checkpoint == checkpoint


def test_control_plane_rejects_generation_without_checkpoint():
    connection = sqlite3.connect(":memory:")
    control = ResearchControlPlane(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=17,
    )
    checkpoint = result.final_checkpoint
    assert checkpoint is not None
    control.persist_generation(result.generations[0], seed=17, checkpoint=checkpoint)

    with pytest.raises(ValueError, match="research cycle not found"):
        control.verify_generation(1)


def test_control_plane_rejects_checkpoint_seed_mismatch_before_persisting():
    connection = sqlite3.connect(":memory:")
    control = ResearchControlPlane(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=17,
    )
    checkpoint = result.final_checkpoint
    assert checkpoint is not None
    tampered = checkpoint.to_dict()
    tampered["next_seed"] = checkpoint.next_seed + 10
    tampered["state_digest"] = ""
    invalid = ResearchCheckpoint.from_dict(tampered)

    with pytest.raises(ValueError, match="next_seed does not follow"):
        control.persist_generation(result.generations[0], seed=17, checkpoint=invalid)

    assert control.cycle_registry.list_cycles() == []
    assert control.checkpoint_store.latest() is None


def test_control_plane_fails_closed_on_malformed_cycle_plan():
    connection = sqlite3.connect(":memory:")
    control = ResearchControlPlane(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=17,
    )
    checkpoint = result.final_checkpoint
    assert checkpoint is not None
    control.persist_generation(result.generations[0], seed=17, checkpoint=checkpoint)
    connection.execute(
        "UPDATE research_cycles SET plan_json = ? WHERE cycle_id = ?",
        ("{not-json", "generation-0"),
    )
    connection.commit()

    with pytest.raises(ValueError, match="plan is invalid"):
        control.verify_generation(0)

def test_control_plane_rejects_discontinuous_seed_before_second_generation():
    connection = sqlite3.connect(":memory:")
    control = ResearchControlPlane(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=17,
    )
    checkpoint = result.final_checkpoint
    assert checkpoint is not None
    control.persist_generation(result.generations[0], seed=17, checkpoint=checkpoint)

    next_result = run_research(
        list(checkpoint.population),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=2),
        seed=99,
    )
    second_checkpoint = next_result.final_checkpoint
    assert second_checkpoint is not None
    second = next_result.generations[0]

    with pytest.raises(ValueError, match="seed does not follow durable checkpoint"):
        control.persist_generation(second, seed=99, checkpoint=second_checkpoint)

    assert len(control.cycle_registry.list_cycles()) == 1
    assert control.checkpoint_store.latest() == checkpoint



def test_control_plane_rejects_discontinuous_seed_during_durable_resume():
    connection = sqlite3.connect(":memory:")
    control = ResearchControlPlane(connection)
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=17,
    )
    checkpoint = result.final_checkpoint
    assert checkpoint is not None
    control.persist_generation(result.generations[0], seed=17, checkpoint=checkpoint)

    next_result = run_research(
        list(checkpoint.population),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=2),
        seed=18,
    )
    second = next_result.generations[0]
    second_checkpoint = next_result.final_checkpoint
    assert second_checkpoint is not None

    with pytest.raises(ValueError, match="seed does not follow durable checkpoint"):
        control.persist_generation(second, seed=19, checkpoint=second_checkpoint)

    assert len(control.cycle_registry.list_cycles()) == 1
    assert control.checkpoint_store.latest() == checkpoint
