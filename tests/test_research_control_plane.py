import sqlite3

import pytest

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
