import sqlite3

import pytest

from atsf.research_checkpoint import ResearchCheckpoint
from atsf.research_checkpoint_store import ResearchCheckpointStore
from atsf.research_runner import ResearchRunPolicy, run_research
from tests.test_population import make_parent_population
from tests.test_research_runner import cycle_policy, fake_evaluation


def _checkpoint():
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=17,
    )
    return result.final_checkpoint


def test_checkpoint_store_round_trips_and_survives_restart():
    checkpoint = _checkpoint()
    assert checkpoint is not None
    connection = sqlite3.connect(":memory:")
    store = ResearchCheckpointStore(connection)
    store.save(checkpoint, cycle_id="generation-0")
    assert store.load(1) == checkpoint
    store.verify()
    reopened = ResearchCheckpointStore(connection)
    assert reopened.latest() == checkpoint


def test_checkpoint_store_is_immutable():
    checkpoint = _checkpoint()
    assert checkpoint is not None
    connection = sqlite3.connect(":memory:")
    store = ResearchCheckpointStore(connection)
    store.save(checkpoint, cycle_id="generation-0")
    payload = checkpoint.to_dict()
    payload["next_seed"] += 1
    tampered = ResearchCheckpoint.from_dict({**payload, "state_digest": ""})
    with pytest.raises(ValueError, match="immutable"):
        store.save(tampered, cycle_id="generation-0")


def test_checkpoint_store_rolls_back_with_outer_transaction():
    checkpoint = _checkpoint()
    assert checkpoint is not None
    connection = sqlite3.connect(":memory:")
    store = ResearchCheckpointStore(connection)
    with pytest.raises(RuntimeError):
        with connection:
            store.save_in_transaction(checkpoint, cycle_id="generation-0")
            raise RuntimeError("rollback")
    assert store.get(1) is None


def test_checkpoint_store_detects_tampering():
    checkpoint = _checkpoint()
    assert checkpoint is not None
    connection = sqlite3.connect(":memory:")
    store = ResearchCheckpointStore(connection)
    store.save(checkpoint, cycle_id="generation-0")
    connection.execute(
        "UPDATE research_checkpoints SET payload_json = ? WHERE next_generation = 1",
        (checkpoint.to_json().replace('"stopped":false', '"stopped":true'),),
    )
    connection.commit()
    with pytest.raises(ValueError, match="integrity"):
        store.verify()


def test_checkpoint_store_rejects_invalid_cycle_id_and_generation():
    checkpoint = _checkpoint()
    assert checkpoint is not None
    connection = sqlite3.connect(":memory:")
    store = ResearchCheckpointStore(connection)
    with pytest.raises(ValueError, match="cycle_id"):
        store.save(checkpoint, cycle_id="")
    with pytest.raises(ValueError, match="durable checkpoint"):
        ResearchCheckpointStore(connection).save(
            ResearchCheckpoint(
                next_generation=0,
                next_seed=1,
                population=checkpoint.population,
                history=checkpoint.history,
                provenance=checkpoint.provenance,
            ),
            cycle_id="generation-0",
        )
