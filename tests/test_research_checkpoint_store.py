import sqlite3

import pytest

from atsf.research_checkpoint import ResearchCheckpoint
from atsf.research_checkpoint_store import ResearchCheckpointStore
from atsf.research_runner import ResearchRunPolicy, run_research
from tests.test_population import make_parent_population
from tests.test_research_runner import cycle_policy, fake_evaluation


def _checkpoint(seed=17):
    result = run_research(
        make_parent_population(),
        fake_evaluation,
        cycle_policy=cycle_policy(),
        run_policy=ResearchRunPolicy(max_generations=1),
        seed=seed,
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


def test_checkpoint_store_rejects_unsupported_schema():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE research_checkpoints ("
        "checkpoint_id TEXT PRIMARY KEY, next_generation INTEGER NOT NULL UNIQUE, "
        "cycle_id TEXT NOT NULL, state_digest TEXT NOT NULL, payload_json TEXT NOT NULL, "
        "schema_version INTEGER NOT NULL)"
    )
    connection.execute(
        "INSERT INTO research_checkpoints VALUES ('bad', 1, 'generation-0', 'x', '{}', 99)"
    )
    connection.commit()

    with pytest.raises(ValueError, match="schema version"):
        ResearchCheckpointStore(connection)


def test_checkpoint_store_binds_checkpoint_to_completed_cycle():
    checkpoint = _checkpoint()
    assert checkpoint is not None
    connection = sqlite3.connect(":memory:")
    store = ResearchCheckpointStore(connection)

    with pytest.raises(ValueError, match="does not match completed generation"):
        store.save(checkpoint, cycle_id="generation-99")

    assert store.latest() is None


def test_checkpoint_store_rejects_generation_gaps():
    connection = sqlite3.connect(":memory:")
    store = ResearchCheckpointStore(connection)
    checkpoint = _checkpoint()
    store.save(checkpoint, cycle_id="generation-0")

    payload = checkpoint.to_dict()
    payload["next_generation"] = 3
    payload["next_seed"] = 3
    gapped = ResearchCheckpoint.from_dict({**payload, "state_digest": ""})

    with pytest.raises(ValueError, match="generation sequence"):
        store.save(gapped, cycle_id="generation-2")


def test_checkpoint_store_rejects_generation_gap_before_insert():
    connection = sqlite3.connect(":memory:")
    store = ResearchCheckpointStore(connection)
    first = _checkpoint(1)
    store.save(first, cycle_id="generation-0")

    second_payload = first.to_dict()
    second_payload["next_generation"] = 3
    second_payload["next_seed"] = 19
    second_payload["state_digest"] = ""
    second = ResearchCheckpoint.from_dict(second_payload)

    with pytest.raises(ValueError, match="generation sequence"):
        store.save(second, cycle_id="generation-2")

    assert store.latest() == first
