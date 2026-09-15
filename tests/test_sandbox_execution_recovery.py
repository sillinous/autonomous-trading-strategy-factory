import pytest

from atsf.registry import ExperimentRegistry
from atsf.sandbox_execution import ExecutionState
from atsf.sandbox_execution_journal import SandboxExecutionJournal


def test_recover_state_requires_verified_chain():
    registry = ExperimentRegistry(":memory:")
    journal = SandboxExecutionJournal(registry)
    journal.append("intent-1", ExecutionState.CREATED, timestamp=100.0)
    journal.append("intent-1", ExecutionState.VALIDATED, timestamp=101.0)
    assert journal.recover_state("intent-1") is ExecutionState.VALIDATED
    registry.close()


def test_recover_state_fails_closed_on_tampering():
    registry = ExperimentRegistry(":memory:")
    journal = SandboxExecutionJournal(registry)
    journal.append("intent-1", ExecutionState.CREATED, timestamp=100.0)
    journal.append("intent-1", ExecutionState.VALIDATED, timestamp=101.0)
    registry._connection.execute(
        "UPDATE sandbox_execution_events SET detail = 'tampered' WHERE intent_id = ? AND sequence = 1",
        ("intent-1",),
    )
    registry._connection.commit()
    with pytest.raises(ValueError, match="integrity verification failed"):
        journal.recover_state("intent-1")
    registry.close()


def test_recover_state_returns_none_for_unknown_intent():
    registry = ExperimentRegistry(":memory:")
    journal = SandboxExecutionJournal(registry)
    assert journal.recover_state("unknown") is None
    registry.close()


def test_recover_state_survives_registry_restart(tmp_path):
    database = tmp_path / "sandbox.sqlite"
    first = ExperimentRegistry(database)
    first_journal = SandboxExecutionJournal(first)
    first_journal.append("intent-1", ExecutionState.CREATED, timestamp=100.0)
    first_journal.append("intent-1", ExecutionState.VALIDATED, timestamp=101.0)
    first.close()

    second = ExperimentRegistry(database)
    second_journal = SandboxExecutionJournal(second)
    assert second_journal.recover_state("intent-1") is ExecutionState.VALIDATED
    assert second_journal.verify("intent-1") is True
    second.close()


def test_legacy_journal_schema_fails_closed_without_explicit_migration():
    registry = ExperimentRegistry(":memory:")
    registry._connection.execute(
        """CREATE TABLE sandbox_execution_events (
            intent_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            state TEXT NOT NULL,
            timestamp REAL NOT NULL,
            event_fingerprint TEXT NOT NULL UNIQUE,
            detail TEXT NOT NULL DEFAULT '',
            previous_fingerprint TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (intent_id, sequence)
        )"""
    )
    registry._connection.commit()
    with pytest.raises(ValueError, match="explicit migration is required"):
        SandboxExecutionJournal(registry)
    registry.close()


def test_unsupported_schema_version_fails_closed():
    registry = ExperimentRegistry(":memory:")
    registry._connection.execute(
        """CREATE TABLE sandbox_execution_events (
            intent_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            state TEXT NOT NULL,
            timestamp REAL NOT NULL,
            event_fingerprint TEXT NOT NULL UNIQUE,
            detail TEXT NOT NULL DEFAULT '',
            previous_fingerprint TEXT NOT NULL DEFAULT '',
            schema_version INTEGER NOT NULL,
            PRIMARY KEY (intent_id, sequence)
        )"""
    )
    registry._connection.execute(
        "INSERT INTO sandbox_execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("intent-1", 1, ExecutionState.CREATED.value, 100.0, "x" * 64, "", "", 999),
    )
    registry._connection.commit()
    with pytest.raises(ValueError, match="unsupported sandbox execution journal schema version"):
        SandboxExecutionJournal(registry)
    registry.close()
