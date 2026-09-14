from pathlib import Path

import pytest

from atsf.registry import ExperimentRegistry
from atsf.sandbox_execution import ExecutionState
from atsf.sandbox_execution_journal import SandboxExecutionJournal


def test_event_journal_is_append_only_and_recoverable(tmp_path: Path):
    db = tmp_path / "research.db"
    registry = ExperimentRegistry(db)
    journal = SandboxExecutionJournal(registry)

    journal.append("intent-1", ExecutionState.CREATED, timestamp=100.0)
    journal.append("intent-1", ExecutionState.VALIDATED, timestamp=101.0)
    journal.append("intent-1", ExecutionState.ADMITTED, timestamp=102.0)
    journal.append("intent-1", ExecutionState.CONSUMED, timestamp=103.0)
    journal.append("intent-1", ExecutionState.FILLED, timestamp=104.0, detail="price=100.25")

    assert journal.latest("intent-1").state is ExecutionState.FILLED
    assert journal.verify("intent-1")
    registry.close()

    restarted = ExperimentRegistry(db)
    recovered = SandboxExecutionJournal(restarted)
    events = recovered.events("intent-1")
    assert [event.state for event in events] == [
        ExecutionState.CREATED,
        ExecutionState.VALIDATED,
        ExecutionState.ADMITTED,
        ExecutionState.CONSUMED,
        ExecutionState.FILLED,
    ]
    assert recovered.verify("intent-1")
    restarted.close()


def test_event_fingerprints_are_deterministic():
    registry = ExperimentRegistry(":memory:")
    journal = SandboxExecutionJournal(registry)
    first = journal.append("intent-1", ExecutionState.CREATED, timestamp=100.0)
    assert journal.verify("intent-1")
    assert first.event_fingerprint
    registry.close()


def test_invalid_event_timestamp_is_rejected():
    registry = ExperimentRegistry(":memory:")
    journal = SandboxExecutionJournal(registry)
    with pytest.raises(ValueError, match="timestamp must be finite"):
        journal.append("intent-1", ExecutionState.CREATED, timestamp=float("nan"))
    registry.close()


def test_empty_intent_id_is_rejected():
    registry = ExperimentRegistry(":memory:")
    journal = SandboxExecutionJournal(registry)
    with pytest.raises(ValueError, match="intent_id is required"):
        journal.append("", ExecutionState.CREATED, timestamp=100.0)
    registry.close()
