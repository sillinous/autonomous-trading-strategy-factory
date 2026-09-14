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


def test_first_event_must_create_execution_state():
    registry = ExperimentRegistry(":memory:")
    journal = SandboxExecutionJournal(registry)
    with pytest.raises(ValueError, match="first execution event must be CREATED"):
        journal.append("intent-1", ExecutionState.VALIDATED, timestamp=100.0)
    registry.close()


@pytest.mark.parametrize(
    ("current", "next_state"),
    [
        (ExecutionState.CREATED, ExecutionState.CONSUMED),
        (ExecutionState.VALIDATED, ExecutionState.CREATED),
        (ExecutionState.ADMITTED, ExecutionState.FILLED),
        (ExecutionState.CONSUMED, ExecutionState.CREATED),
        (ExecutionState.FILLED, ExecutionState.CREATED),
        (ExecutionState.REJECTED, ExecutionState.FILLED),
        (ExecutionState.CANCELLED, ExecutionState.VALIDATED),
        (ExecutionState.EXPIRED, ExecutionState.ADMITTED),
    ],
)
def test_impossible_execution_transition_is_rejected(current, next_state):
    registry = ExperimentRegistry(":memory:")
    journal = SandboxExecutionJournal(registry)
    journal.append("intent-1", ExecutionState.CREATED, timestamp=100.0)
    if current is not ExecutionState.CREATED:
        journal.append("intent-1", current, timestamp=101.0)
    with pytest.raises(ValueError, match="invalid execution transition"):
        journal.append("intent-1", next_state, timestamp=102.0)
    registry.close()


def test_terminal_states_cannot_advance():
    registry = ExperimentRegistry(":memory:")
    journal = SandboxExecutionJournal(registry)
    for terminal in (ExecutionState.FILLED, ExecutionState.REJECTED, ExecutionState.CANCELLED, ExecutionState.EXPIRED):
        intent_id = f"{terminal.value.lower()}-intent"
        journal.append(intent_id, ExecutionState.CREATED, timestamp=100.0)
        if terminal is ExecutionState.FILLED:
            for state in (ExecutionState.VALIDATED, ExecutionState.ADMITTED, ExecutionState.CONSUMED):
                journal.append(intent_id, state, timestamp=101.0)
        journal.append(intent_id, terminal, timestamp=102.0)
        with pytest.raises(ValueError, match="invalid execution transition"):
            journal.append(intent_id, ExecutionState.CREATED, timestamp=103.0)
    registry.close()
