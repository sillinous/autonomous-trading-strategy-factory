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
    try:
        journal.recover_state("intent-1")
    except ValueError as exc:
        assert "integrity verification failed" in str(exc)
    else:
        raise AssertionError("tampered journal must fail closed")
    registry.close()


def test_recover_state_returns_none_for_unknown_intent():
    registry = ExperimentRegistry(":memory:")
    journal = SandboxExecutionJournal(registry)
    assert journal.recover_state("unknown") is None
    registry.close()
