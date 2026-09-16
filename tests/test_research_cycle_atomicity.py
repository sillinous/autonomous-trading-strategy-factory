from __future__ import annotations

import sqlite3

import pytest

from atsf.research_cycle_registry import ResearchCycleRegistry


def test_cycle_and_audit_commit_together() -> None:
    connection = sqlite3.connect(":memory:")
    registry = ResearchCycleRegistry(connection)

    with connection:
        registry.save_cycle_in_transaction(
            "cycle:1",
            1,
            plan={"generation": 1},
            feedback={},
            admissions={},
        )

    assert registry.get_cycle("cycle:1") is not None
    assert connection.execute("SELECT COUNT(*) FROM research_cycle_audit").fetchone()[0] == 1
    registry.verify()


def test_cycle_and_audit_rollback_together() -> None:
    connection = sqlite3.connect(":memory:")
    registry = ResearchCycleRegistry(connection)

    with pytest.raises(RuntimeError, match="abort"):
        with connection:
            registry.save_cycle_in_transaction(
                "cycle:1",
                1,
                plan={"generation": 1},
                feedback={},
                admissions={},
            )
            raise RuntimeError("abort")

    assert registry.get_cycle("cycle:1") is None
    assert connection.execute("SELECT COUNT(*) FROM research_cycle_audit").fetchone()[0] == 0
    registry.verify()


def test_failed_successor_does_not_leave_partial_cycle_or_audit() -> None:
    connection = sqlite3.connect(":memory:")
    registry = ResearchCycleRegistry(connection)
    registry.save_cycle("cycle:1", 1, plan={}, feedback={}, admissions={})

    with pytest.raises(RuntimeError, match="abort"):
        with connection:
            registry.save_cycle_in_transaction(
                "cycle:2",
                2,
                plan={"generation": 2},
                feedback={},
                admissions={},
            )
            raise RuntimeError("abort")

    assert registry.get_cycle("cycle:2") is None
    assert [item.cycle_id for item in registry.list_cycles()] == ["cycle:1"]
    assert connection.execute("SELECT COUNT(*) FROM research_cycle_audit").fetchone()[0] == 1
    registry.verify()


def test_tampered_history_blocks_transactional_extension() -> None:
    connection = sqlite3.connect(":memory:")
    registry = ResearchCycleRegistry(connection)
    registry.save_cycle("cycle:1", 1, plan={}, feedback={}, admissions={})
    connection.execute(
        "UPDATE research_cycle_audit SET payload_digest = 'tampered' WHERE cycle_id = 'cycle:1'"
    )
    connection.commit()

    with pytest.raises(ValueError, match="integrity verification failed"):
        with connection:
            registry.save_cycle_in_transaction(
                "cycle:2",
                2,
                plan={},
                feedback={},
                admissions={},
            )

    assert registry.get_cycle("cycle:2") is None
    assert connection.execute("SELECT COUNT(*) FROM research_cycles").fetchone()[0] == 1
