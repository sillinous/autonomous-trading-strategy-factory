from __future__ import annotations

import sqlite3

import pytest

from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_store import LifecycleStore


def test_lifecycle_store_survives_restart() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)
    record = store.save("strategy-1", StrategyLifecycleStage.PAPER, reason="verified admission")
    assert store.get("strategy-1") == record
    assert store.restore_stage("strategy-1", StrategyLifecycleStage.RESEARCH) == StrategyLifecycleStage.PAPER


def test_lifecycle_store_is_immutable() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)
    store.save("strategy-1", StrategyLifecycleStage.PROMOTED, reason="promotion gate")
    with pytest.raises(ValueError, match="immutable"):
        store.save("strategy-1", StrategyLifecycleStage.PAPER, reason="different state")


def test_lifecycle_store_fails_closed_on_tampered_stage() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)
    store.save("strategy-1", StrategyLifecycleStage.PAPER, reason="verified admission")
    connection.execute("UPDATE strategy_lifecycle SET stage = 'not-a-stage' WHERE strategy_id = 'strategy-1'")
    connection.commit()
    with pytest.raises(ValueError, match="integrity"):
        store.get("strategy-1")


def test_lifecycle_store_records_append_only_transition_history() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)
    store.save("strategy-1", StrategyLifecycleStage.RESEARCH, reason="seed")
    store.transition("strategy-1", StrategyLifecycleStage.RESEARCH, StrategyLifecycleStage.VALIDATED, reason="validation passed")
    store.transition("strategy-1", StrategyLifecycleStage.VALIDATED, StrategyLifecycleStage.PROMOTED, reason="promotion gate passed")

    history = store.history("strategy-1")
    assert [event.target_stage for event in history] == [
        StrategyLifecycleStage.RESEARCH,
        StrategyLifecycleStage.VALIDATED,
        StrategyLifecycleStage.PROMOTED,
    ]
    assert history[0].source_stage is None
    assert history[1].source_stage is StrategyLifecycleStage.RESEARCH
    assert history[2].source_stage is StrategyLifecycleStage.VALIDATED
    assert [event.sequence for event in history] == sorted(event.sequence for event in history)


def test_lifecycle_store_fails_closed_on_tampered_history_event() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)
    store.save("strategy-1", StrategyLifecycleStage.RESEARCH, reason="seed")
    connection.execute("UPDATE strategy_lifecycle_events SET reason = 'tampered' WHERE strategy_id = 'strategy-1'")
    connection.commit()
    with pytest.raises(ValueError, match="event integrity"):
        store.history("strategy-1")


def test_lifecycle_store_invalid_transition_does_not_append_history() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)
    store.save("strategy-1", StrategyLifecycleStage.RESEARCH, reason="seed")
    before = store.history("strategy-1")
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        store.transition("strategy-1", StrategyLifecycleStage.RESEARCH, StrategyLifecycleStage.PAPER, reason="invalid")
    assert store.history("strategy-1") == before
