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
    with pytest.raises(ValueError, match="invalid"):
        store.get("strategy-1")
