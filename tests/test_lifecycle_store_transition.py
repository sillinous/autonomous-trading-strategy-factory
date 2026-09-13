import sqlite3

import pytest

from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_store import LifecycleStore


def test_promoted_to_paper_transition_survives_restart(tmp_path):
    path = tmp_path / "lifecycle.sqlite3"
    first = sqlite3.connect(path)
    first.row_factory = sqlite3.Row
    store = LifecycleStore(first)
    store.save("strategy-1", StrategyLifecycleStage.PROMOTED, reason="promotion gate")
    store.transition("strategy-1", StrategyLifecycleStage.PROMOTED, StrategyLifecycleStage.PAPER, reason="verified admission")
    first.close()

    reopened = sqlite3.connect(path)
    reopened.row_factory = sqlite3.Row
    recovered = LifecycleStore(reopened).get("strategy-1")
    assert recovered is not None
    assert recovered.stage is StrategyLifecycleStage.PAPER
    assert recovered.reason == "verified admission"
    reopened.close()


def test_transition_requires_existing_matching_source():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)
    with pytest.raises(ValueError, match="missing"):
        store.transition("strategy-1", StrategyLifecycleStage.PROMOTED, StrategyLifecycleStage.PAPER, reason="verified")
    store.save("strategy-1", StrategyLifecycleStage.VALIDATED, reason="validated")
    with pytest.raises(ValueError, match="source stage"):
        store.transition("strategy-1", StrategyLifecycleStage.PROMOTED, StrategyLifecycleStage.PAPER, reason="verified")


def test_transition_rejects_invalid_jump():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)
    store.save("strategy-1", StrategyLifecycleStage.RESEARCH, reason="research")
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        store.transition("strategy-1", StrategyLifecycleStage.RESEARCH, StrategyLifecycleStage.PAPER, reason="invalid")


def test_valid_stage_tampering_is_detected():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)
    store.save("strategy-1", StrategyLifecycleStage.PROMOTED, reason="promotion gate")
    connection.execute("UPDATE strategy_lifecycle SET stage = 'paper' WHERE strategy_id = 'strategy-1'")
    connection.commit()
    with pytest.raises(ValueError, match="integrity"):
        store.get("strategy-1")
