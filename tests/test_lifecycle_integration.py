from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_integration import synchronize_candidate_lifecycle
from atsf.lifecycle_store import LifecycleStore


def evaluation(*, validation_passed: bool, promoted: bool) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_id="strategy-1",
        validation_passed=validation_passed,
        promotion=SimpleNamespace(eligible=promoted),
    )


def test_candidate_lifecycle_promotes_only_after_validation_and_promotion() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)

    result = synchronize_candidate_lifecycle(store, evaluation(validation_passed=True, promoted=True))

    assert result.stage is StrategyLifecycleStage.PROMOTED
    assert [event.target_stage for event in store.history("strategy-1")] == [
        StrategyLifecycleStage.RESEARCH,
        StrategyLifecycleStage.VALIDATED,
        StrategyLifecycleStage.PROMOTED,
    ]


def test_candidate_lifecycle_degrades_failed_validation() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)

    result = synchronize_candidate_lifecycle(store, evaluation(validation_passed=False, promoted=False))

    assert result.stage is StrategyLifecycleStage.DEGRADED
    assert store.history("strategy-1")[-1].target_stage is StrategyLifecycleStage.DEGRADED


def test_candidate_lifecycle_keeps_validated_when_promotion_fails() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)

    result = synchronize_candidate_lifecycle(store, evaluation(validation_passed=True, promoted=False))

    assert result.stage is StrategyLifecycleStage.VALIDATED
    assert [event.target_stage for event in store.history("strategy-1")] == [
        StrategyLifecycleStage.RESEARCH,
        StrategyLifecycleStage.VALIDATED,
    ]


def test_candidate_lifecycle_is_idempotent_after_promotion() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = LifecycleStore(connection)
    candidate = evaluation(validation_passed=True, promoted=True)

    first = synchronize_candidate_lifecycle(store, candidate)
    before = store.history("strategy-1")
    second = synchronize_candidate_lifecycle(store, candidate)

    assert second == first
    assert store.history("strategy-1") == before
