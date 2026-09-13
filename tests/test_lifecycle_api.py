from fastapi.testclient import TestClient

from atsf.api import create_app
from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_store import LifecycleStore
from atsf.registry import ExperimentRegistry


def test_lifecycle_api_exposes_state_and_integrity_checked_history():
    registry = ExperimentRegistry()
    strategy_id = registry.save_strategy({"name": "api-lifecycle"}) if False else "strategy-api-lifecycle"
    LifecycleStore(registry._connection).save(strategy_id, StrategyLifecycleStage.RESEARCH, reason="research admitted")
    LifecycleStore(registry._connection).transition(strategy_id, StrategyLifecycleStage.RESEARCH, StrategyLifecycleStage.VALIDATED, reason="validation passed")
    client = TestClient(create_app(registry))

    current = client.get(f"/strategies/{strategy_id}/lifecycle")
    assert current.status_code == 200
    assert current.json() == {
        "strategy_id": strategy_id,
        "stage": "validated",
        "reason": "validation passed",
    }

    history = client.get(f"/strategies/{strategy_id}/lifecycle/history")
    assert history.status_code == 200
    payload = history.json()
    assert payload["event_count"] == 2
    assert [event["target_stage"] for event in payload["events"]] == ["research", "validated"]
    assert payload["events"][0]["source_stage"] is None
    assert all(event["integrity_hash"] for event in payload["events"])
    registry.close()


def test_lifecycle_api_fails_closed_on_tampered_state():
    registry = ExperimentRegistry()
    strategy_id = "strategy-api-tamper"
    LifecycleStore(registry._connection).save(strategy_id, StrategyLifecycleStage.RESEARCH, reason="research admitted")
    registry._connection.execute("UPDATE strategy_lifecycle SET reason = ? WHERE strategy_id = ?", ("tampered", strategy_id))
    registry._connection.commit()
    client = TestClient(create_app(registry))

    response = client.get(f"/strategies/{strategy_id}/lifecycle")
    assert response.status_code == 409
    assert "integrity" in str(response.json()["detail"]).lower()
    registry.close()
