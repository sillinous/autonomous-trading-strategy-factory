import pandas as pd
from fastapi.testclient import TestClient

from atsf.api import create_app
from atsf.dataset_bundle import DatasetBundleIdentity, bundle_identity
from atsf.experiment import ExperimentResult, ExperimentSpec
from atsf.feedback_loop import process_strategy_health
from atsf.feedback_provenance import build_feedback_provenance
from atsf.feedback_registry import FeedbackEventStore
from atsf.lifecycle import StrategyLifecycle
from atsf.lineage import LineageRecord
from atsf.monitoring import DegradationReport
from atsf.registry import ExperimentRegistry
from atsf.research_queue import ResearchQueue
from atsf.strategy import Comparator, Condition, PositionSizing, RiskLimits, Signal, StrategySpec


def make_strategy() -> StrategySpec:
    return StrategySpec(
        name="api-test",
        version=1,
        universe=["TEST"],
        timeframe="1d",
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=2)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=1)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def seed(registry: ExperimentRegistry) -> str:
    strategy = make_strategy()
    strategy_id = registry.save_strategy(strategy)
    registry.save_lineage(LineageRecord(strategy_id=strategy_id, generation=0))
    registry.register_dataset(
        DatasetBundleIdentity("prices", "v1", ("TEST",), 1, "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        source="fixture",
    )
    spec = ExperimentSpec(strategy, "prices", "v1", seed=1)
    registry.save_experiment(spec, ExperimentResult(spec.experiment_id, "paper", score=1.0))
    registry.save_evaluation_evidence(
        spec.experiment_id,
        {"promotion": {"stage": "paper", "eligible": True, "reasons": []}},
    )
    market_data = pd.DataFrame([bar for bar in bars()]).set_index("timestamp")
    data_bundle_version = bundle_identity(
        {strategy_id: market_data},
        "prices",
        source="fixture",
        timeframe="1d",
    ).version
    registry.save_portfolio(
        "portfolio-1",
        {
            "dataset_id": "prices",
            "dataset_version": "v1",
            "data_bundle_version": data_bundle_version,
            "data_source": "fixture",
            "data_timeframe": "1d",
            "data_schema_version": "ohlcv.v1",
            "experiment_ids": {strategy_id: spec.experiment_id},
        },
        {strategy_id: 1.0},
    )
    return strategy_id


def bars(count: int = 4) -> list[dict]:
    frame = pd.DataFrame(
        {"close": [1.0 + (index % 3) for index in range(count)]},
        index=pd.date_range("2026-01-01", periods=count),
    )
    return [
        {
            "timestamp": timestamp.isoformat(),
            "open": value,
            "high": value,
            "low": value,
            "close": value,
            "volume": 1.0,
        }
        for timestamp, value in frame["close"].items()
    ]


def test_health_and_capabilities_are_paper_only(monkeypatch):
    monkeypatch.delenv("ATSF_API_KEY", raising=False)
    registry = ExperimentRegistry()
    client = TestClient(create_app(registry))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/capabilities").json() == {"live_execution_enabled": False}
    registry.close()


def test_configured_api_key_protects_operational_endpoints(monkeypatch):
    monkeypatch.setenv("ATSF_API_KEY", "test-secret")
    registry = ExperimentRegistry()
    client = TestClient(create_app(registry))
    assert client.get("/health").status_code == 200
    assert client.get("/capabilities").status_code == 401
    assert client.get("/capabilities", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/capabilities", headers={"X-API-Key": "test-secret"}).status_code == 200
    registry.close()


def test_research_run_endpoint_persists_research_artifacts(monkeypatch):
    monkeypatch.delenv("ATSF_API_KEY", raising=False)
    registry = ExperimentRegistry()
    client = TestClient(create_app(registry))
    response = client.post(
        "/research/runs",
        json={
            "dataset_id": "research-prices",
            "data": bars(30),
            "seeds": [make_strategy().model_dump(mode="json")],
            "generations": 1,
            "population_size": 1,
            "survivor_count": 1,
            "seed": 7,
        },
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["dataset_id"] == "research-prices"
    assert len(payload["dataset_version"]) == 16
    assert payload["generations"] == 1
    assert payload["final_population_size"] == 1
    registry.close()


def test_strategy_feedback_endpoint_exposes_persisted_history(monkeypatch):
    monkeypatch.delenv("ATSF_API_KEY", raising=False)
    registry = ExperimentRegistry()
    strategy_id = seed(registry)
    report = DegradationReport(
        degraded=True,
        observations=30,
        total_return=-0.12,
        max_drawdown=-0.22,
        volatility=0.08,
        reasons=("minimum return breached",),
    )
    action = process_strategy_health(strategy_id, StrategyLifecycle(), report, ResearchQueue())
    event = build_feedback_provenance(strategy_id, action, report, previous_state="active")
    FeedbackEventStore(registry).save(event)

    client = TestClient(create_app(registry))
    response = client.get(f"/strategies/{strategy_id}/feedback")
    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy_id"] == strategy_id
    assert payload["event_count"] == 1
    assert payload["events"][0]["event_id"] == event.event_id
    assert payload["events"][0]["research_request_id"] == event.research_request_id
    registry.close()


def test_paper_run_endpoint_executes_and_exposes_integrity_verification(monkeypatch):
    monkeypatch.delenv("ATSF_API_KEY", raising=False)
    registry = ExperimentRegistry()
    strategy_id = seed(registry)
    client = TestClient(create_app(registry))
    response = client.post(
        "/portfolios/portfolio-1/paper-runs",
        json={"dataset_version": "v1", "data": {strategy_id: bars()}},
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["portfolio_id"] == "portfolio-1"
    assert payload["final_equity"] > 0
    assert payload["execution_fingerprint"]
    assert payload["audit_event_count"] == payload["fill_count"]

    verification = client.get(f"/runs/{payload['run_id']}/verify")
    assert verification.status_code == 200
    assert verification.json()["valid"] is True
    assert verification.json()["event_count"] == payload["audit_event_count"]
    assert verification.json()["ledger_fingerprint"]

    replay = client.post(
        f"/runs/{payload['run_id']}/verify-replay",
        json={"dataset_version": "v1", "data": {strategy_id: bars()}},
    )
    assert replay.status_code == 200
    assert replay.json()["valid"] is True

    certificate = client.post(
        f"/runs/{payload['run_id']}/certificate",
        json={"dataset_version": "v1", "data": {strategy_id: bars()}},
    )
    assert certificate.status_code == 200
    certificate_payload = certificate.json()
    assert certificate_payload["verified"] is True
    assert len(certificate_payload["certificate_id"]) == 24
    assert certificate_payload["provenance_graph_fingerprint"]
    assert certificate_payload["ledger_fingerprint"] == verification.json()["ledger_fingerprint"]

    graph = client.get(f"/runs/{payload['run_id']}/provenance-graph")
    assert graph.status_code == 200
    assert graph.json()["fingerprint"] == certificate_payload["provenance_graph_fingerprint"]
    assert any(node["kind"] == "promotion" for node in graph.json()["nodes"])
    registry.close()


def test_paper_replay_endpoint_exposes_admission_integrity(monkeypatch):
    monkeypatch.delenv("ATSF_API_KEY", raising=False)
    registry = ExperimentRegistry()
    strategy_id = seed(registry)
    client = TestClient(create_app(registry))
    response = client.post(
        "/portfolios/portfolio-1/paper-runs",
        json={"dataset_version": "v1", "data": {strategy_id: bars()}},
    )
    assert response.status_code == 201
    run_id = response.json()["run_id"]
    before_admission = client.get(f"/runs/{run_id}/paper-replay")
    assert before_admission.status_code == 200
    assert before_admission.json()["replayable"] is False
    assert "paper admission is missing" in before_admission.json()["reasons"]
    registry.close()


def test_replay_verification_rejects_changed_market_data(monkeypatch):
    monkeypatch.delenv("ATSF_API_KEY", raising=False)
    registry = ExperimentRegistry()
    strategy_id = seed(registry)
    client = TestClient(create_app(registry))
    response = client.post(
        "/portfolios/portfolio-1/paper-runs",
        json={"dataset_version": "v1", "data": {strategy_id: bars()}},
    )
    assert response.status_code == 201
    run_id = response.json()["run_id"]
    altered = bars()
    altered[2]["close"] = 9.0
    replay = client.post(
        f"/runs/{run_id}/verify-replay",
        json={"dataset_version": "v1", "data": {strategy_id: altered}},
    )
    assert replay.status_code == 200
    assert replay.json()["valid"] is False
    assert "data bundle" in replay.json()["reason"]


def test_paper_run_endpoint_rejects_unknown_portfolio_with_not_found(monkeypatch):
    monkeypatch.delenv("ATSF_API_KEY", raising=False)
    registry = ExperimentRegistry()
    client = TestClient(create_app(registry))
    response = client.post(
        "/portfolios/missing/paper-runs",
        json={"dataset_version": "v1", "data": {"missing": bars()}},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "portfolio not found"
    registry.close()


def test_paper_run_endpoint_rejects_invalid_market_data_with_bad_request(monkeypatch):
    monkeypatch.delenv("ATSF_API_KEY", raising=False)
    registry = ExperimentRegistry()
    strategy_id = seed(registry)
    client = TestClient(create_app(registry))
    invalid = bars()
    invalid[1]["high"] = 0.0
    response = client.post(
        "/portfolios/portfolio-1/paper-runs",
        json={"dataset_version": "v1", "data": {strategy_id: invalid}},
    )
    assert response.status_code == 400
    registry.close()
