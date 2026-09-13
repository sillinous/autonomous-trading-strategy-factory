import pandas as pd
from fastapi.testclient import TestClient

from atsf.api import create_app
from atsf.dataset_bundle import DatasetBundleIdentity, bundle_identity
from atsf.experiment import ExperimentResult, ExperimentSpec
from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_store import LifecycleStore
from atsf.lineage import LineageRecord
from atsf.registry import ExperimentRegistry
from atsf.strategy import Comparator, Condition, PositionSizing, RiskLimits, Signal, StrategySpec


def strategy() -> StrategySpec:
    return StrategySpec(
        name="admission-api-test", version=1, universe=["TEST"], timeframe="1d",
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=2)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=1)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def bars(count: int = 4) -> list[dict]:
    values = [1.0 + (index % 3) for index in range(count)]
    return [{"timestamp": timestamp.isoformat(), "open": value, "high": value, "low": value, "close": value, "volume": 1.0}
            for timestamp, value in zip(pd.date_range("2026-01-01", periods=count), values)]


def seed(registry: ExperimentRegistry) -> str:
    item = strategy()
    strategy_id = registry.save_strategy(item)
    registry.save_lineage(LineageRecord(strategy_id=strategy_id, generation=0))
    LifecycleStore(registry._connection).save(strategy_id, StrategyLifecycleStage.PROMOTED, reason="promotion gate")
    registry.register_dataset(DatasetBundleIdentity("prices", "v1", ("TEST",), 1, "2026-01-01T00:00:00", "2026-01-01T00:00:00"), source="fixture")
    spec = ExperimentSpec(item, "prices", "v1", seed=1)
    registry.save_experiment(spec, ExperimentResult(spec.experiment_id, "paper", score=1.0))
    registry.save_evaluation_evidence(spec.experiment_id, {"promotion": {"stage": "promoted", "eligible": True, "reasons": []}})
    frame = pd.DataFrame(bars()).set_index("timestamp")
    bundle = bundle_identity({strategy_id: frame}, "prices", source="fixture", timeframe="1d")
    registry.save_portfolio("portfolio-1", {"dataset_id": "prices", "dataset_version": "v1", "data_bundle_version": bundle.version,
                                             "data_source": "fixture", "data_timeframe": "1d", "data_schema_version": "ohlcv.v1",
                                             "experiment_ids": {strategy_id: spec.experiment_id}}, {strategy_id: 1.0})
    return strategy_id


def test_paper_admission_api_requires_verified_certificate_and_persists_admission():
    registry = ExperimentRegistry()
    strategy_id = seed(registry)
    client = TestClient(create_app(registry))
    run_response = client.post("/portfolios/portfolio-1/paper-runs", json={"dataset_version": "v1", "data": {strategy_id: bars()}})
    assert run_response.status_code == 201
    run_id = run_response.json()["run_id"]

    before_certificate = client.post(f"/runs/{run_id}/paper-admission", json={"strategy_id": strategy_id})
    assert before_certificate.status_code == 409
    assert "certificate" in str(before_certificate.json()["detail"]).lower()

    certificate = client.post(f"/runs/{run_id}/certificate", json={"dataset_version": "v1", "data": {strategy_id: bars()}})
    assert certificate.status_code == 200, certificate.json()

    admission = client.post(f"/runs/{run_id}/paper-admission", json={"strategy_id": strategy_id})
    assert admission.status_code == 200
    payload = admission.json()
    assert payload["admitted"] is True
    assert payload["source_stage"] == "promoted"
    assert payload["target_stage"] == "paper"
    assert payload["admission_id"]
    assert LifecycleStore(registry._connection).get(strategy_id).stage is StrategyLifecycleStage.PAPER

    replay = client.get(f"/runs/{run_id}/paper-replay")
    assert replay.status_code == 200
    assert replay.json()["replayable"] is True
    assert replay.json()["admission_id"] == payload["admission_id"]

    repeated = client.post(f"/runs/{run_id}/paper-admission", json={"strategy_id": strategy_id})
    assert repeated.status_code == 200
    assert repeated.json()["admission_id"] == payload["admission_id"]
    registry.close()
