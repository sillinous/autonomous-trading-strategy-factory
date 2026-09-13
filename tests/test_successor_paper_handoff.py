import pandas as pd

from fastapi.testclient import TestClient

from atsf.api import create_app
from atsf.dataset_bundle import DatasetBundleIdentity
from atsf.experiment import ExperimentResult, ExperimentSpec
from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_store import LifecycleStore
from atsf.lineage import LineageRecord
from atsf.registry import ExperimentRegistry
from atsf.strategy import Comparator, Condition, PositionSizing, RiskLimits, Signal, StrategySpec
from atsf.successor_paper_handoff import handoff_successor_to_paper


def strategy(name: str) -> StrategySpec:
    return StrategySpec(
        name=name,
        version=1,
        universe=["TEST"],
        timeframe="1d",
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=2)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=1)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def bars(count: int = 4) -> list[dict]:
    values = [1.0 + (index % 3) for index in range(count)]
    return [
        {"timestamp": timestamp.isoformat(), "open": value, "high": value, "low": value, "close": value, "volume": 1.0}
        for timestamp, value in zip(pd.date_range("2026-01-01", periods=count), values)
    ]


def seed(registry: ExperimentRegistry) -> str:
    item = strategy("successor-handoff")
    strategy_id = registry.save_strategy(item)
    parent_id = registry.save_strategy(strategy("parent"))
    registry.save_lineage(LineageRecord(strategy_id=parent_id, generation=0))
    registry.save_lineage(LineageRecord(strategy_id=strategy_id, generation=2, parent_ids=(parent_id,), operator="mutation"))
    LifecycleStore(registry._connection).save(strategy_id, StrategyLifecycleStage.PROMOTED, reason="successor promotion")
    registry.register_dataset(
        DatasetBundleIdentity("prices", "v1", ("TEST",), 1, "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        source="fixture",
    )
    spec = ExperimentSpec(item, "prices", "v1", seed=1)
    registry.save_experiment(spec, ExperimentResult(spec.experiment_id, "paper", score=1.0))
    registry.save_evaluation_evidence(
        spec.experiment_id,
        {"promotion": {"stage": "promoted", "eligible": True, "reasons": []}},
    )
    frame = pd.DataFrame(bars()).set_index("timestamp")
    from atsf.dataset_bundle import bundle_identity
    bundle = bundle_identity({strategy_id: frame}, "prices", source="fixture", timeframe="1d")
    registry.save_portfolio(
        "portfolio-successor",
        {"dataset_id": "prices", "dataset_version": "v1", "data_bundle_version": bundle.version,
         "data_source": "fixture", "data_timeframe": "1d", "data_schema_version": "ohlcv.v1",
         "experiment_ids": {strategy_id: spec.experiment_id}},
        {strategy_id: 1.0},
    )
    return strategy_id


def test_successor_handoff_requires_verified_paper_run_and_admits():
    registry = ExperimentRegistry()
    strategy_id = seed(registry)
    client = TestClient(create_app(registry))
    run = client.post(
        "/portfolios/portfolio-successor/paper-runs",
        json={"dataset_version": "v1", "data": {strategy_id: bars()}},
    )
    assert run.status_code == 201, run.json()
    run_id = run.json()["run_id"]
    certificate = client.post(
        f"/runs/{run_id}/certificate",
        json={"dataset_version": "v1", "data": {strategy_id: bars()}},
    )
    assert certificate.status_code == 200, certificate.json()

    decision = handoff_successor_to_paper(registry, strategy_id, run_id)
    assert decision.admitted is True
    assert decision.admission_id
    assert decision.reasons == ()
    assert LifecycleStore(registry._connection).get(strategy_id).stage is StrategyLifecycleStage.PAPER

    repeated = handoff_successor_to_paper(registry, strategy_id, run_id)
    assert repeated.admitted is True
    assert repeated.admission_id == decision.admission_id
    registry.close()


def test_successor_handoff_rejects_incomplete_multi_strategy_provenance():
    registry = ExperimentRegistry()
    strategy_id = seed(registry)
    second = registry.save_strategy(strategy("second"))
    registry.save_portfolio(
        "portfolio-multi",
        {"dataset_id": "prices", "dataset_version": "v1", "data_bundle_version": "bundle",
         "data_source": "fixture", "data_timeframe": "1d", "data_schema_version": "ohlcv.v1",
         "experiment_ids": {strategy_id: "missing", second: "missing"}},
        {strategy_id: 0.5, second: 0.5},
    )
    registry.save_portfolio_run(
        "multi-run", "portfolio-multi", 100000.0, False, None,
        [{"strategy_id": strategy_id, "return_contribution": 0.0, "risk_contribution": 0.0}],
    )
    decision = handoff_successor_to_paper(registry, strategy_id, "multi-run")
    assert decision.admitted is False
    assert decision.reasons
    assert LifecycleStore(registry._connection).get(strategy_id).stage is StrategyLifecycleStage.PROMOTED
    registry.close()
