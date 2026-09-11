import pandas as pd
import pytest

from atsf.dataset_bundle import DatasetBundleIdentity, bundle_identity
from atsf.experiment import ExperimentResult, ExperimentSpec
from atsf.portfolio_audit import PortfolioAuditEvent
from atsf.portfolio_executor import execute_persisted_portfolio
from atsf.registry import ExperimentRegistry
from atsf.strategy import Comparator, Condition, Indicator, PositionSizing, RiskLimits, Signal, StrategySpec


def make_strategy(name: str, period: int = 3) -> StrategySpec:
    return StrategySpec(
        name=name, version=1, universe=["TEST"], timeframe="1d",
        indicators=[Indicator(name="sma", source="close", period=period)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=1.0, max_position=1.0),
        risk=RiskLimits(max_position=1.0),
    )


def make_data(offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=12, freq="D")
    close = [100, 101, 102, 103, 104, 103, 102, 101, 102, 104, 103, 105]
    close = [value + offset for value in close]
    return pd.DataFrame(
        {
            "open": [value - 0.5 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1000] * len(close),
        },
        index=index,
    )


def seed_persisted_portfolio(registry: ExperimentRegistry) -> tuple[str, str, str, str]:
    strategies = [make_strategy("one", 3), make_strategy("two", 4)]
    ids = [registry.save_strategy(strategy) for strategy in strategies]
    data = {ids[0]: make_data(), ids[1]: make_data()}
    data_bundle_version = bundle_identity(data, "prices", source="fixture", timeframe="1d").version
    registry.register_dataset(
        DatasetBundleIdentity("prices", "v1", ("TEST",), 12, "2026-01-01T00:00:00", "2026-01-12T00:00:00"),
        source="fixture",
    )
    experiment_ids = {}
    for strategy_id, strategy in zip(ids, strategies):
        spec = ExperimentSpec(strategy, "prices", "v1", seed=1)
        result = ExperimentResult(spec.experiment_id, "paper", score=1.0)
        registry.save_experiment(spec, result)
        registry.save_evaluation_evidence(spec.experiment_id, {"promotion": {"stage": "paper", "eligible": True, "reasons": []}})
        experiment_ids[strategy_id] = spec.experiment_id
    registry.save_portfolio(
        "portfolio-1",
        {
            "dataset_id": "prices",
            "dataset_version": "v1",
            "data_bundle_version": data_bundle_version,
            "data_source": "fixture",
            "data_timeframe": "1d",
            "data_schema_version": "ohlcv.v1",
            "experiment_ids": experiment_ids,
        },
        {ids[0]: 0.6, ids[1]: 0.3},
    )
    return ids[0], ids[1], "v1", data_bundle_version


def test_executor_uses_persisted_members_and_saves_run():
    registry = ExperimentRegistry()
    first, second, dataset_version, bundle_version = seed_persisted_portfolio(registry)
    result = execute_persisted_portfolio(registry, "portfolio-1", {first: make_data(), second: make_data()}, dataset_version=dataset_version)
    assert result.identity.portfolio_id == "portfolio-1"
    assert result.identity.dataset_version == dataset_version
    assert result.identity.execution_fingerprint
    assert result.paper.final_equity > 0
    assert {item.strategy_id for item in result.attribution.contributions} == {first, second}
    stored = registry.get_portfolio_run(result.identity.run_id)
    assert stored is not None
    assert stored["final_equity"] == pytest.approx(result.paper.final_equity)
    assert len(stored["attribution"]) == 2
    assert [item["sequence"] for item in stored["audit_events"]] == list(range(len(result.audit_events)))
    assert stored["provenance"]["dataset_id"] == "prices"
    assert stored["provenance"]["dataset_version"] == dataset_version
    assert stored["provenance"]["data_bundle_version"] == bundle_version
    assert stored["provenance"]["execution_fingerprint"] == result.identity.execution_fingerprint
    assert stored["provenance"]["execution_config"]["initial_cash"] == 100_000.0
    registry.close()


def test_executor_is_idempotency_guarded_and_data_changes_are_rejected():
    registry = ExperimentRegistry()
    first, second, dataset_version, _bundle_version = seed_persisted_portfolio(registry)
    data = {first: make_data(), second: make_data()}
    result = execute_persisted_portfolio(registry, "portfolio-1", data, dataset_version=dataset_version)
    with pytest.raises(ValueError, match="already exists"):
        execute_persisted_portfolio(registry, "portfolio-1", data, dataset_version=dataset_version)
    with pytest.raises(ValueError, match="data bundle version"):
        execute_persisted_portfolio(registry, "portfolio-1", {first: make_data(), second: make_data(1.0)}, dataset_version=dataset_version)
    assert result.identity.execution_fingerprint
    registry.close()


def test_executor_rejects_dataset_version_mismatch():
    registry = ExperimentRegistry()
    first, second, _dataset_version, _bundle_version = seed_persisted_portfolio(registry)
    with pytest.raises(ValueError, match="dataset_version"):
        execute_persisted_portfolio(registry, "portfolio-1", {first: make_data(), second: make_data()}, dataset_version="wrong")
    registry.close()


def test_atomic_execution_rejects_invalid_audit_without_partial_rows():
    registry = ExperimentRegistry()
    first, second, dataset_version, bundle_version = seed_persisted_portfolio(registry)
    audit = [PortfolioAuditEvent(1, first, "buy", "2026-01-01T00:00:00", 1.0, 100.0, 0.01)]
    attribution = [
        {"strategy_id": first, "return_contribution": 0.01, "risk_contribution": 0.02},
        {"strategy_id": second, "return_contribution": 0.01, "risk_contribution": 0.02},
    ]
    with pytest.raises(ValueError, match="contiguous"):
        registry.save_portfolio_execution(
            "atomic-failure",
            "portfolio-1",
            100_000.0,
            False,
            None,
            attribution,
            dataset_id="prices",
            dataset_version=dataset_version,
            data_bundle_version=bundle_version,
            execution_fingerprint="fingerprint",
            execution_config={"initial_cash": 100_000.0},
            audit_events=audit,
        )
    assert registry.get_portfolio_run("atomic-failure") is None
    assert registry._connection.execute("SELECT COUNT(*) FROM portfolio_run_provenance WHERE run_id = ?", ("atomic-failure",)).fetchone()[0] == 0
    assert registry._connection.execute("SELECT COUNT(*) FROM portfolio_attribution WHERE run_id = ?", ("atomic-failure",)).fetchone()[0] == 0
    assert registry._connection.execute("SELECT COUNT(*) FROM portfolio_audit_events WHERE run_id = ?", ("atomic-failure",)).fetchone()[0] == 0
    registry.close()


def test_registry_audit_ledger_rejects_gaps_and_non_members():
    registry = ExperimentRegistry()
    first, _second, dataset_version, bundle_version = seed_persisted_portfolio(registry)
    registry.save_portfolio_run(
        "manual-run", "portfolio-1", 100_000.0, False, None, [],
        dataset_id="prices", dataset_version=dataset_version,
        data_bundle_version=bundle_version, execution_fingerprint="manual",
        execution_config={"initial_cash": 100_000.0},
    )
    event = PortfolioAuditEvent(1, first, "buy", "2026-01-01T00:00:00", 1.0, 100.0, 0.01)
    with pytest.raises(ValueError, match="contiguous"):
        registry.save_portfolio_audit_events("manual-run", [event])
    unknown = PortfolioAuditEvent(0, "not-a-member", "buy", "2026-01-01T00:00:00", 1.0, 100.0, 0.01)
    with pytest.raises(ValueError, match="persisted portfolio members"):
        registry.save_portfolio_audit_events("manual-run", [unknown])
    registry.close()
