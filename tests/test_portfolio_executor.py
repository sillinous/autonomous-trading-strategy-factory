import pandas as pd
import pytest

from atsf.experiment import ExperimentResult, ExperimentSpec
from atsf.portfolio_executor import execute_persisted_portfolio
from atsf.registry import ExperimentRegistry
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)


def make_strategy(name: str, period: int = 3) -> StrategySpec:
    return StrategySpec(
        name=name,
        version=1,
        universe=["TEST"],
        timeframe="1d",
        indicators=[Indicator(name="sma", source="close", period=period)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=1.0, max_position=1.0),
        risk=RiskLimits(max_position=1.0),
    )


def make_data() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=12, freq="D")
    close = [100, 101, 102, 103, 104, 103, 102, 101, 102, 104, 103, 105]
    return pd.DataFrame({"close": close}, index=index)


def seed_persisted_portfolio(registry: ExperimentRegistry) -> tuple[str, str]:
    strategies = [make_strategy("one", 3), make_strategy("two", 4)]
    ids = [registry.save_strategy(strategy) for strategy in strategies]
    experiment_ids = {}
    for strategy_id, strategy in zip(ids, strategies):
        spec = ExperimentSpec(strategy, "prices", "v1", seed=1)
        result = ExperimentResult(spec.experiment_id, "paper", score=1.0)
        registry.save_experiment(spec, result)
        registry.save_evaluation_evidence(
            spec.experiment_id,
            {"promotion": {"stage": "paper", "eligible": True, "reasons": []}},
        )
        experiment_ids[strategy_id] = spec.experiment_id
    registry.save_portfolio(
        "portfolio-1",
        {"dataset_id": "prices", "dataset_version": "v1", "experiment_ids": experiment_ids},
        {ids[0]: 0.6, ids[1]: 0.3},
    )
    return ids[0], ids[1]


def test_executor_uses_persisted_members_and_saves_run():
    registry = ExperimentRegistry()
    first, second = seed_persisted_portfolio(registry)
    data = {first: make_data(), second: make_data()}

    result = execute_persisted_portfolio(registry, "portfolio-1", data, dataset_version="v1")

    assert result.identity.portfolio_id == "portfolio-1"
    assert result.identity.dataset_version == "v1"
    assert result.paper.final_equity > 0
    assert {item.strategy_id for item in result.attribution.contributions} == {first, second}
    stored = registry.get_portfolio_run(result.identity.run_id)
    assert stored is not None
    assert stored["final_equity"] == pytest.approx(result.paper.final_equity)
    assert len(stored["attribution"]) == 2
    registry.close()


def test_executor_rejects_dataset_version_mismatch():
    registry = ExperimentRegistry()
    first, second = seed_persisted_portfolio(registry)
    data = {first: make_data(), second: make_data()}
    with pytest.raises(ValueError, match="dataset_version"):
        execute_persisted_portfolio(registry, "portfolio-1", data, dataset_version="wrong")
    registry.close()
