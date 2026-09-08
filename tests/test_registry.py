from atsf.experiment import ExperimentResult, ExperimentSpec
from atsf.lineage import LineageRecord
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


def make_strategy() -> StrategySpec:
    return StrategySpec(
        name="registry-test",
        version=1,
        universe=["TEST"],
        timeframe="1d",
        indicators=[Indicator(name="sma", source="close", period=5)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def test_registry_round_trips_strategy_and_lineage():
    registry = ExperimentRegistry()
    strategy = make_strategy()
    identifier = registry.save_strategy(strategy)
    assert registry.get_strategy(identifier) == strategy

    lineage = LineageRecord(
        strategy_id=identifier,
        generation=2,
        parent_ids=("parent-1",),
        operator="mutate_threshold",
        parameters={"factor": 1.05},
    )
    registry.save_lineage(lineage)
    assert registry.get_lineage(identifier) == lineage
    registry.close()


def test_registry_persists_experiment_metadata():
    registry = ExperimentRegistry()
    strategy = make_strategy()
    spec = ExperimentSpec(strategy, "prices", "v1", seed=7)
    result = ExperimentResult(spec.experiment_id, "paper", score=1.2)
    registry.save_experiment(spec, result)

    row = registry._connection.execute(
        "SELECT * FROM experiments WHERE experiment_id = ?", (spec.experiment_id,)
    ).fetchone()
    assert row["strategy_id"] == registry.save_strategy(strategy)
    assert row["dataset_version"] == "v1"
    assert row["seed"] == 7
    assert row["status"] == "paper"
    registry.close()
