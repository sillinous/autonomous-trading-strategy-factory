import pytest

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

    row = registry.get_experiment(spec.experiment_id)
    assert row is not None
    assert row["strategy_id"] == registry.save_strategy(strategy)
    assert row["dataset_id"] == "prices"
    assert row["dataset_version"] == "v1"
    assert row["seed"] == 7
    assert row["status"] == "paper"
    assert registry.list_experiments("prices")[0]["experiment_id"] == spec.experiment_id
    registry.close()


def test_registry_round_trips_evaluation_evidence():
    registry = ExperimentRegistry()
    strategy = make_strategy()
    spec = ExperimentSpec(strategy, "prices", "v1", seed=3)
    result = ExperimentResult(spec.experiment_id, "research", score=0.8)
    registry.save_experiment(spec, result)
    evidence = {
        "walk_forward": {"oos_sharpe": 0.8, "oos_drawdown": 0.1},
        "promotion": {"stage": "research", "eligible": False},
    }
    registry.save_evaluation_evidence(spec.experiment_id, evidence)
    assert registry.get_evaluation_evidence(spec.experiment_id) == evidence
    registry.close()


def test_registry_rejects_non_json_evidence():
    registry = ExperimentRegistry()
    with pytest.raises((TypeError, ValueError)):
        registry.save_evaluation_evidence("missing", {"bad": float("nan")})
    registry.close()


def test_registry_ranks_only_eligible_experiments():
    registry = ExperimentRegistry()
    strategy = make_strategy()
    for seed, status, score in ((1, "eligible", 0.5), (2, "eligible", 1.5), (3, "research", 99.0)):
        spec = ExperimentSpec(strategy.model_copy(update={"version": seed}), "prices", "v1", seed)
        registry.save_experiment(spec, ExperimentResult(spec.experiment_id, status, score=score))

    ranked = registry.rank_experiments("prices")
    assert [row["score"] for row in ranked] == [1.5, 0.5]
    assert registry.rank_experiments("prices", limit=1)[0]["score"] == 1.5
    with pytest.raises(ValueError):
        registry.rank_experiments(limit=0)
    registry.close()


def test_registry_round_trips_portfolio_definition_and_run_attribution():
    registry = ExperimentRegistry()
    strategy = make_strategy()
    strategy_id = registry.save_strategy(strategy)
    registry.save_portfolio(
        "portfolio-1",
        {"policy": "inverse_volatility", "version": 1},
        {strategy_id: 0.5},
    )
    assert registry.get_portfolio("portfolio-1") == {
        "portfolio_id": "portfolio-1",
        "definition": {"policy": "inverse_volatility", "version": 1},
        "members": {strategy_id: 0.5},
    }
    registry.save_portfolio_run(
        "run-1",
        "portfolio-1",
        101_000.0,
        False,
        None,
        [{"strategy_id": strategy_id, "return_contribution": 0.01, "risk_contribution": 0.02}],
    )
    row = registry._connection.execute(
        "SELECT * FROM portfolio_attribution WHERE run_id = ?", ("run-1",)
    ).fetchone()
    assert row["strategy_id"] == strategy_id
    assert row["return_contribution"] == pytest.approx(0.01)
    registry.close()
