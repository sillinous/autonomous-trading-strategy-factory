from atsf.catalog import promotion_ready_experiments
from atsf.dataset_bundle import DatasetBundleIdentity
from atsf.experiment import ExperimentResult, ExperimentSpec
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


def make_strategy(version: int) -> StrategySpec:
    return StrategySpec(
        name="catalog-test",
        version=version,
        universe=["TEST"],
        timeframe="1d",
        indicators=[Indicator(name="sma", source="close", period=5)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def test_catalog_is_fail_closed_and_ranks_paper_candidates():
    registry = ExperimentRegistry()
    registry.register_dataset(
        DatasetBundleIdentity("prices", "v1", ("TEST",), 1, "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        source="fixture",
    )
    entries = []
    for version, score, eligible, status in (
        (1, 0.8, True, "paper"),
        (2, 1.2, True, "paper"),
        (3, 9.9, False, "paper"),
        (4, 99.0, True, "research"),
    ):
        strategy = make_strategy(version)
        spec = ExperimentSpec(strategy, "prices", "v1", seed=version)
        result = ExperimentResult(spec.experiment_id, status, score=score)
        registry.save_experiment(spec, result)
        registry.save_evaluation_evidence(
            spec.experiment_id,
            {"promotion": {"eligible": eligible}},
        )
        entries.append(spec.experiment_id)

    ranked = promotion_ready_experiments(registry, "prices")
    assert [entry.score for entry in ranked] == [1.2, 0.8]
    assert ranked[0].experiment_id == entries[1]
    assert ranked[0].evidence["promotion"]["eligible"] is True
    assert len(promotion_ready_experiments(registry, "prices", limit=1)) == 1

    registry.save_evaluation_evidence(entries[1], {"promotion": {"eligible": False}})
    assert [entry.score for entry in promotion_ready_experiments(registry, "prices")] == [0.8]
    registry.close()
