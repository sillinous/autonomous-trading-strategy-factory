from __future__ import annotations

from types import SimpleNamespace

from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_store import LifecycleStore
from atsf.registry import ExperimentRegistry
from atsf.research_lifecycle import (
    lifecycle_stage,
    synchronize_research_lifecycle,
)
from atsf.scheduler import GenerationResult
from atsf.research import ResearchRunResult


def _evaluation(strategy_id: str, *, validation_passed: bool, promotion_eligible: bool) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_id=strategy_id,
        validation_passed=validation_passed,
        promotion=SimpleNamespace(eligible=promotion_eligible),
    )


def _research(*evaluations: SimpleNamespace) -> ResearchRunResult:
    generation = GenerationResult(
        generation=1,
        evaluations=tuple(evaluations),
        survivors=(),
        next_population=(),
    )
    return ResearchRunResult(
        generations=(generation,),
        final_population=(),
        dataset_id="dataset",
        dataset_version="version",
    )


def test_synchronize_research_lifecycle_persists_promotion() -> None:
    registry = ExperimentRegistry()
    try:
        result = synchronize_research_lifecycle(
            registry,
            _research(_evaluation("strategy-1", validation_passed=True, promotion_eligible=True)),
        )
        assert result[-1].stage is StrategyLifecycleStage.PROMOTED
        assert lifecycle_stage(registry, "strategy-1") is StrategyLifecycleStage.PROMOTED
        history = LifecycleStore(registry._connection).history("strategy-1")
        assert [event.target_stage for event in history] == [
            StrategyLifecycleStage.RESEARCH,
            StrategyLifecycleStage.VALIDATED,
            StrategyLifecycleStage.PROMOTED,
        ]
    finally:
        registry.close()


def test_synchronize_research_lifecycle_degrades_failed_validation() -> None:
    registry = ExperimentRegistry()
    try:
        result = synchronize_research_lifecycle(
            registry,
            _research(_evaluation("strategy-2", validation_passed=False, promotion_eligible=False)),
        )
        assert result[-1].stage is StrategyLifecycleStage.DEGRADED
        assert lifecycle_stage(registry, "strategy-2") is StrategyLifecycleStage.DEGRADED
    finally:
        registry.close()


def test_synchronize_research_lifecycle_is_idempotent_for_existing_promotion() -> None:
    registry = ExperimentRegistry()
    try:
        research = _research(_evaluation("strategy-3", validation_passed=True, promotion_eligible=True))
        first = synchronize_research_lifecycle(registry, research)
        second = synchronize_research_lifecycle(registry, research)
        assert first[-1] == second[-1]
        assert len(LifecycleStore(registry._connection).history("strategy-3")) == 3
    finally:
        registry.close()
