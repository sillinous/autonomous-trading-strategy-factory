from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .lifecycle import StrategyLifecycleStage
from .lifecycle_integration import synchronize_candidate_lifecycle
from .lifecycle_store import LifecycleStore, PersistedLifecycle
from .registry import ExperimentRegistry
from .research import ResearchRunResult, run_research
from .strategy import StrategySpec


@dataclass(frozen=True)
class ResearchLifecycleResult:
    """Research output plus the durable lifecycle states produced by its evaluations."""

    research: ResearchRunResult
    lifecycle: tuple[PersistedLifecycle, ...]


def synchronize_research_lifecycle(
    registry: ExperimentRegistry,
    research: ResearchRunResult,
) -> tuple[PersistedLifecycle, ...]:
    """Synchronize every completed evaluation into durable lifecycle state."""
    store = LifecycleStore(registry._connection)
    states: list[PersistedLifecycle] = []
    for generation in research.generations:
        for evaluation in generation.evaluations:
            states.append(synchronize_candidate_lifecycle(store, evaluation))
    return tuple(states)


def run_research_with_lifecycle(
    seeds: list[StrategySpec],
    data: pd.DataFrame,
    *,
    registry: ExperimentRegistry | None = None,
    **kwargs: object,
) -> ResearchLifecycleResult:
    """Run the research factory and durably synchronize its lifecycle outcomes."""
    owned = registry is None
    store = registry or ExperimentRegistry()
    try:
        result = run_research(seeds, data, registry=store, **kwargs)
        states = synchronize_research_lifecycle(store, result)
        return ResearchLifecycleResult(result, states)
    finally:
        if owned:
            store.close()


def lifecycle_stage(
    registry: ExperimentRegistry,
    strategy_id: str,
) -> StrategyLifecycleStage | None:
    """Return an integrity-checked persisted stage for supervisory callers."""
    record = LifecycleStore(registry._connection).get(strategy_id)
    return None if record is None else record.stage
