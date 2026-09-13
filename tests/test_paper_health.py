from __future__ import annotations

import pandas as pd
import pytest

from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_store import LifecycleStore
from atsf.monitoring import DegradationPolicy
from atsf.paper_health import assess_paper_health
from atsf.registry import ExperimentRegistry
from atsf.research_queue import ResearchQueue
from atsf.research_registry import ResearchRequestStore


def _equity() -> pd.Series:
    return pd.Series(
        [100.0] * 10 + [90.0, 80.0] + [79.0] * 10,
        index=pd.date_range("2025-01-01", periods=22),
    )


def _policy() -> DegradationPolicy:
    return DegradationPolicy(
        max_drawdown=0.10,
        min_return=-0.50,
        max_volatility=1.0,
        min_observations=20,
    )


def test_paper_health_degrades_durably_and_creates_research_work(tmp_path):
    registry = ExperimentRegistry(tmp_path / "atsf.sqlite3")
    LifecycleStore(registry._connection).save(
        "strategy-1", StrategyLifecycleStage.PAPER, reason="verified paper admission"
    )
    queue = ResearchQueue()

    decision = assess_paper_health(
        registry, "strategy-1", _equity(), policy=_policy(), queue=queue
    )

    assert decision.degraded
    assert decision.stage is StrategyLifecycleStage.DEGRADED
    assert decision.research_request is not None
    assert decision.feedback_event_id is not None
    assert len(queue) == 1
    assert LifecycleStore(registry._connection).get("strategy-1").stage is StrategyLifecycleStage.DEGRADED
    assert len(ResearchRequestStore(registry).list_for_strategy("strategy-1")) == 1
    assert len(LifecycleStore(registry._connection).history("strategy-1")) == 2
    registry.close()


def test_paper_health_is_restart_safe_and_does_not_append_duplicate_events(tmp_path):
    path = tmp_path / "atsf.sqlite3"
    first = ExperimentRegistry(path)
    LifecycleStore(first._connection).save(
        "strategy-1", StrategyLifecycleStage.PAPER, reason="verified paper admission"
    )
    first_decision = assess_paper_health(first, "strategy-1", _equity(), policy=_policy())
    first.close()

    second = ExperimentRegistry(path)
    second_decision = assess_paper_health(second, "strategy-1", _equity(), policy=_policy())

    assert second_decision.stage is StrategyLifecycleStage.DEGRADED
    assert second_decision.research_request == first_decision.research_request
    assert len(LifecycleStore(second._connection).history("strategy-1")) == 2
    assert len(ResearchRequestStore(second).list_for_strategy("strategy-1")) == 1
    second.close()


def test_paper_health_rejects_non_paper_lifecycle(tmp_path):
    registry = ExperimentRegistry(tmp_path / "atsf.sqlite3")
    LifecycleStore(registry._connection).save(
        "strategy-1", StrategyLifecycleStage.PROMOTED, reason="promotion gate"
    )
    with pytest.raises(ValueError, match="PAPER or DEGRADED"):
        assess_paper_health(registry, "strategy-1", _equity(), policy=_policy())
    registry.close()
