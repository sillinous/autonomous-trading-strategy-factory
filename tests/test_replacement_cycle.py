import pandas as pd
import pytest

from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_store import LifecycleStore
from atsf.replacement_cycle import ReplacementCycleStore, run_replacement_cycle
from atsf.research_queue import ResearchReason, ResearchRequest
from atsf.research_registry import ResearchRequestStore
from atsf.registry import ExperimentRegistry


def data() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=80, freq="D")
    close = [100 + ((i % 9) * 1.5) + ((i // 9) % 2) * 3 for i in range(80)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [v + 1 for v in close],
            "low": [v - 1 for v in close],
            "close": close,
            "volume": [1000] * len(close),
        },
        index=index,
    )


def test_replacement_cycle_persists_candidates_evaluations_and_admissions():
    registry = ExperimentRegistry()
    lifecycle = LifecycleStore(registry._connection)
    lifecycle.save("failed-strategy", StrategyLifecycleStage.DEGRADED, reason="paper health degradation")
    request = ResearchRequest(
        request_id="replacement-cycle-1",
        source_strategy_id="failed-strategy",
        reason=ResearchReason.DEGRADED,
        priority=0,
        constraints=("avoid_known_failure",),
    )
    ResearchRequestStore(registry).save(request)

    result = run_replacement_cycle(
        registry,
        request,
        data(),
        ["TEST"],
        "replacement-data",
        "v1",
        seed=17,
    )

    assert result.cycle_id
    assert len(result.research.candidates) == len(result.evaluation.evaluations)
    assert len(result.admissions) == len(result.evaluation.evaluations)
    assert all(
        evaluation.candidate_id == registry.get_lineage(evaluation.candidate_id).strategy_id
        for evaluation in result.evaluation.evaluations
        if registry.get_lineage(evaluation.candidate_id) is not None
    )
    assert ReplacementCycleStore(registry).get(request.request_id) is not None
    assert lifecycle.get("failed-strategy").stage is StrategyLifecycleStage.RESEARCH
    registry.close()


def test_replacement_cycle_requires_degraded_source_and_persisted_request():
    registry = ExperimentRegistry()
    lifecycle = LifecycleStore(registry._connection)
    lifecycle.save("active-strategy", StrategyLifecycleStage.PAPER, reason="paper admission")
    request = ResearchRequest(
        request_id="replacement-cycle-2",
        source_strategy_id="active-strategy",
        reason=ResearchReason.DEGRADED,
        priority=0,
        constraints=(),
    )
    with pytest.raises(ValueError, match="not durably persisted"):
        run_replacement_cycle(registry, request, data(), ["TEST"], "data", "v1")
    ResearchRequestStore(registry).save(request)
    with pytest.raises(ValueError, match="DEGRADED"):
        run_replacement_cycle(registry, request, data(), ["TEST"], "data", "v1")
    registry.close()
