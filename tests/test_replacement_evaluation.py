import pandas as pd

from atsf.replacement_evaluation import evaluate_replacements
from atsf.replacement_research import ReplacementResearchResult
from atsf.research_queue import ResearchReason, ResearchRequest
from atsf.research_registry import ResearchRequestStore
from atsf.registry import ExperimentRegistry
from atsf.replacement_research import generate_replacements


def data() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=60, freq="D")
    close = [100 + ((i % 9) * 1.5) + ((i // 9) % 2) * 3 for i in range(60)]
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


def test_replacements_use_the_normal_deterministic_evaluation_pipeline():
    registry = ExperimentRegistry()
    store = ResearchRequestStore(registry)
    request = ResearchRequest(
        request_id="evaluation-1",
        source_strategy_id="failed-strategy",
        reason=ResearchReason.DEGRADED,
        priority=0,
        constraints=(),
    )
    store.save(request)
    research = generate_replacements(request, ["TEST"], request_store=store)
    result = evaluate_replacements(research, data(), "replacement-data", "v1", seed=11)

    assert result.request_id == request.request_id
    assert len(result.evaluations) == len(research.candidates)
    assert all(evaluation.experiment.experiment_id for evaluation in result.evaluations)
    assert set(result.eligible_strategy_ids).issubset(
        {evaluation.candidate_id for evaluation in result.evaluations}
    )
    assert all(
        evaluation.promotion.stage in {"reject", "paper"}
        for evaluation in result.evaluations
    )
    registry.close()
