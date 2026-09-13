from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import pandas as pd

from .control_plane import StrategyControlPlane
from .feedback_provenance import build_feedback_provenance
from .feedback_registry import FeedbackEventStore
from .feedback_loop import FeedbackAction
from .lifecycle import StrategyLifecycleStage, StrategyState
from .lifecycle_store import LifecycleStore
from .monitoring import DegradationPolicy, DegradationReport, assess_degradation
from .registry import ExperimentRegistry
from .research_queue import ResearchQueue, ResearchReason, ResearchRequest
from .research_registry import ResearchRequestStore


@dataclass(frozen=True)
class PaperHealthDecision:
    strategy_id: str
    stage: StrategyLifecycleStage
    degraded: bool
    report: DegradationReport
    research_request: ResearchRequest | None
    feedback_event_id: str | None


def _request(report: DegradationReport, strategy_id: str) -> ResearchRequest:
    request_id = sha256(
        f"{strategy_id}:degraded:{report.observations}:{','.join(report.reasons)}".encode("utf-8")
    ).hexdigest()[:16]
    return ResearchRequest(
        request_id=request_id,
        source_strategy_id=strategy_id,
        reason=ResearchReason.DEGRADED,
        priority=0,
        constraints=("avoid_known_failure",),
    )


def assess_paper_health(
    registry: ExperimentRegistry,
    strategy_id: str,
    equity: pd.Series,
    *,
    policy: DegradationPolicy | None = None,
    queue: ResearchQueue | None = None,
) -> PaperHealthDecision:
    """Assess paper equity and durably transition PAPER to DEGRADED on failure.

    This boundary uses the durable lifecycle as the authority, persists the
    replacement request and feedback provenance, and is idempotent across restarts.
    It never enables live brokerage execution.
    """
    if not strategy_id.strip():
        raise ValueError("strategy_id is required")

    lifecycle = LifecycleStore(registry._connection)
    current = lifecycle.get(strategy_id)
    if current is None:
        raise ValueError("strategy lifecycle is missing")
    if current.stage not in {StrategyLifecycleStage.PAPER, StrategyLifecycleStage.DEGRADED}:
        raise ValueError("paper health requires PAPER or DEGRADED lifecycle stage")

    report = assess_degradation(equity, policy)
    if not report.degraded:
        return PaperHealthDecision(strategy_id, current.stage, False, report, None, None)

    request = _request(report, strategy_id)
    requests = ResearchRequestStore(registry)
    requests.save(request)
    if queue is not None:
        queue.enqueue(request)

    if current.stage is StrategyLifecycleStage.PAPER:
        updated = lifecycle.transition(
            strategy_id,
            StrategyLifecycleStage.PAPER,
            StrategyLifecycleStage.DEGRADED,
            reason="paper health degradation: " + "; ".join(report.reasons),
        )
        control_plane = StrategyControlPlane(registry)
        control_plane.record(strategy_id, StrategyState.DEGRADED, "; ".join(report.reasons))
        action = FeedbackAction(strategy_id, StrategyState.DEGRADED, request)
        provenance = build_feedback_provenance(
            strategy_id,
            action,
            report,
            previous_state=StrategyState.ACTIVE.value,
        )
        FeedbackEventStore(registry).save(provenance)
        return PaperHealthDecision(
            strategy_id,
            updated.stage,
            True,
            report,
            request,
            provenance.event_id,
        )

    existing = FeedbackEventStore(registry).list_for_strategy(strategy_id)
    event_id = existing[-1]["event_id"] if existing else None
    return PaperHealthDecision(strategy_id, current.stage, True, report, request, event_id)
