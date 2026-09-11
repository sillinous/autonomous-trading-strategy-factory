from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .feedback_loop import FeedbackAction
from .monitoring import DegradationReport
from .research_queue import ResearchRequest


@dataclass(frozen=True)
class FeedbackProvenance:
    event_id: str
    strategy_id: str
    previous_state: str
    resulting_state: str
    report_fingerprint: str
    research_request_id: str | None
    research_fingerprint: str | None
    fingerprint: str


def _fingerprint(payload: Any, length: int = 24) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def build_feedback_provenance(
    strategy_id: str,
    action: FeedbackAction,
    report: DegradationReport,
    *,
    previous_state: str,
) -> FeedbackProvenance:
    if not strategy_id:
        raise ValueError("strategy_id cannot be empty")
    report_fingerprint = _fingerprint({
        "strategy_id": strategy_id,
        "observations": report.observations,
        "total_return": report.total_return,
        "max_drawdown": report.max_drawdown,
        "volatility": report.volatility,
        "reasons": tuple(report.reasons),
    })
    request = action.research_request
    research_fingerprint = None
    request_id = None
    if request is not None:
        request_id = request.request_id
        research_fingerprint = _fingerprint({
            "request_id": request.request_id,
            "source_strategy_id": request.source_strategy_id,
            "reason": request.reason,
            "priority": request.priority,
            "constraints": request.constraints,
        })
    payload = {
        "strategy_id": strategy_id,
        "previous_state": previous_state,
        "resulting_state": action.state,
        "report_fingerprint": report_fingerprint,
        "research_request_id": request_id,
        "research_fingerprint": research_fingerprint,
    }
    return FeedbackProvenance(
        event_id=_fingerprint({"type": "feedback", **payload}),
        strategy_id=strategy_id,
        previous_state=previous_state,
        resulting_state=str(action.state),
        report_fingerprint=report_fingerprint,
        research_request_id=request_id,
        research_fingerprint=research_fingerprint,
        fingerprint=_fingerprint(payload),
    )


def verify_feedback_provenance(event: FeedbackProvenance) -> bool:
    payload = {
        "strategy_id": event.strategy_id,
        "previous_state": event.previous_state,
        "resulting_state": event.resulting_state,
        "report_fingerprint": event.report_fingerprint,
        "research_request_id": event.research_request_id,
        "research_fingerprint": event.research_fingerprint,
    }
    return event.fingerprint == _fingerprint(payload)
