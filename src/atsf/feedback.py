from __future__ import annotations

from dataclasses import dataclass

from .lifecycle import StrategyState
from .monitoring import DegradationReport


@dataclass(frozen=True)
class FeedbackDecision:
    state: StrategyState
    action: str
    reason: str


def decide_feedback(report: DegradationReport) -> FeedbackDecision:
    if report.degraded:
        return FeedbackDecision(
            StrategyState.DEGRADED,
            "retire_from_paper",
            "; ".join(report.reasons),
        )
    return FeedbackDecision(StrategyState.ACTIVE, "remain_in_paper", "health within policy")
