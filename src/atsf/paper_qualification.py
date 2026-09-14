from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import pandas as pd

from .lifecycle import StrategyLifecycleStage
from .monitoring import DegradationPolicy, DegradationReport, assess_degradation


class PaperQualificationDecision(str, Enum):
    ELIGIBLE_FOR_LIVE_REVIEW = "eligible_for_live_review"
    REJECT = "reject"


@dataclass(frozen=True)
class PaperQualificationPolicy:
    """Conservative paper-stage gates; this layer never authorizes live trading."""

    min_observations: int = 60
    min_total_return: float = 0.0
    max_drawdown: float = 0.15
    max_volatility: float = 0.08

    def __post_init__(self) -> None:
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2")
        if not 0.0 < self.max_drawdown < 1.0:
            raise ValueError("max_drawdown must be between 0 and 1")
        if self.max_volatility < 0.0:
            raise ValueError("max_volatility cannot be negative")
        for name, value in (("min_total_return", self.min_total_return),):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class PaperQualification:
    strategy_id: str
    decision: PaperQualificationDecision
    execution_authority: bool
    report: DegradationReport
    reasons: tuple[str, ...]


def qualify_paper_strategy(
    strategy_id: str,
    stage: StrategyLifecycleStage,
    equity: pd.Series,
    *,
    policy: PaperQualificationPolicy | None = None,
) -> PaperQualification:
    """Determine whether PAPER evidence warrants a separate live review."""
    policy = policy or PaperQualificationPolicy()
    reasons: list[str] = []
    if not strategy_id.strip():
        reasons.append("strategy_id is required")
    if stage is not StrategyLifecycleStage.PAPER:
        reasons.append("strategy must remain in PAPER lifecycle stage")

    report: DegradationReport
    try:
        report = assess_degradation(
            equity,
            DegradationPolicy(
                max_drawdown=policy.max_drawdown,
                min_return=policy.min_total_return,
                max_volatility=policy.max_volatility,
                min_observations=policy.min_observations,
            ),
        )
    except (TypeError, ValueError) as exc:
        report = DegradationReport(reasons=(f"paper evidence invalid: {exc}",))
        reasons.append(report.reasons[0])
    else:
        if report.degraded:
            reasons.extend(report.reasons)
        if report.total_return < policy.min_total_return:
            reasons.append("paper total return is below qualification threshold")
        if report.observations < policy.min_observations:
            reasons.append("paper observation count is below qualification threshold")

    decision = (
        PaperQualificationDecision.ELIGIBLE_FOR_LIVE_REVIEW
        if not reasons
        else PaperQualificationDecision.REJECT
    )
    return PaperQualification(
        strategy_id=strategy_id,
        decision=decision,
        execution_authority=False,
        report=report,
        reasons=tuple(dict.fromkeys(reasons)),
    )
