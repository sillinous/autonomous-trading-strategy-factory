from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .monitoring import DegradationReport


class StrategyState(StrEnum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    HALTED = "halted"


class StrategyLifecycleStage(StrEnum):
    RESEARCH = "research"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    PAPER = "paper"
    DEGRADED = "degraded"
    RETIRED = "retired"
    REPLACED = "replaced"


_PROMOTION_TRANSITIONS: dict[StrategyLifecycleStage, frozenset[StrategyLifecycleStage]] = {
    StrategyLifecycleStage.RESEARCH: frozenset({StrategyLifecycleStage.VALIDATED, StrategyLifecycleStage.DEGRADED}),
    StrategyLifecycleStage.VALIDATED: frozenset({StrategyLifecycleStage.PROMOTED, StrategyLifecycleStage.DEGRADED}),
    StrategyLifecycleStage.PROMOTED: frozenset({StrategyLifecycleStage.PAPER, StrategyLifecycleStage.DEGRADED, StrategyLifecycleStage.RETIRED}),
    StrategyLifecycleStage.PAPER: frozenset({StrategyLifecycleStage.DEGRADED, StrategyLifecycleStage.RETIRED, StrategyLifecycleStage.REPLACED}),
    StrategyLifecycleStage.DEGRADED: frozenset({StrategyLifecycleStage.RESEARCH, StrategyLifecycleStage.RETIRED, StrategyLifecycleStage.REPLACED}),
    StrategyLifecycleStage.RETIRED: frozenset(),
    StrategyLifecycleStage.REPLACED: frozenset(),
}


@dataclass(frozen=True)
class PromotionLifecycleEvent:
    strategy_id: str
    previous: StrategyLifecycleStage
    current: StrategyLifecycleStage
    reason: str


def can_promote_stage(source: StrategyLifecycleStage, target: StrategyLifecycleStage) -> bool:
    return target in _PROMOTION_TRANSITIONS[source]


def transition_promotion_stage(strategy_id: str, source: StrategyLifecycleStage, target: StrategyLifecycleStage, *, reason: str) -> PromotionLifecycleEvent:
    if not strategy_id.strip():
        raise ValueError("strategy_id is required")
    if not reason.strip():
        raise ValueError("reason is required")
    if not can_promote_stage(source, target):
        raise ValueError(f"invalid lifecycle transition: {source.value} -> {target.value}")
    return PromotionLifecycleEvent(strategy_id, source, target, reason)


@dataclass(frozen=True)
class LifecycleEvent:
    previous: StrategyState
    current: StrategyState
    reason: str
    observations: int


class StrategyLifecycle:
    """Fail-closed state machine for paper strategy health."""

    def __init__(self) -> None:
        self.state = StrategyState.ACTIVE
        self.events: list[LifecycleEvent] = []

    def apply(self, report: DegradationReport) -> StrategyState:
        if self.state == StrategyState.HALTED:
            return self.state
        if report.degraded:
            previous = self.state
            self.state = StrategyState.DEGRADED
            self.events.append(
                LifecycleEvent(previous, self.state, "; ".join(report.reasons), report.observations)
            )
        return self.state

    def halt(self, reason: str, observations: int = 0) -> StrategyState:
        if self.state != StrategyState.HALTED:
            previous = self.state
            self.state = StrategyState.HALTED
            self.events.append(LifecycleEvent(previous, self.state, reason, observations))
        return self.state

    @property
    def can_trade(self) -> bool:
        return self.state == StrategyState.ACTIVE
