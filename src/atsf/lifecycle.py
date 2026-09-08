from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .monitoring import DegradationReport


class StrategyState(StrEnum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    HALTED = "halted"


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
