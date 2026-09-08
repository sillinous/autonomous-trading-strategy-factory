from __future__ import annotations

from dataclasses import dataclass

from .promotion import PromotionDecision


@dataclass(frozen=True)
class PaperRiskState:
    peak_equity: float
    halted: bool
    reason: str | None = None


class PaperRiskController:
    """Fail-closed runtime guard for paper execution."""

    def __init__(self, max_drawdown: float | None = None) -> None:
        if max_drawdown is not None and not 0 < max_drawdown < 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        self.max_drawdown = max_drawdown
        self._state = PaperRiskState(peak_equity=0.0, halted=False)

    @property
    def state(self) -> PaperRiskState:
        return self._state

    def check(self, equity: float) -> bool:
        if equity <= 0:
            self._state = PaperRiskState(self._state.peak_equity, True, "non-positive equity")
            return False
        peak = max(self._state.peak_equity, equity)
        drawdown = 1.0 - equity / peak if peak else 0.0
        if self.max_drawdown is not None and drawdown >= self.max_drawdown:
            self._state = PaperRiskState(peak, True, "maximum drawdown breached")
            return False
        self._state = PaperRiskState(peak, False, None)
        return True


def require_paper_eligibility(decision: PromotionDecision) -> None:
    if not decision.eligible or decision.stage not in {"paper", "live"}:
        raise PermissionError("strategy is not eligible for paper execution")
