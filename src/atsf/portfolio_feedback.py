from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .portfolio_intelligence import PortfolioIntelligenceResult
from .research_director import ResearchSignal
from .research_queue import ResearchReason


@dataclass(frozen=True)
class PortfolioFeedbackResult:
    generation: int
    portfolio_volatility: float
    admission_passed: bool
    average_correlation: float
    signals: tuple[ResearchSignal, ...]


def _bounded(value: float, *, default: float = 0.0) -> float:
    if not isfinite(value):
        return default
    return min(1.0, max(0.0, value))


def build_portfolio_feedback(
    portfolio: PortfolioIntelligenceResult,
    *,
    generation: int = 0,
) -> PortfolioFeedbackResult:
    """Convert portfolio admission diagnostics into deterministic research signals."""
    if not isinstance(generation, int) or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    volatility = portfolio.portfolio_volatility
    if not isfinite(volatility) or volatility < 0:
        raise ValueError("portfolio volatility must be finite and non-negative")
    correlation = portfolio.selection.average_correlation
    if not isfinite(correlation):
        raise ValueError("average correlation must be finite")
    correlation_pressure = _bounded(max(0.0, correlation))
    risk_pressure = _bounded(volatility / max(volatility, 1.0))

    if not portfolio.selection.selected:
        raise ValueError("portfolio feedback requires selected strategies")

    signals: list[ResearchSignal] = []
    for strategy_id in portfolio.selection.selected:
        if not portfolio.admission_passed:
            reason = ResearchReason.DEGRADED
            robustness = 0.0
            capacity_gap = 0.0
            uncertainty = max(correlation_pressure, risk_pressure)
        elif correlation_pressure >= 0.75:
            reason = ResearchReason.DIVERSIFICATION
            robustness = 1.0
            capacity_gap = 0.0
            uncertainty = correlation_pressure
        else:
            reason = ResearchReason.CAPACITY
            robustness = 1.0
            capacity_gap = 1.0
            uncertainty = correlation_pressure
        signals.append(
            ResearchSignal(
                strategy_id=str(strategy_id),
                reason=reason,
                fitness=0.0,
                robustness=robustness,
                novelty=_bounded(1.0 - correlation_pressure),
                uncertainty=_bounded(uncertainty),
                capacity_gap=capacity_gap,
            )
        )

    signals.sort(key=lambda signal: (signal.reason, signal.strategy_id))
    return PortfolioFeedbackResult(
        generation=generation,
        portfolio_volatility=volatility,
        admission_passed=bool(portfolio.admission_passed),
        average_correlation=correlation,
        signals=tuple(signals),
    )
