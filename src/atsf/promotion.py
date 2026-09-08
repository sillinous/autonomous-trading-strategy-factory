from __future__ import annotations

from dataclasses import dataclass

from .evaluation import WalkForwardEvaluation
from .robustness import MonteCarloResult


@dataclass(frozen=True)
class PromotionPolicy:
    min_oos_sharpe: float = 0.5
    max_oos_drawdown: float = 0.25
    min_monte_carlo_pass_rate: float = 0.95
    min_monte_carlo_lower_return: float = -0.10
    require_walk_forward_pass: bool = True


@dataclass(frozen=True)
class PromotionDecision:
    stage: str
    eligible: bool
    reasons: tuple[str, ...]


def research_to_paper(
    evaluation: WalkForwardEvaluation,
    monte_carlo: MonteCarloResult,
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    """Apply hard deterministic gates before allowing a strategy into paper trading."""
    policy = policy or PromotionPolicy()
    reasons: list[str] = []
    if policy.require_walk_forward_pass and not evaluation.passed:
        reasons.append("walk-forward/OOS evaluation failed")
    if evaluation.oos_sharpe < policy.min_oos_sharpe:
        reasons.append("OOS Sharpe is below the promotion minimum")
    if evaluation.oos_drawdown > policy.max_oos_drawdown:
        reasons.append("OOS drawdown exceeds the promotion maximum")
    if monte_carlo.pass_rate < policy.min_monte_carlo_pass_rate:
        reasons.append("Monte Carlo pass rate is below the promotion minimum")
    if monte_carlo.lower_percentile_return < policy.min_monte_carlo_lower_return:
        reasons.append("Monte Carlo lower-percentile return is too weak")
    return PromotionDecision(
        stage="paper" if not reasons else "research",
        eligible=not reasons,
        reasons=tuple(reasons),
    )
