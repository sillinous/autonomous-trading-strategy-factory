from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .evaluation import WalkForwardEvaluation
from .perturbation import PerturbationResult
from .robustness import MonteCarloResult, RegimeStabilityResult


@dataclass(frozen=True)
class PromotionPolicy:
    min_oos_sharpe: float = 0.5
    max_oos_drawdown: float = 0.25
    min_monte_carlo_pass_rate: float = 0.95
    min_monte_carlo_lower_return: float = -0.10
    min_perturbation_pass_rate: float = 0.80
    min_regime_stability: float = -0.05
    require_walk_forward_pass: bool = True


@dataclass(frozen=True)
class PromotionDecision:
    stage: str
    eligible: bool
    reasons: tuple[str, ...]


def research_to_paper(
    evaluation: WalkForwardEvaluation,
    monte_carlo: MonteCarloResult,
    perturbation: PerturbationResult | None = None,
    regime: RegimeStabilityResult | None = None,
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    """Apply deterministic robustness gates before paper trading."""
    policy = policy or PromotionPolicy()
    reasons: list[str] = []
    finite_metrics = (
        evaluation.oos_sharpe,
        evaluation.oos_drawdown,
        monte_carlo.pass_rate,
        monte_carlo.lower_percentile_return,
    )
    if not all(isfinite(value) for value in finite_metrics):
        reasons.append("non-finite promotion metric detected")
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
    if perturbation is None:
        reasons.append("parameter perturbation evidence")
    elif not isfinite(perturbation.pass_rate) or perturbation.pass_rate < policy.min_perturbation_pass_rate:
        reasons.append("parameter perturbation stability is below the promotion minimum")
    if regime is None:
        reasons.append("regime stability evidence")
    elif not isfinite(regime.score) or regime.score < policy.min_regime_stability:
        reasons.append("regime stability is below the promotion minimum")
    return PromotionDecision(
        stage="paper" if not reasons else "research",
        eligible=not reasons,
        reasons=tuple(reasons),
    )
