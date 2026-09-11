from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .evaluation import WalkForwardEvaluation
from .perturbation import PerturbationResult
from .robustness import MonteCarloResult, RegimeStabilityResult, RobustnessResult


@dataclass(frozen=True)
class PromotionPolicy:
    min_oos_sharpe: float = 0.5
    max_oos_drawdown: float = 0.25
    min_monte_carlo_pass_rate: float = 0.95
    min_monte_carlo_lower_return: float = -0.10
    min_perturbation_pass_rate: float = 0.80
    min_regime_stability: float = -0.05
    min_robustness_equity_ratio: float = 0.90
    require_walk_forward_pass: bool = True
    require_robustness: bool = True


@dataclass(frozen=True)
class PromotionDecision:
    stage: str
    eligible: bool
    reasons: tuple[str, ...]


def research_to_paper(evaluation: WalkForwardEvaluation, monte_carlo: MonteCarloResult,
                      perturbation: PerturbationResult | None = None,
                      regime: RegimeStabilityResult | None = None,
                      robustness: RobustnessResult | None = None,
                      policy: PromotionPolicy | None = None) -> PromotionDecision:
    # Preserve the historical positional form: (..., perturbation, regime, policy).
    if isinstance(robustness, PromotionPolicy) and policy is None:
        policy = robustness
        robustness = None
    policy = policy or PromotionPolicy()
    if not 0 < policy.min_robustness_equity_ratio <= 1:
        raise ValueError("min_robustness_equity_ratio must be in (0, 1]")
    reasons: list[str] = []
    finite_metrics = (evaluation.oos_sharpe, evaluation.oos_drawdown, monte_carlo.pass_rate, monte_carlo.lower_percentile_return)
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
    if policy.require_robustness and robustness is None:
        reasons.append("execution robustness evidence")
    elif robustness is not None:
        if not robustness.passed:
            reasons.extend(robustness.reasons)
        else:
            baseline_equity = float(robustness.baseline.equity.iloc[-1])
            stressed_equities = [float(scenario.equity.iloc[-1]) for name, scenario in robustness.scenarios]
            if baseline_equity <= 0 or not isfinite(baseline_equity):
                reasons.append("robustness baseline equity is invalid")
            elif any(not isfinite(equity) or equity / baseline_equity < policy.min_robustness_equity_ratio for equity in stressed_equities):
                reasons.append("robustness equity ratio is below the promotion minimum")
    return PromotionDecision("paper" if not reasons else "research", not reasons, tuple(reasons))
