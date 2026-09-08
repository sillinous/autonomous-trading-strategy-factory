from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FitnessPolicy:
    min_sharpe: float = 0.5
    max_drawdown: float = 0.25
    sharpe_weight: float = 0.7
    drawdown_weight: float = 0.3


@dataclass(frozen=True)
class FitnessResult:
    score: float
    eligible: bool
    reasons: tuple[str, ...]


def score_strategy(sharpe: float, drawdown: float, policy: FitnessPolicy | None = None) -> FitnessResult:
    """Score a candidate while applying hard rejection gates first."""
    policy = policy or FitnessPolicy()
    if policy.sharpe_weight < 0 or policy.drawdown_weight < 0:
        raise ValueError("fitness weights must be non-negative")
    weight_total = policy.sharpe_weight + policy.drawdown_weight
    if weight_total <= 0:
        raise ValueError("at least one fitness weight must be positive")

    reasons: list[str] = []
    if sharpe < policy.min_sharpe:
        reasons.append(f"sharpe {sharpe:.3f} below minimum {policy.min_sharpe:.3f}")
    if drawdown > policy.max_drawdown:
        reasons.append(f"drawdown {drawdown:.3f} above maximum {policy.max_drawdown:.3f}")

    normalized_sharpe = max(0.0, sharpe) / max(policy.min_sharpe, 1e-12)
    normalized_drawdown = max(0.0, 1.0 - drawdown / max(policy.max_drawdown, 1e-12))
    score = (
        policy.sharpe_weight * normalized_sharpe
        + policy.drawdown_weight * normalized_drawdown
    ) / weight_total
    return FitnessResult(score=score, eligible=not reasons, reasons=tuple(reasons))
