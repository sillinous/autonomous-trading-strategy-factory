from __future__ import annotations

from dataclasses import dataclass

from .metrics import max_drawdown, sharpe_ratio


@dataclass(frozen=True)
class ValidationPolicy:
    min_sharpe: float = 0.5
    max_drawdown: float = 0.25


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    sharpe: float
    drawdown: float
    reasons: tuple[str, ...]


def validate_equity(equity, policy: ValidationPolicy | None = None) -> ValidationResult:
    policy = policy or ValidationPolicy()
    sharpe = sharpe_ratio(equity)
    drawdown = abs(max_drawdown(equity))
    reasons: list[str] = []
    if sharpe < policy.min_sharpe:
        reasons.append(f"sharpe {sharpe:.3f} below minimum {policy.min_sharpe:.3f}")
    if drawdown > policy.max_drawdown:
        reasons.append(f"drawdown {drawdown:.3f} above maximum {policy.max_drawdown:.3f}")
    return ValidationResult(not reasons, sharpe, drawdown, tuple(reasons))
