from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .portfolio_attribution import PortfolioAttribution


@dataclass(frozen=True)
class PortfolioHealthPolicy:
    min_total_return: float = 0.0
    max_volatility: float = 0.25
    max_single_strategy_risk_fraction: float = 0.60
    min_observations: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("min_total_return", self.min_total_return),
            ("max_volatility", self.max_volatility),
            ("max_single_strategy_risk_fraction", self.max_single_strategy_risk_fraction),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.max_volatility <= 0:
            raise ValueError("max_volatility must be positive")
        if not 0 < self.max_single_strategy_risk_fraction <= 1:
            raise ValueError("max_single_strategy_risk_fraction must be in (0, 1]")
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2")


@dataclass(frozen=True)
class PortfolioHealthResult:
    status: str
    total_return: float
    volatility: float
    max_risk_fraction: float
    breached_limits: tuple[str, ...]
    replace_strategy_ids: tuple[str, ...]
    execution_authority: bool = False

    @property
    def healthy(self) -> bool:
        return self.status == "HEALTHY"

    @property
    def decision(self) -> str:
        if self.status == "HEALTHY":
            return "CONTINUE_PORTFOLIO"
        if "concentration" in self.breached_limits:
            return "REBALANCE_OR_RESEARCH_REPLACEMENT"
        return "RESEARCH_REPLACEMENT"


def assess_portfolio_health(
    attribution: PortfolioAttribution,
    *,
    observation_count: int,
    policy: PortfolioHealthPolicy | None = None,
) -> PortfolioHealthResult:
    policy = policy or PortfolioHealthPolicy()
    if observation_count < policy.min_observations:
        raise ValueError("insufficient observations for portfolio health assessment")
    values = (attribution.total_return, attribution.volatility)
    if not all(isfinite(value) for value in values):
        raise ValueError("portfolio attribution contains non-finite values")

    total_risk = sum(abs(item.risk_contribution) for item in attribution.contributions)
    if not isfinite(total_risk):
        raise ValueError("portfolio risk attribution is non-finite")
    risk_fractions = (
        {
            item.strategy_id: abs(item.risk_contribution) / total_risk
            for item in attribution.contributions
        }
        if total_risk
        else {item.strategy_id: 0.0 for item in attribution.contributions}
    )
    max_risk_fraction = max(risk_fractions.values(), default=0.0)

    breached: list[str] = []
    if attribution.total_return < policy.min_total_return:
        breached.append("total_return")
    if attribution.volatility > policy.max_volatility:
        breached.append("volatility")
    if max_risk_fraction > policy.max_single_strategy_risk_fraction:
        breached.append("concentration")

    replacements = tuple(
        sorted(
            strategy_id
            for strategy_id, fraction in risk_fractions.items()
            if fraction > policy.max_single_strategy_risk_fraction
        )
    )
    return PortfolioHealthResult(
        status="HEALTHY" if not breached else "REVIEW_REQUIRED",
        total_return=attribution.total_return,
        volatility=attribution.volatility,
        max_risk_fraction=max_risk_fraction,
        breached_limits=tuple(breached),
        replace_strategy_ids=replacements,
        execution_authority=False,
    )
