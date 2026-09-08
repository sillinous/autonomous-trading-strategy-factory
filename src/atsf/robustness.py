from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    simulations: int
    seed: int
    median_return: float
    worst_return: float
    lower_percentile_return: float
    pass_rate: float


def monte_carlo_trade_bootstrap(
    trade_returns: pd.Series | list[float],
    simulations: int = 1000,
    seed: int = 0,
    lower_percentile: float = 5.0,
    min_return: float = 0.0,
) -> MonteCarloResult:
    """Bootstrap trade returns to test sequence uncertainty."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if not 0 <= lower_percentile <= 100:
        raise ValueError("lower_percentile must be between 0 and 100")
    returns = np.asarray(trade_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        raise ValueError("trade_returns must contain at least one finite value")
    if np.any(returns <= -1):
        raise ValueError("trade returns must be greater than -100%")

    rng = Random(seed)
    outcomes = np.empty(simulations, dtype=float)
    for index in range(simulations):
        sample = [returns[rng.randrange(len(returns))] for _ in returns]
        outcomes[index] = float(np.prod(1.0 + np.asarray(sample)) - 1.0)

    return MonteCarloResult(
        simulations=simulations,
        seed=seed,
        median_return=float(np.median(outcomes)),
        worst_return=float(np.min(outcomes)),
        lower_percentile_return=float(np.percentile(outcomes, lower_percentile)),
        pass_rate=float(np.mean(outcomes >= min_return)),
    )


def regime_returns(
    equity: pd.Series,
    benchmark: pd.Series,
    volatility_window: int = 20,
) -> dict[str, pd.Series]:
    """Partition strategy returns into simple market and volatility regimes."""
    if not equity.index.equals(benchmark.index):
        raise ValueError("equity and benchmark indexes must match")
    if volatility_window <= 1:
        raise ValueError("volatility_window must be greater than one")
    strategy_returns = equity.pct_change().fillna(0.0)
    benchmark_returns = benchmark.pct_change().fillna(0.0)
    volatility = benchmark_returns.rolling(
        volatility_window, min_periods=volatility_window
    ).std()
    high_vol = volatility >= volatility.median()
    rising = benchmark_returns >= 0
    return {
        "rising": strategy_returns[rising],
        "falling": strategy_returns[~rising],
        "high_volatility": strategy_returns[high_vol.fillna(False)],
        "low_volatility": strategy_returns[(~high_vol).fillna(False)],
    }


@dataclass(frozen=True)
class RegimeStabilityResult:
    score: float
    regime_returns: dict[str, float]
    covered_regimes: tuple[str, ...]


def score_regime_stability(regimes: dict[str, pd.Series]) -> RegimeStabilityResult:
    """Score consistency across non-empty regimes using the weakest regime return."""
    if not regimes:
        raise ValueError("regimes must not be empty")
    values: dict[str, float] = {}
    for name, returns in regimes.items():
        finite = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            continue
        values[name] = float((1.0 + finite).prod() - 1.0)
    if not values:
        raise ValueError("regimes contain no finite observations")
    covered = tuple(sorted(values))
    score = float(min(values.values()))
    return RegimeStabilityResult(score=score, regime_returns=values, covered_regimes=covered)
