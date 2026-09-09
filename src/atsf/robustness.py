from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_long_signal_backtest
from .generator import StrategyCandidate
from .signals import strategy_signals


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
        simulations, seed, float(np.median(outcomes)), float(np.min(outcomes)),
        float(np.percentile(outcomes, lower_percentile)),
        float(np.mean(outcomes >= min_return)),
    )


@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    config: BacktestConfig


@dataclass(frozen=True)
class RobustnessResult:
    candidate_id: str
    baseline: BacktestResult
    scenarios: tuple[tuple[str, BacktestResult], ...]
    passed: bool
    reasons: tuple[str, ...]


def default_scenarios(config: BacktestConfig | None = None) -> tuple[RobustnessScenario, ...]:
    base = config or BacktestConfig()
    return (
        RobustnessScenario("baseline", base),
        RobustnessScenario("cost_2x", replace(base, commission_bps=base.commission_bps * 2)),
        RobustnessScenario("slippage_2x", replace(base, slippage_bps=base.slippage_bps * 2)),
        RobustnessScenario(
            "cost_slippage_2x",
            replace(base, commission_bps=base.commission_bps * 2, slippage_bps=base.slippage_bps * 2),
        ),
    )


def test_robustness(
    data: pd.DataFrame,
    candidate: StrategyCandidate,
    scenarios: tuple[RobustnessScenario, ...] | None = None,
    min_equity_ratio: float = 0.90,
) -> RobustnessResult:
    """Stress transaction costs and slippage without changing strategy logic."""
    if min_equity_ratio <= 0 or min_equity_ratio > 1:
        raise ValueError("min_equity_ratio must be in (0, 1]")
    selected = scenarios or default_scenarios()
    if not selected:
        raise ValueError("scenarios cannot be empty")
    entry, _ = strategy_signals(data, candidate.strategy)
    results = tuple(
        (scenario.name, run_long_signal_backtest(data, entry, candidate.strategy, scenario.config))
        for scenario in selected
    )
    baseline = next((result for name, result in results if name == "baseline"), results[0][1])
    baseline_equity = float(baseline.equity.iloc[-1])
    reasons = []
    if baseline_equity <= 0:
        reasons.append("baseline equity is non-positive")
    for name, result in results:
        if float(result.equity.iloc[-1]) < baseline_equity * min_equity_ratio:
            reasons.append(f"{name} equity fell below robustness threshold")
    return RobustnessResult(candidate.candidate_id, baseline, results, not reasons, tuple(reasons))


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
    volatility = benchmark_returns.rolling(volatility_window, min_periods=volatility_window).std()
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
    """Score consistency across non-empty regimes using the weakest observation."""
    if not regimes:
        raise ValueError("regimes must not be empty")
    values: dict[str, float] = {}
    for name, returns in regimes.items():
        finite = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            continue
        values[name] = float(finite.mean())
    if not values:
        raise ValueError("regimes contain no finite observations")
    covered = tuple(sorted(values))
    return RegimeStabilityResult(float(min(values.values())), values, covered)
