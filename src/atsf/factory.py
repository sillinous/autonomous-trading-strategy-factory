"""End-to-end research factory: data in, ranked and gated strategies out.

``run_factory`` is the single front door used by the CLI and API. It generates a
candidate population, evaluates every candidate through the same deterministic gates
used everywhere else (walk-forward OOS, Monte Carlo, parameter perturbation, regime,
execution robustness), then applies population-level multiple-testing corrections:
each candidate's promotion decision is re-made with the true trial count and the
population's cross-sectional Sharpe variance, and the whole population is checked
for backtest overfitting (PBO). It never places orders.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .execution import PERIODS_PER_YEAR
from .generator import ARCHETYPES, StrategyGenerator
from .orchestrator import CandidateEvaluation, evaluate_candidate
from .population import seed_population
from .promotion import PromotionPolicy, research_to_paper
from .research_queue import ResearchReason, ResearchRequest
from .selection import population_overfitting, population_trial_statistics
from .statistics import deflated_sharpe_ratio
from .strategy import StrategySpec


@dataclass(frozen=True)
class FactoryRow:
    strategy_id: str
    name: str
    family: str
    oos_return: float
    oos_sharpe: float
    oos_drawdown: float
    oos_trades: int
    deflated_sharpe: float
    monte_carlo_pass_rate: float
    perturbation_pass_rate: float
    train_passed: bool | None
    promoted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FactoryReport:
    symbol: str
    dataset_version: str
    start: str
    end: str
    bars: int
    n_trials: int
    pbo: float | None
    population_overfit: bool
    strict_pbo: bool
    benchmark_sharpe: float | None
    rows: tuple[FactoryRow, ...]
    strategies: dict[str, dict]

    @property
    def promoted(self) -> tuple[FactoryRow, ...]:
        return tuple(row for row in self.rows if row.promoted)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["rows"] = [asdict(row) for row in self.rows]
        return payload


def dataset_fingerprint(data: pd.DataFrame) -> str:
    """Content hash of the exact bars evaluated."""
    canonical = data.sort_index().to_csv(float_format="%.10g").encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def default_strategies(symbol: str) -> list[StrategySpec]:
    request = ResearchRequest(f"factory-{symbol.lower()}", None, ResearchReason.DIVERSIFICATION, 1)
    return [candidate.strategy for candidate in StrategyGenerator().generate(request, [symbol])]


MA_VARIANTS = ("sma_fast", "ema_fast", "sma_slow")


def _family(strategy: StrategySpec) -> str:
    if strategy.metadata.get("origin") == "claude":
        return f"claude:{strategy.name}"[:19]
    for name in (*ARCHETYPES, *MA_VARIANTS):
        if strategy.name.endswith(name):
            return name if name in ARCHETYPES else f"ma_cross_{name}"
    return "custom"


def _annualized_sharpe(returns: pd.Series | None, timeframe: str) -> float:
    if returns is None or len(returns) < 3 or float(returns.std(ddof=1)) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR[timeframe]))


def run_factory(
    data: pd.DataFrame,
    symbol: str,
    *,
    strategies: list[StrategySpec] | None = None,
    seed: int = 0,
    perturbation_samples: int = 12,
    promotion_policy: PromotionPolicy | None = None,
    max_pbo: float = 0.5,
    pbo_blocks: int = 10,
    strict_pbo: bool = False,
    require_benchmark: bool = True,
) -> FactoryReport:
    """Generate, evaluate, deflate, and gate a strategy population on ``data``.

    Multiple testing is handled by the deflated Sharpe ratio, which asks whether each
    strategy's OOS Sharpe is real given how many were tried. PBO answers a different
    question — whether the in-sample *ranking* survives out of sample — so it vetoes
    rank-based survivor selection (see :func:`atsf.selection.select_population`) but,
    for absolute-gate promotion to paper, it is reported as a warning unless
    ``strict_pbo`` is set.

    With ``require_benchmark`` (default), a strategy must also match or beat the
    buy-and-hold Sharpe over the same OOS period: a long-only strategy that is worse
    risk-adjusted than simply holding the asset has not earned paper capital.
    """
    if len(data) < 500:
        raise ValueError("the factory needs at least 500 bars for meaningful OOS evidence")
    strategies = strategies or default_strategies(symbol)
    candidates = seed_population(strategies)
    version = dataset_fingerprint(data)
    evaluations: list[CandidateEvaluation] = [
        evaluate_candidate(candidate, data, symbol, version, seed=seed + offset,
                           promotion_policy=promotion_policy,
                           perturbation_samples=perturbation_samples)
        for offset, candidate in enumerate(candidates)
    ]
    n_trials, variance = population_trial_statistics(evaluations)
    overfitting = population_overfitting(evaluations, pbo_blocks)
    overfit = overfitting is not None and overfitting.pbo > max_pbo
    veto = overfit and strict_pbo

    benchmark_sharpe = None
    oos_index = next((e.walk_forward.oos_returns.index for e in evaluations
                      if e.walk_forward.oos_returns is not None), None)
    if oos_index is not None:
        hold = data["close"].pct_change().reindex(oos_index).dropna()
        benchmark_sharpe = _annualized_sharpe(hold, strategies[0].timeframe)

    rows: list[FactoryRow] = []
    for candidate, evaluation in zip(candidates, evaluations):
        decision = research_to_paper(
            evaluation.walk_forward, evaluation.monte_carlo, evaluation.perturbation,
            evaluation.regime, evaluation.robustness, promotion_policy,
            n_trials=n_trials, trial_sharpe_variance=variance,
        )
        reasons = list(decision.reasons)
        if veto:
            reasons.append(f"population PBO {overfitting.pbo:.2f} exceeds {max_pbo:.2f}")
        returns = evaluation.walk_forward.oos_returns
        oos_sharpe = _annualized_sharpe(returns, candidate.strategy.timeframe)
        below_benchmark = (require_benchmark and benchmark_sharpe is not None
                           and oos_sharpe < benchmark_sharpe)
        if below_benchmark:
            reasons.append(f"OOS Sharpe {oos_sharpe:.2f} is below buy-and-hold {benchmark_sharpe:.2f}")
        inference = deflated_sharpe_ratio(() if returns is None else returns, n_trials, variance)
        rows.append(FactoryRow(
            strategy_id=candidate.strategy_id,
            name=candidate.strategy.name,
            family=_family(candidate.strategy),
            oos_return=float(evaluation.walk_forward.oos_return),
            oos_sharpe=oos_sharpe,
            oos_drawdown=float(evaluation.walk_forward.oos_drawdown),
            oos_trades=len(evaluation.walk_forward.oos_trade_returns),
            deflated_sharpe=float(inference.probability),
            monte_carlo_pass_rate=float(evaluation.monte_carlo.pass_rate),
            perturbation_pass_rate=float(evaluation.perturbation.pass_rate),
            train_passed=evaluation.walk_forward.train_passed,
            promoted=decision.eligible and not veto and not below_benchmark,
            reasons=tuple(reasons),
        ))
    rows.sort(key=lambda row: (row.promoted, row.deflated_sharpe, row.oos_sharpe), reverse=True)
    return FactoryReport(
        symbol=symbol,
        dataset_version=version,
        start=data.index[0].isoformat(),
        end=data.index[-1].isoformat(),
        bars=len(data),
        n_trials=n_trials,
        pbo=None if overfitting is None else float(overfitting.pbo),
        population_overfit=overfit,
        strict_pbo=strict_pbo,
        benchmark_sharpe=benchmark_sharpe,
        rows=tuple(rows),
        strategies={c.strategy_id: c.strategy.model_dump(mode="json") for c in candidates},
    )


def summarize_backtest(equity: pd.Series, timeframe: str = "1d") -> dict[str, float]:
    returns = equity.pct_change().dropna()
    years = max(len(returns) / PERIODS_PER_YEAR[timeframe], 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    return {
        "total_return": total,
        "cagr": float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 else -1.0,
        "sharpe": _annualized_sharpe(returns, timeframe),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "exposure": float(np.mean(returns != 0)) if len(returns) else 0.0,
    }
