"""Statistical defenses against backtest overfitting.

* :func:`probabilistic_sharpe_ratio` — probability the true Sharpe exceeds a benchmark,
  correcting for sample length, skew, and fat tails (Bailey & López de Prado, 2012).
* :func:`deflated_sharpe_ratio` — PSR against the Sharpe a researcher should *expect*
  from the best of ``n_trials`` skill-less strategies (Bailey & López de Prado, 2014).
* :func:`probability_of_backtest_overfitting` — PBO via combinatorially symmetric
  cross-validation (Bailey, Borwein, López de Prado & Zhu, 2017).

All Sharpe ratios here are per-period (not annualized); the tests are scale-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True)
class SharpeInference:
    sharpe: float              # per-period observed Sharpe
    benchmark: float           # per-period Sharpe the observation must beat
    probability: float         # P(true Sharpe > benchmark)
    observations: int
    skewness: float
    kurtosis: float            # non-excess (normal = 3)
    n_trials: int = 1


@dataclass(frozen=True)
class OverfittingReport:
    pbo: float                 # probability the IS-best config underperforms OOS median
    logits: tuple[float, ...]
    splits: int
    n_blocks: int
    degradation_slope: float   # OOS vs IS Sharpe slope across splits (<0 is bad)


def _moments(returns: np.ndarray) -> tuple[float, float, float]:
    std = returns.std(ddof=1)
    if not math.isfinite(std) or std <= 0:
        return 0.0, 0.0, 3.0
    centered = returns - returns.mean()
    sharpe = float(returns.mean() / std)
    m2 = float((centered ** 2).mean())
    skew = float((centered ** 3).mean() / m2 ** 1.5) if m2 > 0 else 0.0
    kurt = float((centered ** 4).mean() / m2 ** 2) if m2 > 0 else 3.0
    return sharpe, skew, kurt


def _clean(returns: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    return values


def probabilistic_sharpe_ratio(returns: pd.Series | np.ndarray | list[float],
                               benchmark_sharpe: float = 0.0) -> SharpeInference:
    """P(true per-period Sharpe > ``benchmark_sharpe``) given the observed returns."""
    values = _clean(returns)
    n = len(values)
    if n < 3:
        return SharpeInference(0.0, benchmark_sharpe, 0.0, n, 0.0, 3.0)
    std = float(values.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return SharpeInference(0.0, benchmark_sharpe, 0.0, n, 0.0, 3.0)
    sharpe, skew, kurt = _moments(values)
    variance = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe ** 2
    if variance <= 0 or not math.isfinite(variance):
        return SharpeInference(sharpe, benchmark_sharpe, 0.0, n, skew, kurt)
    z = (sharpe - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(variance)
    return SharpeInference(sharpe, benchmark_sharpe, float(_NORMAL.cdf(z)), n, skew, kurt)


def expected_max_sharpe(n_trials: int, trial_sharpe_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` skill-less strategies."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if trial_sharpe_variance < 0 or not math.isfinite(trial_sharpe_variance):
        raise ValueError("trial_sharpe_variance must be finite and non-negative")
    if n_trials == 1 or trial_sharpe_variance == 0:
        return 0.0
    first = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    second = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(trial_sharpe_variance) * (
        (1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second
    )


def deflated_sharpe_ratio(returns: pd.Series | np.ndarray | list[float], n_trials: int,
                          trial_sharpe_variance: float | None = None) -> SharpeInference:
    """PSR against the expected best Sharpe of ``n_trials`` null strategies.

    ``trial_sharpe_variance`` is the cross-sectional variance of per-period Sharpes of
    *all* strategies tried. When unknown, the sampling variance of a null Sharpe
    estimator, ``1 / (T - 1)``, is used, which is conservative for small trial sets.
    """
    values = _clean(returns)
    if trial_sharpe_variance is None:
        trial_sharpe_variance = 1.0 / max(len(values) - 1, 1)
    benchmark = expected_max_sharpe(n_trials, trial_sharpe_variance)
    inference = probabilistic_sharpe_ratio(values, benchmark)
    return SharpeInference(inference.sharpe, benchmark, inference.probability,
                           inference.observations, inference.skewness, inference.kurtosis,
                           n_trials)


def _sharpe_columns(matrix: np.ndarray) -> np.ndarray:
    std = matrix.std(axis=0, ddof=1)
    mean = matrix.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0, mean / std, 0.0)
    return np.nan_to_num(sharpe, nan=0.0)


def probability_of_backtest_overfitting(returns: pd.DataFrame, n_blocks: int = 10
                                        ) -> OverfittingReport:
    """PBO by CSCV over a (time x configuration) matrix of per-period returns.

    Rows are split into ``n_blocks`` contiguous blocks; every half/half combination is
    used as an in-sample/out-of-sample pair. PBO is the share of splits in which the
    in-sample winner ranks at or below the out-of-sample median.
    """
    if returns.shape[1] < 2:
        raise ValueError("PBO needs at least two strategy configurations")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even integer >= 2")
    matrix = returns.dropna(how="any").to_numpy(dtype=float)
    if len(matrix) < n_blocks * 2:
        raise ValueError("not enough observations for the requested number of blocks")
    blocks = np.array_split(np.arange(len(matrix)), n_blocks)
    logits: list[float] = []
    is_best: list[float] = []
    oos_of_best: list[float] = []
    n_configs = matrix.shape[1]
    for chosen in combinations(range(n_blocks), n_blocks // 2):
        in_rows = np.concatenate([blocks[i] for i in chosen])
        out_rows = np.concatenate([blocks[i] for i in range(n_blocks) if i not in chosen])
        in_sharpe = _sharpe_columns(matrix[in_rows])
        out_sharpe = _sharpe_columns(matrix[out_rows])
        best = int(np.argmax(in_sharpe))
        # relative rank of the IS winner out-of-sample, in (0, 1)
        rank = (np.sum(out_sharpe < out_sharpe[best]) + 0.5 * (
            np.sum(out_sharpe == out_sharpe[best]) - 1) + 1) / (n_configs + 1)
        logits.append(float(math.log(rank / (1.0 - rank))))
        is_best.append(float(in_sharpe[best]))
        oos_of_best.append(float(out_sharpe[best]))
    slope = 0.0
    if len(set(is_best)) > 1:
        slope = float(np.polyfit(is_best, oos_of_best, 1)[0])
    pbo = float(np.mean([logit <= 0 for logit in logits]))
    return OverfittingReport(pbo, tuple(logits), len(logits), n_blocks, slope)
