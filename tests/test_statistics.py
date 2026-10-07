from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from atsf.selection import SelectionPolicy, population_overfitting, select_population
from atsf.statistics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)


def noise(seed: int, n: int = 1000, columns: int = 40, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(drift, 0.01, (n, columns)),
                        index=pd.bdate_range("2020-01-01", periods=n))


def test_psr_is_high_for_real_skill_and_near_half_for_none():
    rng = np.random.default_rng(0)
    skilled = rng.normal(0.002, 0.01, 1000)
    assert probabilistic_sharpe_ratio(skilled).probability > 0.999
    centered = rng.normal(0, 0.01, 1000)
    centered -= centered.mean()
    assert probabilistic_sharpe_ratio(centered).probability == pytest.approx(0.5, abs=0.01)


def test_psr_fails_closed_on_degenerate_input():
    assert probabilistic_sharpe_ratio([0.01, 0.01, 0.01]).probability == 0.0
    assert probabilistic_sharpe_ratio([0.01]).probability == 0.0


def test_expected_max_sharpe_grows_with_trials():
    values = [expected_max_sharpe(n, 0.001) for n in (1, 10, 100, 1000)]
    assert values[0] == 0.0 and values == sorted(values)


def test_best_of_many_noise_strategies_is_exposed_by_deflation():
    matrix = noise(1, columns=300)
    sharpe = matrix.mean() / matrix.std()
    best = matrix[sharpe.idxmax()]
    assert probabilistic_sharpe_ratio(best).probability > 0.95  # looks great alone
    deflated = deflated_sharpe_ratio(best, 300, float(sharpe.var()))
    assert deflated.probability < 0.95                           # not after deflation
    assert deflated.n_trials == 300


def test_pbo_is_calibrated_near_half_on_pure_noise():
    values = [probability_of_backtest_overfitting(noise(seed, columns=20), 8).pbo
              for seed in range(25)]
    assert 0.35 < float(np.mean(values)) < 0.65


def test_pbo_is_low_when_one_configuration_has_real_edge():
    matrix = noise(3, columns=20)
    matrix[0] += 0.002
    assert probability_of_backtest_overfitting(matrix, 8).pbo < 0.05


def test_pbo_validates_inputs():
    with pytest.raises(ValueError, match="two strategy"):
        probability_of_backtest_overfitting(noise(0, columns=1), 8)
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(noise(0), 7)
    with pytest.raises(ValueError, match="observations"):
        probability_of_backtest_overfitting(noise(0, n=10), 8)


def _evaluation(candidate_id, returns):
    baseline = pd.Series([1.0, 100.0])
    return SimpleNamespace(
        candidate_id=candidate_id, fitness=SimpleNamespace(score=1.0),
        backtest=SimpleNamespace(max_drawdown=-0.1), monte_carlo=SimpleNamespace(pass_rate=1.0),
        regime=SimpleNamespace(score=0.0),
        robustness=SimpleNamespace(baseline=SimpleNamespace(equity=baseline),
                                   scenarios=(("s", SimpleNamespace(equity=baseline)),)),
        walk_forward=SimpleNamespace(oos_returns=returns),
        promotion=SimpleNamespace(eligible=True), validation_passed=True)


def _candidate(candidate_id, value):
    return SimpleNamespace(strategy_id=candidate_id,
                           strategy=SimpleNamespace(model_dump=lambda mode=None: {"v": value}))


def test_selection_rejects_an_overfit_population():
    matrix = noise(5, columns=12)
    evaluations = [_evaluation(f"c{i}", matrix[i]) for i in range(12)]
    candidates = [_candidate(f"c{i}", i) for i in range(12)]
    report = population_overfitting(evaluations, 8)
    assert report is not None
    policy = SelectionPolicy(population_size=1, max_pbo=min(report.pbo, 0.99) - 0.01 or 0.01,
                             min_deflated_sharpe=0.0, pbo_blocks=8)
    with pytest.raises(ValueError, match="overfit"):
        select_population(candidates, evaluations, policy)


def test_selection_deflates_for_prior_generations():
    rng = np.random.default_rng(9)
    returns = pd.Series(rng.normal(0.0009, 0.01, 750), index=pd.bdate_range("2022-01-03", periods=750))
    evaluations = [_evaluation("a", returns)]
    candidates = [_candidate("a", 1)]
    policy = SelectionPolicy(population_size=1)
    assert select_population(candidates, evaluations, policy)  # one trial: passes
    with pytest.raises(ValueError, match="no candidates"):
        select_population(candidates, evaluations, policy, prior_trials=10_000)


def test_selection_compares_drawdown_magnitude():
    returns = pd.Series(np.random.default_rng(2).normal(0.002, 0.01, 750))
    deep = _evaluation("a", returns)
    deep.backtest = SimpleNamespace(max_drawdown=-0.40)
    with pytest.raises(ValueError, match="no candidates"):
        select_population([_candidate("a", 1)], [deep],
                          SelectionPolicy(population_size=1, max_drawdown=0.25))
