import numpy as np
import pandas as pd

from atsf.backtest import BacktestResult
from atsf.evaluation import WalkForwardEvaluation
from atsf.perturbation import PerturbationResult
from atsf.promotion import PromotionPolicy, research_to_paper
from atsf.robustness import MonteCarloResult, RegimeStabilityResult, RobustnessResult


def oos_returns(daily_mean: float = 0.0012, days: int = 750, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(daily_mean, 0.01, days),
                     index=pd.bdate_range("2022-01-03", periods=days))


def evaluation(passed: bool = True, trades: int = 30, daily_mean: float = 0.0012) -> WalkForwardEvaluation:
    return WalkForwardEvaluation(
        windows=(), passed=passed, oos_return=0.20, oos_sharpe=1.2, oos_drawdown=0.10,
        oos_returns=oos_returns(daily_mean), oos_trade_returns=(0.01,) * trades,
    )


def monte_carlo(pass_rate: float = 0.98, lower: float = 0.02) -> MonteCarloResult:
    return MonteCarloResult(100, 7, 0.20, -0.05, lower, pass_rate)


def perturbation(pass_rate: float = 0.90) -> PerturbationResult:
    return PerturbationResult(20, 7, pass_rate, 0.1, 1.0, ())


def regime(score: float = 0.01) -> RegimeStabilityResult:
    return RegimeStabilityResult(score, {"rising": 0.01, "falling": score}, ("falling", "rising"))


def robustness(passed: bool = True) -> RobustnessResult:
    index = pd.date_range("2026-01-01", periods=2, freq="D")
    equity = pd.Series([100_000.0, 105_000.0], index=index)
    result = BacktestResult(
        equity=equity,
        returns=pd.Series([0.0, 0.05], index=index),
        trades=pd.DataFrame(columns=["timestamp", "action"]),
        total_return=0.05,
        max_drawdown=0.0,
        trade_returns=(0.05,),
    )
    return RobustnessResult(
        "candidate", result, (("baseline", result), ("cost_2x", result)), passed,
        () if passed else ("stressed execution failed",),
    )


def test_promotion_requires_all_hard_gates():
    decision = research_to_paper(evaluation(), monte_carlo(), perturbation(), regime(), robustness())
    assert decision.eligible
    assert decision.stage == "paper"


def test_missing_robustness_evidence_blocks_paper_promotion():
    decision = research_to_paper(evaluation(), monte_carlo())
    assert not decision.eligible
    assert decision.stage == "research"
    assert any("parameter perturbation evidence" in reason for reason in decision.reasons)
    assert any("regime stability evidence" in reason for reason in decision.reasons)


def test_failed_oos_blocks_paper_promotion():
    decision = research_to_paper(evaluation(False), monte_carlo(), perturbation(), regime(), robustness())
    assert not decision.eligible
    assert decision.stage == "research"
    assert decision.reasons


def test_weak_monte_carlo_blocks_paper_promotion():
    policy = PromotionPolicy(min_monte_carlo_pass_rate=0.99)
    decision = research_to_paper(evaluation(), monte_carlo(0.98), perturbation(), regime(), robustness(), policy)
    assert not decision.eligible
    assert any("Monte Carlo pass rate" in reason for reason in decision.reasons)


def test_weak_parameter_stability_blocks_paper_promotion():
    decision = research_to_paper(evaluation(), monte_carlo(), perturbation(0.5), regime(), robustness())
    assert not decision.eligible
    assert "parameter perturbation" in " ".join(decision.reasons)


def test_weak_regime_stability_blocks_paper_promotion():
    decision = research_to_paper(evaluation(), monte_carlo(), perturbation(), regime(-0.2), robustness())
    assert not decision.eligible
    assert "regime stability" in " ".join(decision.reasons)


def test_too_few_oos_trades_blocks_promotion():
    decision = research_to_paper(evaluation(trades=5), monte_carlo(), perturbation(), regime(), robustness())
    assert not decision.eligible
    assert any("trade count 5" in reason for reason in decision.reasons)


def test_sharpe_that_does_not_survive_deflation_blocks_promotion():
    """A respectable single-trial Sharpe fails once 500 variants were searched."""
    weak = evaluation(daily_mean=0.0006)
    alone = research_to_paper(weak, monte_carlo(), perturbation(), regime(), robustness())
    searched = research_to_paper(weak, monte_carlo(), perturbation(), regime(), robustness(),
                                 n_trials=500, trial_sharpe_variance=0.002)
    assert alone.eligible
    assert not searched.eligible
    assert any("deflated Sharpe" in reason for reason in searched.reasons)


def test_missing_oos_returns_fail_closed():
    bare = WalkForwardEvaluation(windows=(), passed=True, oos_return=0.2, oos_sharpe=1.2,
                                 oos_drawdown=0.1, oos_trade_returns=(0.01,) * 30)
    decision = research_to_paper(bare, monte_carlo(), perturbation(), regime(), robustness())
    assert not decision.eligible
