from atsf.evaluation import WalkForwardEvaluation
from atsf.promotion import PromotionPolicy, research_to_paper
from atsf.robustness import MonteCarloResult


def evaluation(passed: bool = True) -> WalkForwardEvaluation:
    return WalkForwardEvaluation(
        windows=(),
        passed=passed,
        oos_return=0.20,
        oos_sharpe=1.2,
        oos_drawdown=0.10,
    )


def monte_carlo(pass_rate: float = 0.98, lower: float = 0.02) -> MonteCarloResult:
    return MonteCarloResult(
        simulations=100,
        seed=7,
        median_return=0.20,
        worst_return=-0.05,
        lower_percentile_return=lower,
        pass_rate=pass_rate,
    )


def test_promotion_requires_all_hard_gates():
    decision = research_to_paper(evaluation(), monte_carlo())
    assert decision.eligible
    assert decision.stage == "paper"


def test_failed_oos_blocks_paper_promotion():
    decision = research_to_paper(evaluation(False), monte_carlo())
    assert not decision.eligible
    assert decision.stage == "research"
    assert decision.reasons


def test_weak_monte_carlo_blocks_paper_promotion():
    policy = PromotionPolicy(min_monte_carlo_pass_rate=0.99)
    decision = research_to_paper(evaluation(), monte_carlo(0.98), policy)
    assert not decision.eligible
    assert "Monte Carlo pass rate" in decision.reasons[0]
