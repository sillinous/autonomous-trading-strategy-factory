from atsf.evaluation import WalkForwardEvaluation
from atsf.perturbation import PerturbationResult
from atsf.promotion import PromotionPolicy, research_to_paper
from atsf.robustness import MonteCarloResult, RegimeStabilityResult


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


def perturbation(pass_rate: float = 0.90) -> PerturbationResult:
    return PerturbationResult(
        samples=20,
        seed=7,
        pass_rate=pass_rate,
        worst_score=0.1,
        median_score=1.0,
        strategy_ids=(),
    )


def regime(score: float = 0.01) -> RegimeStabilityResult:
    return RegimeStabilityResult(
        score=score,
        regime_returns={"rising": 0.01, "falling": score},
        covered_regimes=("falling", "rising"),
    )


def test_promotion_requires_all_hard_gates():
    decision = research_to_paper(evaluation(), monte_carlo(), perturbation(), regime())
    assert decision.eligible
    assert decision.stage == "paper"


def test_missing_robustness_evidence_blocks_paper_promotion():
    decision = research_to_paper(evaluation(), monte_carlo())
    assert not decision.eligible
    assert decision.stage == "research"
    assert any("parameter perturbation evidence" in reason for reason in decision.reasons)
    assert any("regime stability evidence" in reason for reason in decision.reasons)


def test_failed_oos_blocks_paper_promotion():
    decision = research_to_paper(evaluation(False), monte_carlo(), perturbation(), regime())
    assert not decision.eligible
    assert decision.stage == "research"
    assert decision.reasons


def test_weak_monte_carlo_blocks_paper_promotion():
    policy = PromotionPolicy(min_monte_carlo_pass_rate=0.99)
    decision = research_to_paper(
        evaluation(), monte_carlo(0.98), perturbation(), regime(), policy
    )
    assert not decision.eligible
    assert "Monte Carlo pass rate" in decision.reasons[0]


def test_weak_parameter_stability_blocks_paper_promotion():
    decision = research_to_paper(evaluation(), monte_carlo(), perturbation(0.5), regime())
    assert not decision.eligible
    assert "parameter perturbation" in " ".join(decision.reasons)


def test_weak_regime_stability_blocks_paper_promotion():
    decision = research_to_paper(evaluation(), monte_carlo(), perturbation(), regime(-0.2))
    assert not decision.eligible
    assert "regime stability" in " ".join(decision.reasons)
