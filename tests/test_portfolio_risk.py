import numpy as np
import pandas as pd

from atsf.portfolio_risk import PortfolioRiskPolicy, build_risk_aware_portfolio
from atsf.ranking import RankedCandidate


def _ranked() -> tuple[RankedCandidate, ...]:
    return tuple(
        RankedCandidate(
            candidate_id=sid,
            fitness_score=1.0 - i * 0.1,
            robustness_score=1.0,
            diversity_score=1.0,
            final_score=1.0 - i * 0.1,
        )
        for i, sid in enumerate(("a", "b", "c"))
    )


def test_risk_admission_removes_lowest_ranked_strategy_when_needed():
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 0.01, 80)
    independent = rng.normal(0.0, 0.01, 80)
    returns = pd.DataFrame({"a": base, "b": independent, "c": rng.normal(0.0, 0.08, 80)})

    result = build_risk_aware_portfolio(
        _ranked(), returns, policy=PortfolioRiskPolicy(max_portfolio_volatility=0.0075)
    )

    assert result.admission_passed
    assert result.admitted_strategy_ids == ("a", "b")
    assert result.risk_rejected_strategy_ids == ("c",)
    assert result.portfolio_volatility <= 0.0075


def test_risk_admission_is_deterministic():
    returns = pd.DataFrame({"a": [0.01, -0.01] * 40, "b": [0.01, -0.01] * 40, "c": [0.04, -0.04] * 40})
    policy = PortfolioRiskPolicy(max_portfolio_volatility=0.03)
    assert build_risk_aware_portfolio(_ranked(), returns, policy=policy) == build_risk_aware_portfolio(
        _ranked(), returns, policy=policy
    )


def test_risk_rejects_all_when_even_single_strategy_is_too_volatile():
    returns = pd.DataFrame({"a": [0.2, -0.2] * 40, "b": [0.3, -0.3] * 40, "c": [0.4, -0.4] * 40})
    result = build_risk_aware_portfolio(
        _ranked(), returns, policy=PortfolioRiskPolicy(max_portfolio_volatility=0.01)
    )

    assert not result.admission_passed
    assert result.admitted_strategy_ids == ()
    assert result.risk_rejected_strategy_ids == ("a",)
    assert result.allocation.weights == {}
    assert result.portfolio_volatility > 0.01
