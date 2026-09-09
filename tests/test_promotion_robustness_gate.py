import pandas as pd

from atsf.backtest import BacktestConfig, run_long_signal_backtest
from atsf.promotion import PromotionPolicy, research_to_paper
from atsf.robustness import RobustnessResult
from atsf.evaluation import WalkForwardEvaluation
from atsf.robustness import MonteCarloResult


def test_promotion_requires_execution_robustness():
    evaluation = WalkForwardEvaluation(
        passed=True,
        oos_sharpe=1.0,
        oos_drawdown=0.10,
        folds=(),
    )
    mc = MonteCarloResult(100, 1, 0.10, -0.05, 0.01, 0.99)
    decision = research_to_paper(evaluation, mc)
    assert not decision.eligible
    assert "execution robustness evidence" in decision.reasons
