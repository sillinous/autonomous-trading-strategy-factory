import pandas as pd
import pytest

from atsf.lifecycle import StrategyLifecycleStage
from atsf.paper_qualification import (
    PaperQualificationDecision,
    PaperQualificationPolicy,
    qualify_paper_strategy,
)


def equity(values):
    return pd.Series(values, index=pd.RangeIndex(len(values)))


def test_healthy_paper_strategy_becomes_eligible_for_live_review():
    result = qualify_paper_strategy(
        "strategy-1",
        StrategyLifecycleStage.PAPER,
        equity([1.0 + 0.002 * i for i in range(60)]),
    )
    assert result.decision is PaperQualificationDecision.ELIGIBLE_FOR_LIVE_REVIEW
    assert result.execution_authority is False
    assert result.reasons == ()


def test_qualification_rejects_non_paper_stage():
    result = qualify_paper_strategy(
        "strategy-1",
        StrategyLifecycleStage.PROMOTED,
        equity([1.0 + 0.002 * i for i in range(60)]),
    )
    assert result.decision is PaperQualificationDecision.REJECT
    assert any("PAPER lifecycle" in reason for reason in result.reasons)


def test_qualification_rejects_insufficient_observations():
    result = qualify_paper_strategy(
        "strategy-1",
        StrategyLifecycleStage.PAPER,
        equity([1.0] * 10),
    )
    assert result.decision is PaperQualificationDecision.REJECT
    assert any("observations" in reason for reason in result.reasons)


def test_qualification_rejects_drawdown_breach():
    values = [1.0 + 0.001 * i for i in range(30)] + [0.80] + [0.81 + 0.001 * i for i in range(29)]
    result = qualify_paper_strategy(
        "strategy-1",
        StrategyLifecycleStage.PAPER,
        equity(values),
    )
    assert result.decision is PaperQualificationDecision.REJECT
    assert any("drawdown" in reason for reason in result.reasons)


def test_qualification_rejects_nonfinite_equity():
    values = [1.0] * 59 + [float("nan")]
    result = qualify_paper_strategy(
        "strategy-1",
        StrategyLifecycleStage.PAPER,
        equity(values),
    )
    assert result.decision is PaperQualificationDecision.REJECT
    assert any("evidence invalid" in reason for reason in result.reasons)


def test_policy_validation_is_fail_closed():
    with pytest.raises(ValueError):
        PaperQualificationPolicy(min_observations=1)
    with pytest.raises(ValueError):
        PaperQualificationPolicy(max_drawdown=1.0)
    with pytest.raises(ValueError):
        PaperQualificationPolicy(max_volatility=-0.1)
