from dataclasses import replace

import pytest

from atsf.live_authorization import (
    LiveAuthorizationDecision,
    LiveAuthorizationPolicy,
    authorize_live_strategy,
)
from atsf.paper_qualification import PaperQualification, PaperQualificationDecision


def qualified() -> PaperQualification:
    return PaperQualification(
        strategy_id="strategy-1",
        decision=PaperQualificationDecision.ELIGIBLE_FOR_LIVE_REVIEW,
        execution_authority=False,
        report=object(),
        reasons=(),
    )


def test_human_approval_is_required() -> None:
    result = authorize_live_strategy(
        qualified(), human_approved=False, requested_capital_fraction=0.05
    )
    assert result.decision is LiveAuthorizationDecision.REJECT
    assert result.execution_authority is False


def test_approved_strategy_gets_bounded_authority() -> None:
    result = authorize_live_strategy(
        qualified(), human_approved=True, requested_capital_fraction=0.05
    )
    assert result.decision is LiveAuthorizationDecision.AUTHORIZED
    assert result.execution_authority is True
    assert result.capital_fraction == 0.05


def test_capital_limit_fails_closed() -> None:
    result = authorize_live_strategy(
        qualified(), human_approved=True, requested_capital_fraction=0.11
    )
    assert result.decision is LiveAuthorizationDecision.REJECT
    assert result.execution_authority is False


def test_unqualified_strategy_cannot_be_authorized() -> None:
    rejected = replace(qualified(), decision=PaperQualificationDecision.REJECT)
    result = authorize_live_strategy(
        rejected, human_approved=True, requested_capital_fraction=0.05
    )
    assert result.decision is LiveAuthorizationDecision.REJECT
    assert result.execution_authority is False


def test_qualification_cannot_self_authorize() -> None:
    compromised = replace(qualified(), execution_authority=True)
    result = authorize_live_strategy(
        compromised, human_approved=True, requested_capital_fraction=0.05
    )
    assert result.decision is LiveAuthorizationDecision.REJECT
    assert result.execution_authority is False


def test_invalid_policy_is_rejected() -> None:
    with pytest.raises(ValueError):
        LiveAuthorizationPolicy(max_capital_fraction=0.0)
