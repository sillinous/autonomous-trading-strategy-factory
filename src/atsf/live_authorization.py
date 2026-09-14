from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .paper_qualification import PaperQualification, PaperQualificationDecision


class LiveAuthorizationDecision(str, Enum):
    AUTHORIZED = "authorized"
    REJECT = "reject"


@dataclass(frozen=True)
class LiveAuthorizationPolicy:
    """Independent controls for converting review eligibility into authority."""

    require_explicit_human_approval: bool = True
    max_capital_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.max_capital_fraction <= 1.0:
            raise ValueError("max_capital_fraction must be greater than 0 and at most 1")


@dataclass(frozen=True)
class LiveAuthorization:
    strategy_id: str
    decision: LiveAuthorizationDecision
    execution_authority: bool
    capital_fraction: float
    reasons: tuple[str, ...]


def authorize_live_strategy(
    qualification: PaperQualification,
    *,
    human_approved: bool,
    requested_capital_fraction: float,
    policy: LiveAuthorizationPolicy | None = None,
) -> LiveAuthorization:
    """Apply the final explicit human authorization boundary.

    This function is the only policy layer that can return execution_authority=True.
    Paper qualification, research, evolution, and health systems cannot grant it.
    """
    policy = policy or LiveAuthorizationPolicy()
    reasons: list[str] = []
    if not qualification.strategy_id.strip():
        reasons.append("strategy_id is required")
    if qualification.decision is not PaperQualificationDecision.ELIGIBLE_FOR_LIVE_REVIEW:
        reasons.append("strategy is not eligible for live review")
    if qualification.execution_authority:
        reasons.append("qualification layer must not possess execution authority")
    if policy.require_explicit_human_approval and not human_approved:
        reasons.append("explicit human approval is required")
    if not isfinite(requested_capital_fraction) or requested_capital_fraction <= 0.0:
        reasons.append("requested capital fraction must be finite and positive")
    elif requested_capital_fraction > policy.max_capital_fraction:
        reasons.append("requested capital fraction exceeds authorization limit")

    authorized = not reasons
    return LiveAuthorization(
        strategy_id=qualification.strategy_id,
        decision=(
            LiveAuthorizationDecision.AUTHORIZED
            if authorized
            else LiveAuthorizationDecision.REJECT
        ),
        execution_authority=authorized,
        capital_fraction=requested_capital_fraction if authorized else 0.0,
        reasons=tuple(dict.fromkeys(reasons)),
    )
