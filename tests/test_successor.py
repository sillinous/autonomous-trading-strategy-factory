from types import SimpleNamespace

import pytest

from atsf.lineage import LineageRecord
from atsf.population import Candidate
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)
from atsf.successor import admit_successor, successor_lineage


def make_strategy() -> StrategySpec:
    return StrategySpec(
        name="successor",
        universe=["TEST"],
        indicators=[Indicator(name="sma", source="close", period=5)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )


def make_candidate() -> Candidate:
    strategy = make_strategy()
    return Candidate(
        strategy=strategy,
        strategy_id="child-1",
        lineage=LineageRecord("child-1", 1, ("parent-1",), "mutation", {"period": 10}),
    )


def evaluation(candidate_id="child-1", eligible=True):
    return SimpleNamespace(
        candidate_id=candidate_id,
        promotion=SimpleNamespace(eligible=eligible, stage="paper" if eligible else "research", reasons=() if eligible else ("failed",)),
        fitness=SimpleNamespace(score=1.0),
        walk_forward=SimpleNamespace(oos_sharpe=1.0, oos_drawdown=0.1),
    )


def test_admit_successor_requires_matching_identity_and_parentage():
    candidate = make_candidate()
    admission = admit_successor(candidate, evaluation(), generation=2)
    assert admission.admitted
    assert admission.parent_ids == ("parent-1",)
    assert admission.generation == 2


def test_rejected_evaluation_is_not_admitted():
    admission = admit_successor(make_candidate(), evaluation(eligible=False), generation=2)
    assert not admission.admitted
    assert admission.reasons == ("failed",)


def test_successor_lineage_records_admission():
    candidate = make_candidate()
    admission = admit_successor(candidate, evaluation(), generation=2)
    lineage = successor_lineage(candidate, admission)
    assert lineage.generation == 2
    assert lineage.parent_ids == ("parent-1",)
    assert lineage.parameters["admitted"] is True
    assert lineage.parameters["admission_stage"] == "paper"


def test_admission_fails_closed_for_duplicate_or_mismatched_candidates():
    candidate = make_candidate()
    with pytest.raises(ValueError, match="already exists"):
        admit_successor(candidate, evaluation(), generation=2, existing_ids={"child-1"})
    with pytest.raises(ValueError, match="identities"):
        admit_successor(candidate, evaluation("other"), generation=2)
