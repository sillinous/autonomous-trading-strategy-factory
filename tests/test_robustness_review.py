from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from atsf.genome import StrategyGenome
from atsf.lineage import LineageRecord
from atsf.population import Candidate
from atsf.research_provenance import CandidateProvenance, _candidate_record
from atsf.research_scheduler import ResearchSchedule, ResearchScheduleAction
from atsf.robustness import RobustnessResult
from atsf.robustness_review import (
    RobustnessReviewDecision,
    expected_fingerprint,
    review_robustness,
)
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)


def make_candidate() -> Candidate:
    strategy = StrategySpec(
        name="review",
        universe=["TEST"],
        indicators=[Indicator(name="sma", period=20)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=100)]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=100)]),
        position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
        risk=RiskLimits(max_position=0.5),
    )
    strategy_id = StrategyGenome.from_strategy(strategy).strategy_id
    return Candidate(
        strategy=strategy,
        strategy_id=strategy_id,
        lineage=LineageRecord(strategy_id=strategy_id, generation=1),
    )


def make_evaluation(candidate: Candidate):
    robustness = RobustnessResult(
        candidate_id=candidate.strategy_id,
        baseline=SimpleNamespace(equity=pd.Series([1.0, 1.1])),
        scenarios=(("baseline", SimpleNamespace(equity=pd.Series([1.0, 1.1]))),),
        passed=True,
        reasons=(),
    )
    return SimpleNamespace(
        candidate_id=candidate.strategy_id,
        experiment=SimpleNamespace(experiment_id="exp-1"),
        backtest=SimpleNamespace(final_equity=1.1),
        validation_passed=True,
        fitness=SimpleNamespace(score=1.2),
        walk_forward=SimpleNamespace(oos_sharpe=1.2),
        monte_carlo=SimpleNamespace(pass_rate=0.99),
        perturbation=SimpleNamespace(pass_rate=0.95),
        regime=SimpleNamespace(score=-0.01),
        robustness=robustness,
        promotion=SimpleNamespace(eligible=True),
    )


def make_schedule(generation=1, action=ResearchScheduleAction.READY_FOR_ROBUSTNESS_REVIEW):
    return ResearchSchedule(
        action=action,
        health=SimpleNamespace(),
        generation=generation,
        execution_authority=False,
        reasons=(),
    )


def make_provenance(candidate, evaluation, generation=1, seed=7):
    record = _candidate_record(candidate, generation, evaluation, seed)
    return CandidateProvenance(
        strategy_id=record.strategy_id,
        generation=record.generation,
        parent_strategy_ids=record.parent_strategy_ids,
        genome_digest=record.genome_digest,
        evaluation_digest=record.evaluation_digest,
        research_seed=record.research_seed,
    )


def test_robustness_review_admits_only_complete_evidence():
    candidate = make_candidate()
    evaluation = make_evaluation(candidate)
    provenance = make_provenance(candidate, evaluation)

    review = review_robustness(
        candidate,
        evaluation,
        make_schedule(),
        provenance,
    )

    assert review.decision is RobustnessReviewDecision.ADMIT
    assert review.execution_authority is False
    assert review.evidence_fingerprint == provenance.evaluation_digest


def test_robustness_review_rejects_scheduler_not_ready():
    candidate = make_candidate()
    evaluation = make_evaluation(candidate)
    provenance = make_provenance(candidate, evaluation)

    review = review_robustness(
        candidate,
        evaluation,
        make_schedule(action=ResearchScheduleAction.CONTINUE_RESEARCH),
        provenance,
    )

    assert review.decision is RobustnessReviewDecision.REJECT
    assert any("scheduler" in reason for reason in review.reasons)


def test_robustness_review_rejects_failed_robustness():
    candidate = make_candidate()
    evaluation = make_evaluation(candidate)
    evaluation = replace(evaluation, robustness=replace(evaluation.robustness, passed=False))
    provenance = make_provenance(candidate, evaluation)

    review = review_robustness(candidate, evaluation, make_schedule(), provenance)

    assert review.decision is RobustnessReviewDecision.REJECT
    assert any("robustness stress scenarios" in reason for reason in review.reasons)


def test_robustness_review_rejects_tampered_provenance():
    candidate = make_candidate()
    evaluation = make_evaluation(candidate)
    provenance = make_provenance(candidate, evaluation)
    tampered = replace(provenance, evaluation_digest="0" * 64)

    review = review_robustness(candidate, evaluation, make_schedule(), tampered)

    assert review.decision is RobustnessReviewDecision.REJECT
    assert any("evaluation digest" in reason for reason in review.reasons)


def test_robustness_review_rejects_noneligible_candidate():
    candidate = make_candidate()
    evaluation = make_evaluation(candidate)
    evaluation = replace(evaluation, promotion=SimpleNamespace(eligible=False))
    provenance = make_provenance(candidate, evaluation)

    review = review_robustness(candidate, evaluation, make_schedule(), provenance)

    assert review.decision is RobustnessReviewDecision.REJECT
    assert any("promotion-eligible" in reason for reason in review.reasons)


def test_robustness_review_rejects_nonfinite_evidence():
    candidate = make_candidate()
    evaluation = make_evaluation(candidate)
    evaluation = replace(evaluation, walk_forward=SimpleNamespace(oos_sharpe=float("nan")))
    provenance = make_provenance(candidate, evaluation)

    review = review_robustness(candidate, evaluation, make_schedule(), provenance)

    assert review.decision is RobustnessReviewDecision.REJECT
    assert any("oos_sharpe" in reason for reason in review.reasons)


def test_expected_fingerprint_is_deterministic():
    candidate = make_candidate()
    evaluation = make_evaluation(candidate)
    provenance = make_provenance(candidate, evaluation)

    first = expected_fingerprint(candidate, evaluation, provenance)
    second = expected_fingerprint(candidate, evaluation, provenance)

    assert first == second == provenance.evaluation_digest


def test_robustness_review_rejects_generation_mismatch():
    candidate = make_candidate()
    evaluation = make_evaluation(candidate)
    provenance = make_provenance(candidate, evaluation, generation=1)

    review = review_robustness(
        candidate,
        evaluation,
        make_schedule(generation=2),
        provenance,
    )

    assert review.decision is RobustnessReviewDecision.REJECT
    assert any("generation" in reason for reason in review.reasons)


def test_review_rejects_invalid_provenance_seed():
    candidate = make_candidate()
    evaluation = make_evaluation(candidate)
    provenance = make_provenance(candidate, evaluation)
    invalid = replace(provenance, research_seed=-1)

    review = review_robustness(candidate, evaluation, make_schedule(), invalid)

    assert review.decision is RobustnessReviewDecision.REJECT
    assert any("seed" in reason for reason in review.reasons)


def test_missing_candidate_attributes_fail_closed():
    candidate = make_candidate()
    evaluation = make_evaluation(candidate)
    provenance = make_provenance(candidate, evaluation)
    broken = SimpleNamespace(candidate_id=candidate.strategy_id)

    with pytest.raises(AttributeError):
        expected_fingerprint(candidate, broken, provenance)
