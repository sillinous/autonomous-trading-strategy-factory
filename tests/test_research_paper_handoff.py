from types import SimpleNamespace

from atsf.research_paper_handoff import handoff_research_candidate_to_paper
from atsf.robustness_review import RobustnessReviewDecision


def make_review(strategy_id="strategy-1", admitted=True, authority=False):
    return SimpleNamespace(
        candidate_id=strategy_id,
        decision=(
            RobustnessReviewDecision.ADMIT
            if admitted
            else RobustnessReviewDecision.REJECT
        ),
        execution_authority=authority,
        evidence_fingerprint="a" * 64,
    )


def test_research_paper_handoff_rejects_review_mismatch(monkeypatch):
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("paper boundary must not be called")

    monkeypatch.setattr(
        "atsf.research_paper_handoff.handoff_successor_to_paper", unexpected
    )

    decision = handoff_research_candidate_to_paper(
        SimpleNamespace(),
        "strategy-1",
        "run-1",
        make_review(strategy_id="strategy-2"),
    )

    assert not decision.admitted
    assert not decision.execution_authority
    assert not called
    assert any("candidate_id" in reason for reason in decision.reasons)


def test_research_paper_handoff_rejects_nonadmitted_review(monkeypatch):
    monkeypatch.setattr(
        "atsf.research_paper_handoff.handoff_successor_to_paper",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paper boundary must not be called")
        ),
    )

    decision = handoff_research_candidate_to_paper(
        SimpleNamespace(),
        "strategy-1",
        "run-1",
        make_review(admitted=False),
    )

    assert not decision.admitted
    assert any("did not admit" in reason for reason in decision.reasons)


def test_research_paper_handoff_delegates_verified_boundary(monkeypatch):
    expected = SimpleNamespace(
        admitted=True,
        admission_id="admission-1",
        reasons=(),
    )
    captured = {}

    def fake_handoff(registry, strategy_id, run_id, *, reason):
        captured.update(
            registry=registry,
            strategy_id=strategy_id,
            run_id=run_id,
            reason=reason,
        )
        return expected

    monkeypatch.setattr(
        "atsf.research_paper_handoff.handoff_successor_to_paper", fake_handoff
    )
    registry = SimpleNamespace()
    review = make_review()

    decision = handoff_research_candidate_to_paper(
        registry,
        "strategy-1",
        "run-1",
        review,
        reason="test handoff",
    )

    assert decision.admitted
    assert decision.admission_id == "admission-1"
    assert decision.review_fingerprint == review.evidence_fingerprint
    assert not decision.execution_authority
    assert captured == {
        "registry": registry,
        "strategy_id": "strategy-1",
        "run_id": "run-1",
        "reason": "test handoff",
    }


def test_research_paper_handoff_rejects_execution_authority():
    decision = handoff_research_candidate_to_paper(
        SimpleNamespace(),
        "strategy-1",
        "run-1",
        make_review(authority=True),
    )

    assert not decision.admitted
    assert not decision.execution_authority
    assert any("execution authority" in reason for reason in decision.reasons)
