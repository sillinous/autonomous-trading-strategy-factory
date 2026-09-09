from atsf.fitness import FitnessResult
from atsf.ranking import rank_candidates


def test_ranking_orders_eligible_candidates_deterministically():
    good = FitnessResult(2.0, True, ())
    better = FitnessResult(3.0, True, ())
    rejected = FitnessResult(99.0, False, ("rejected",))
    ranked = rank_candidates([
        ("a", good, 0.8, 0.5),
        ("b", better, 0.9, 0.5),
        ("c", rejected, 1.0, 1.0),
    ])
    assert [item.candidate_id for item in ranked] == ["b", "a"]


def test_ranking_rejects_invalid_weights():
    try:
        rank_candidates([], fitness_weight=-1.0)
    except ValueError as exc:
        assert "weights" in str(exc)
    else:
        raise AssertionError("expected invalid-weight failure")
