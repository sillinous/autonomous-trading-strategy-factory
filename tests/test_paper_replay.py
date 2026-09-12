from atsf.paper_replay import verify_paper_replay


def test_missing_admission_fails_closed():
    from atsf.registry import ExperimentRegistry

    registry = ExperimentRegistry()
    result = verify_paper_replay(registry, "missing-run")
    assert not result.replayable
    assert "paper admission is missing" in result.reasons
