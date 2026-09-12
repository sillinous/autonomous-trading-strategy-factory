from atsf.paper_admission_record import build_admission_record


def test_admission_record_is_deterministic():
    first = build_admission_record("strategy-1", "run-1", "cert-1")
    second = build_admission_record("strategy-1", "run-1", "cert-1")
    assert first == second
    assert len(first.admission_id) == 24
    assert first.source_stage == "PROMOTED"
    assert first.target_stage == "PAPER"


def test_admission_record_changes_when_identity_changes():
    first = build_admission_record("strategy-1", "run-1", "cert-1")
    changed = build_admission_record("strategy-2", "run-1", "cert-1")
    assert first.admission_id != changed.admission_id
