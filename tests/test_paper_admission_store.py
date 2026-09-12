import pytest

from atsf.paper_admission_record import build_admission_record
from atsf.paper_admission_store import PaperAdmissionStore
from atsf.registry import ExperimentRegistry


def _registry_with_run() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    registry._connection.execute(
        "INSERT INTO portfolios(portfolio_id, definition_json) VALUES (?, ?)",
        ("portfolio-1", "{}"),
    )
    registry._connection.execute(
        "INSERT INTO portfolio_runs(run_id, portfolio_id, final_equity, halted, halt_reason) VALUES (?, ?, ?, ?, ?)",
        ("run-1", "portfolio-1", 100000.0, 0, None),
    )
    registry._connection.commit()
    return registry


def test_paper_admission_store_is_idempotent():
    registry = _registry_with_run()
    store = PaperAdmissionStore(registry)
    record = build_admission_record("strategy-1", "run-1", "cert-1")

    assert store.save(record) == record
    assert store.save(record) == record
    assert store.get("run-1") == record


def test_paper_admission_store_rejects_mutation():
    registry = _registry_with_run()
    store = PaperAdmissionStore(registry)
    record = build_admission_record("strategy-1", "run-1", "cert-1")
    store.save(record)
    changed = build_admission_record("strategy-2", "run-1", "cert-1")

    with pytest.raises(ValueError, match="immutable"):
        store.save(changed)


def test_paper_admission_store_rejects_halted_run():
    registry = _registry_with_run()
    registry._connection.execute("UPDATE portfolio_runs SET halted = 1 WHERE run_id = 'run-1'")
    registry._connection.commit()
    store = PaperAdmissionStore(registry)
    store.save(build_admission_record("strategy-1", "run-1", "cert-1"))

    assert store.verify("run-1") is False
