import pytest

from atsf.paper_admission_record import build_admission_record
from atsf.paper_admission_store import PaperAdmissionStore
from atsf.registry import ExperimentRegistry


def _registry_with_run(path=":memory:") -> ExperimentRegistry:
    registry = ExperimentRegistry(path)
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

    assert store.save(changed) == changed
    assert store.get("run-1", "strategy-2") == changed
    with pytest.raises(ValueError, match="strategy_id is required"):
        store.get("run-1")


def test_paper_admission_store_rejects_same_strategy_mutation():
    registry = _registry_with_run()
    store = PaperAdmissionStore(registry)
    record = build_admission_record("strategy-1", "run-1", "cert-1")
    store.save(record)
    changed = build_admission_record("strategy-1", "run-1", "cert-2")

    with pytest.raises(ValueError, match="immutable"):
        store.save(changed)


def test_paper_admission_store_rejects_halted_run():
    registry = _registry_with_run()
    registry._connection.execute("UPDATE portfolio_runs SET halted = 1 WHERE run_id = 'run-1'")
    registry._connection.commit()
    store = PaperAdmissionStore(registry)
    store.save(build_admission_record("strategy-1", "run-1", "cert-1"))

    assert store.verify("run-1", "strategy-1") is False


def test_paper_admission_survives_registry_restart(tmp_path):
    path = tmp_path / "atsf.sqlite3"
    first = _registry_with_run(path)
    record = build_admission_record("strategy-1", "run-1", "cert-1")
    PaperAdmissionStore(first).save(record)
    first.close()

    reopened = ExperimentRegistry(path)
    store = PaperAdmissionStore(reopened)
    assert store.get("run-1") == record
    assert store.get("run-1").admission_id == record.admission_id
    reopened.close()


def test_legacy_single_strategy_schema_migrates_and_allows_second_strategy(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    registry = _registry_with_run(path)
    registry._connection.execute(
        """CREATE TABLE paper_admissions_legacy_seed (
            admission_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL,
            run_id TEXT NOT NULL UNIQUE, certificate_id TEXT NOT NULL,
            source_stage TEXT NOT NULL, target_stage TEXT NOT NULL,
            reason TEXT NOT NULL, admission_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    record = build_admission_record("strategy-legacy", "run-1", "cert-legacy")
    registry._connection.execute(
        "INSERT INTO paper_admissions_legacy_seed(admission_id,strategy_id,run_id,certificate_id,source_stage,target_stage,reason,admission_json) VALUES (?,?,?,?,?,?,?,?)",
        (record.admission_id, record.strategy_id, record.run_id, record.certificate_id, record.source_stage, record.target_stage, record.reason, PaperAdmissionStore._payload(record)),
    )
    registry._connection.execute("ALTER TABLE paper_admissions_legacy_seed RENAME TO paper_admissions")
    registry._connection.commit()

    store = PaperAdmissionStore(registry)
    second = build_admission_record("strategy-second", "run-1", "cert-second")
    store.save(second)
    assert store.get("run-1", "strategy-legacy") == record
    assert store.get("run-1", "strategy-second") == second
    assert {item.strategy_id for item in store.list_for_run("run-1")} == {"strategy-legacy", "strategy-second"}
    registry.close()
