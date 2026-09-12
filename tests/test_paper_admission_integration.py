import sqlite3

from atsf.paper_admission import admit_to_paper
from atsf.paper_admission_store import PaperAdmissionStore
from atsf.lifecycle import StrategyLifecycleStage


def test_paper_admission_persists_deterministic_identity():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    # Minimal persistence fixture for the admission store's run foreign key.
    connection.execute("CREATE TABLE portfolio_runs (run_id TEXT PRIMARY KEY, halted INTEGER NOT NULL)")
    connection.execute("INSERT INTO portfolio_runs(run_id, halted) VALUES ('run-1', 0)")
    store = PaperAdmissionStore.__new__(PaperAdmissionStore)
    store._connection = connection
    connection.execute("""CREATE TABLE paper_admissions (
        admission_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL, run_id TEXT NOT NULL UNIQUE,
        certificate_id TEXT NOT NULL, source_stage TEXT NOT NULL, target_stage TEXT NOT NULL,
        reason TEXT NOT NULL, admission_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    connection.commit()

    from atsf.paper_admission_record import build_admission_record
    record = build_admission_record("strategy-1", "run-1", "cert-1")
    store.save(record)
    assert store.get("run-1") == record
