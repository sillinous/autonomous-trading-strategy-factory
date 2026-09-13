import sqlite3

from atsf.lifecycle import StrategyLifecycleStage
from atsf.lifecycle_store import LifecycleStore
from atsf.paper_admission import admit_to_paper
from atsf.paper_admission_store import PaperAdmissionStore
from atsf.registry import ExperimentRegistry


def test_admission_failure_rolls_back_lifecycle():
    registry = ExperimentRegistry()
    registry._connection.execute("INSERT INTO portfolios(portfolio_id, definition_json) VALUES ('p1', '{}')")
    registry._connection.execute(
        "INSERT INTO portfolio_runs(run_id, portfolio_id, final_equity, halted, halt_reason) VALUES ('r1', 'p1', 100.0, 0, NULL)"
    )
    registry._connection.commit()
    LifecycleStore(registry._connection).save("s1", StrategyLifecycleStage.PROMOTED, reason="promotion")

    class FailingAdmissionStore(PaperAdmissionStore):
        def save(self, record):
            raise ValueError("forced admission persistence failure")

    original = PaperAdmissionStore.save
    PaperAdmissionStore.save = FailingAdmissionStore.save
    try:
        decision = admit_to_paper(
            "s1",
            source_stage=StrategyLifecycleStage.PROMOTED,
            run={"run_id": "r1", "halted": False},
            certificate={
                "run_id": "r1", "certificate_id": "c1", "verified": True,
                "dataset_version": "d1", "data_bundle_version": "b1", "execution_fingerprint": "e1",
            },
            registry=registry,
        )
    finally:
        PaperAdmissionStore.save = original

    assert decision.admitted is False
    assert LifecycleStore(registry._connection).get("s1").stage is StrategyLifecycleStage.PROMOTED
    assert registry._connection.execute("SELECT COUNT(*) FROM paper_admissions").fetchone()[0] == 0
    registry.close()
