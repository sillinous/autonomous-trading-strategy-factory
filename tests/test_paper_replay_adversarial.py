from atsf.paper_admission_record import PaperAdmissionRecord, build_admission_record
from atsf.paper_admission_store import PaperAdmissionStore
from atsf.paper_replay import verify_paper_replay
from atsf.registry import ExperimentRegistry


def _run(registry: ExperimentRegistry, run_id: str = "run-1", halted: int = 0) -> None:
    registry._connection.execute(
        "INSERT INTO portfolios(portfolio_id, definition_json) VALUES (?, ?)",
        ("portfolio-1", "{}"),
    )
    registry._connection.execute(
        "INSERT INTO portfolio_runs(run_id, portfolio_id, final_equity, halted, halt_reason) VALUES (?, ?, ?, ?, ?)",
        (run_id, "portfolio-1", 100000.0, halted, "test halt" if halted else None),
    )
    registry._connection.commit()


def test_halted_run_is_not_replayable():
    registry = ExperimentRegistry()
    _run(registry, halted=1)
    PaperAdmissionStore(registry).save(build_admission_record("strategy-1", "run-1", "cert-1"))
    result = verify_paper_replay(registry, "run-1")
    assert not result.replayable
    assert "portfolio execution run is halted" in result.reasons
    registry.close()


def test_tampered_stage_is_rejected():
    registry = ExperimentRegistry()
    _run(registry)
    record = build_admission_record("strategy-1", "run-1", "cert-1")
    tampered = PaperAdmissionRecord(
        admission_id=record.admission_id,
        strategy_id=record.strategy_id,
        run_id=record.run_id,
        certificate_id=record.certificate_id,
        source_stage="VALIDATED",
        target_stage="PAPER",
        reason=record.reason,
    )
    PaperAdmissionStore(registry).save(tampered)
    result = verify_paper_replay(registry, "run-1")
    assert not result.replayable
    assert "admission source stage is not PROMOTED" in result.reasons
    registry.close()
