import pytest

from atsf.portfolio_executor import execute_persisted_portfolio
from atsf.portfolio_replay import verify_persisted_portfolio_run
from atsf.registry import ExperimentRegistry
from tests.test_portfolio_executor import make_data, seed_persisted_portfolio


def execute(registry: ExperimentRegistry):
    first, second, dataset_version, bundle_version = seed_persisted_portfolio(registry)
    result = execute_persisted_portfolio(
        registry,
        "portfolio-1",
        {first: make_data(), second: make_data()},
        dataset_version=dataset_version,
    )
    return result, dataset_version, bundle_version


def test_persisted_execution_replay_verifies_identity_and_ledger_commitments() -> None:
    registry = ExperimentRegistry()
    result, _dataset_version, _bundle_version = execute(registry)

    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)

    assert verification.valid is True
    assert verification.reason is None
    assert verification.manifest.event_count == len(result.audit_events)
    registry.close()


def test_persisted_execution_replay_fails_closed_when_ledger_is_tampered() -> None:
    registry = ExperimentRegistry()
    result, _dataset_version, _bundle_version = execute(registry)
    registry._connection.execute(
        "UPDATE portfolio_audit_events SET price = price + 1 WHERE run_id = ? AND sequence = 0",
        (result.identity.run_id,),
    )
    registry._connection.commit()

    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)

    assert verification.valid is False
    assert verification.reason == "persisted execution ledger fingerprint mismatch"
    registry.close()


def test_replay_fails_closed_when_execution_config_is_tampered() -> None:
    registry = ExperimentRegistry()
    result, _dataset_version, _bundle_version = execute(registry)
    registry._connection.execute(
        "UPDATE portfolio_run_provenance SET execution_config_json = REPLACE(execution_config_json, '1.0', '2.0') WHERE run_id = ?",
        (result.identity.run_id,),
    )
    registry._connection.commit()

    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)

    assert verification.valid is False
    assert "execution fingerprint" in verification.reason or "run ID" in verification.reason
    registry.close()


def test_replay_fails_closed_when_bundle_commitment_is_tampered() -> None:
    registry = ExperimentRegistry()
    result, _dataset_version, _bundle_version = execute(registry)
    registry._connection.execute(
        "UPDATE portfolio_run_provenance SET data_bundle_version = 'tampered' WHERE run_id = ?",
        (result.identity.run_id,),
    )
    registry._connection.commit()

    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)

    assert verification.valid is False
    assert "data_bundle_version" in verification.reason or "run ID" in verification.reason
    registry.close()


def test_replay_fails_closed_when_final_equity_is_tampered() -> None:
    registry = ExperimentRegistry()
    result, _dataset_version, _bundle_version = execute(registry)
    registry._connection.execute(
        "UPDATE portfolio_runs SET final_equity = final_equity + 1 WHERE run_id = ?",
        (result.identity.run_id,),
    )
    registry._connection.commit()

    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)

    assert verification.valid is False
    assert verification.reason is not None
    assert "accounting mismatch" in verification.reason
    registry.close()


def test_persisted_execution_replay_rejects_missing_commitment() -> None:
    registry = ExperimentRegistry()
    _first, _second, dataset_version, bundle_version = seed_persisted_portfolio(registry)
    registry.save_portfolio_run(
        "uncommitted-run",
        "portfolio-1",
        100_000.0,
        False,
        None,
        [],
        dataset_id="prices",
        dataset_version=dataset_version,
        data_bundle_version=bundle_version,
        execution_fingerprint="manual",
        execution_config={"initial_cash": 100_000.0},
    )

    verification = verify_persisted_portfolio_run(registry, "uncommitted-run")

    assert verification.valid is False
    assert "missing a ledger fingerprint commitment" in verification.reason
    registry.close()


@pytest.mark.parametrize("bad_run_id", ["does-not-exist"])
def test_persisted_execution_replay_rejects_unknown_run(bad_run_id: str) -> None:
    registry = ExperimentRegistry()
    with pytest.raises(ValueError, match="unknown portfolio run"):
        verify_persisted_portfolio_run(registry, bad_run_id)
    registry.close()
