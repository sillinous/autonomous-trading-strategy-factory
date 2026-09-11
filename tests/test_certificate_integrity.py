import json

from atsf.certificate_integrity import verify_persisted_certificate
from atsf.portfolio_executor import execute_persisted_portfolio
from atsf.portfolio_replay import verify_persisted_portfolio_run
from atsf.reproducibility import build_reproducibility_certificate
from atsf.registry import ExperimentRegistry
from tests.test_portfolio_executor import make_data, seed_persisted_portfolio


def test_persisted_certificate_integrity_verifies():
    registry = ExperimentRegistry()
    first, second, dataset_version, _ = seed_persisted_portfolio(registry)
    result = execute_persisted_portfolio(registry, "portfolio-1", {first: make_data(), second: make_data()}, dataset_version=dataset_version)
    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)
    certificate = build_reproducibility_certificate(registry, registry.get_portfolio_run(result.identity.run_id), verification)
    registry.save_reproducibility_certificate(certificate)
    integrity = verify_persisted_certificate(registry, result.identity.run_id)
    assert integrity.valid is True
    assert integrity.certificate_id == certificate.certificate_id
    registry.close()


def test_tampered_certificate_is_detected():
    registry = ExperimentRegistry()
    first, second, dataset_version, _ = seed_persisted_portfolio(registry)
    result = execute_persisted_portfolio(registry, "portfolio-1", {first: make_data(), second: make_data()}, dataset_version=dataset_version)
    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)
    certificate = build_reproducibility_certificate(registry, registry.get_portfolio_run(result.identity.run_id), verification)
    registry.save_reproducibility_certificate(certificate)
    payload = registry.get_reproducibility_certificate(result.identity.run_id)
    payload["research_fingerprint"] = "tampered"
    registry._connection.execute(
        "UPDATE reproducibility_certificates SET certificate_json = ? WHERE run_id = ?",
        (json.dumps(payload, sort_keys=True), result.identity.run_id),
    )
    registry._connection.commit()
    integrity = verify_persisted_certificate(registry, result.identity.run_id)
    assert integrity.valid is False
    assert "certificate_id" in (integrity.reason or "")
    registry.close()
