import pytest

from atsf.portfolio_executor import execute_persisted_portfolio
from atsf.portfolio_replay import verify_persisted_portfolio_run
from atsf.reproducibility import build_reproducibility_certificate
from atsf.registry import ExperimentRegistry
from tests.test_portfolio_executor import make_data, seed_persisted_portfolio


def make_certificate(registry: ExperimentRegistry):
    first, second, dataset_version, _bundle_version = seed_persisted_portfolio(registry)
    result = execute_persisted_portfolio(
        registry,
        "portfolio-1",
        {first: make_data(), second: make_data()},
        dataset_version=dataset_version,
    )
    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)
    assert verification.valid
    certificate = build_reproducibility_certificate(
        registry,
        registry.get_portfolio_run(result.identity.run_id),
        verification,
    )
    return result.identity.run_id, certificate


def test_certificate_is_persisted_and_idempotent():
    registry = ExperimentRegistry()
    run_id, certificate = make_certificate(registry)
    registry.save_reproducibility_certificate(certificate)
    registry.save_reproducibility_certificate(certificate)
    stored = registry.get_reproducibility_certificate(run_id)
    assert stored is not None
    assert stored["certificate_id"] == certificate.certificate_id
    assert stored["provenance_graph_fingerprint"] == certificate.provenance_graph_fingerprint
    registry.close()


def test_certificate_cannot_be_replaced():
    registry = ExperimentRegistry()
    run_id, certificate = make_certificate(registry)
    registry.save_reproducibility_certificate(certificate)
    tampered = certificate.__class__(
        run_id=certificate.run_id,
        certificate_id="tampered-certificate",
        provenance_schema_version=certificate.provenance_schema_version,
        verified=True,
        dataset_version=certificate.dataset_version,
        data_bundle_version=certificate.data_bundle_version,
        execution_fingerprint=certificate.execution_fingerprint,
        research_fingerprint=certificate.research_fingerprint,
        provenance_graph_fingerprint=certificate.provenance_graph_fingerprint,
        ledger_fingerprint=certificate.ledger_fingerprint,
        lineage_fingerprint=certificate.lineage_fingerprint,
        attribution_fingerprint=certificate.attribution_fingerprint,
        verification_fingerprint=certificate.verification_fingerprint,
        event_count=certificate.event_count,
    )
    with pytest.raises(ValueError, match="immutable"):
        registry.save_reproducibility_certificate(tampered)
    assert registry.get_reproducibility_certificate(run_id)["certificate_id"] == certificate.certificate_id
    registry.close()


def test_unknown_run_cannot_receive_certificate():
    registry = ExperimentRegistry()
    _run_id, certificate = make_certificate(registry)
    with pytest.raises(ValueError, match="unknown portfolio run"):
        fake = certificate.__class__(
            run_id="missing-run",
            certificate_id=certificate.certificate_id,
            provenance_schema_version=certificate.provenance_schema_version,
            verified=True,
            dataset_version=certificate.dataset_version,
            data_bundle_version=certificate.data_bundle_version,
            execution_fingerprint=certificate.execution_fingerprint,
            research_fingerprint=certificate.research_fingerprint,
            provenance_graph_fingerprint=certificate.provenance_graph_fingerprint,
            ledger_fingerprint=certificate.ledger_fingerprint,
            lineage_fingerprint=certificate.lineage_fingerprint,
            attribution_fingerprint=certificate.attribution_fingerprint,
            verification_fingerprint=certificate.verification_fingerprint,
            event_count=certificate.event_count,
        )
        registry.save_reproducibility_certificate(fake)
    registry.close()
