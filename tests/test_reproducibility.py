import pytest

from atsf.portfolio_executor import execute_persisted_portfolio
from atsf.portfolio_replay import verify_persisted_portfolio_run
from atsf.reproducibility import build_reproducibility_certificate
from atsf.registry import ExperimentRegistry
from tests.test_portfolio_executor import make_data, seed_persisted_portfolio


def execute(registry: ExperimentRegistry):
    first, second, dataset_version, _bundle_version = seed_persisted_portfolio(registry)
    result = execute_persisted_portfolio(
        registry,
        "portfolio-1",
        {first: make_data(), second: make_data()},
        dataset_version=dataset_version,
    )
    return result


def test_reproducibility_certificate_is_deterministic() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)
    assert verification.valid is True
    run = registry.get_portfolio_run(result.identity.run_id)
    first = build_reproducibility_certificate(run, verification)
    second = build_reproducibility_certificate(run, verification)
    assert first == second
    assert len(first.certificate_id) == 24
    assert first.verified is True
    assert first.event_count == len(result.audit_events)
    registry.close()


def test_unverified_run_cannot_be_certified() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)
    bad = verification.__class__(verification.run_id, False, verification.manifest, "tampered")
    run = registry.get_portfolio_run(result.identity.run_id)
    with pytest.raises(ValueError, match="unverified"):
        build_reproducibility_certificate(run, bad)
    registry.close()
