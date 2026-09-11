from atsf.execution_manifest import execution_manifest, verify_execution_manifest
from atsf.portfolio_audit import PortfolioAuditEvent


def events() -> list[PortfolioAuditEvent]:
    return [
        PortfolioAuditEvent(0, "a", "buy", "2026-01-01T00:00:00", 1.0, 100.0, 0.01),
        PortfolioAuditEvent(1, "a", "sell", "2026-01-02T00:00:00", 1.0, 101.0, 0.01),
    ]


def test_manifest_is_deterministic_and_order_independent() -> None:
    original = events()
    assert execution_manifest(original) == execution_manifest(list(reversed(original)))


def test_manifest_detects_modified_event() -> None:
    original = events()
    manifest = execution_manifest(original)
    modified = [original[0], PortfolioAuditEvent(1, "a", "sell", "2026-01-02T00:00:00", 1.0, 102.0, 0.01)]
    assert not verify_execution_manifest(modified, manifest)


def test_manifest_rejects_gaps() -> None:
    original = events()
    manifest = execution_manifest(original)
    gapped = [original[0], PortfolioAuditEvent(2, "a", "sell", "2026-01-02T00:00:00", 1.0, 101.0, 0.01)]
    try:
        verify_execution_manifest(gapped, manifest)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("expected non-contiguous audit ledger to be rejected")
