from atsf.execution_manifest import execution_manifest
from atsf.portfolio_audit import PortfolioAuditEvent
from atsf.replay import verify_replay


def events() -> list[PortfolioAuditEvent]:
    return [
        PortfolioAuditEvent(0, "a", "buy", "2026-01-01T00:00:00", 1.0, 100.0, 0.01),
        PortfolioAuditEvent(1, "a", "sell", "2026-01-02T00:00:00", 1.0, 101.0, 0.01),
    ]


def test_replay_verification_passes_for_identical_ledger():
    original = events()
    result = verify_replay(list(reversed(original)), execution_manifest(original))
    assert result.passed
    assert result.reason is None


def test_replay_verification_rejects_tampering():
    original = events()
    expected = execution_manifest(original)
    changed = [original[0], PortfolioAuditEvent(1, "a", "sell", "2026-01-02T00:00:00", 1.0, 102.0, 0.01)]
    result = verify_replay(changed, expected)
    assert not result.passed
    assert "fingerprint" in result.reason


def test_replay_verification_fails_closed_on_invalid_sequence():
    original = events()
    expected = execution_manifest(original)
    invalid = [original[0], PortfolioAuditEvent(2, "a", "sell", "2026-01-02T00:00:00", 1.0, 101.0, 0.01)]
    result = verify_replay(invalid, expected)
    assert not result.passed
    assert "contiguous" in result.reason
