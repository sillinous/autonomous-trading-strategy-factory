from pathlib import Path

import pytest

from atsf.live_execution_intent import build_execution_intent
from atsf.live_execution_intent_store import LiveExecutionIntentStore
from atsf.live_authorization import LiveAuthorization, LiveAuthorizationDecision
from atsf.live_risk_gateway import issue_live_risk_certificate
from atsf.registry import ExperimentRegistry
from atsf.sandbox_execution import ExecutionState, MarketSnapshot, SandboxExecutionAdapter
from atsf.sandbox_execution_journal import SandboxExecutionJournal


def _context(tmp_path: Path):
    registry = ExperimentRegistry(tmp_path / "recovery.db")
    store = LiveExecutionIntentStore(registry)
    authorization = LiveAuthorization(
        strategy_id="strategy-1",
        decision=LiveAuthorizationDecision.AUTHORIZED,
        execution_authority=True,
        capital_fraction=0.05,
        reasons=(),
    )
    certificate = issue_live_risk_certificate(
        authorization, issued_at=100.0, expires_at=200.0, nonce="recovery-nonce"
    )
    intent = build_execution_intent(
        certificate,
        intent_id="recovery-intent",
        symbol="SPY",
        side="BUY",
        quantity_fraction=0.02,
        now=150.0,
    )
    store.record(intent, now=150.0)
    journal = SandboxExecutionJournal(registry)
    return registry, store, certificate, intent, journal


def test_restart_recovery_requires_verified_chain(tmp_path: Path):
    registry, store, certificate, intent, journal = _context(tmp_path)
    result = SandboxExecutionAdapter(store, journal).execute(
        intent,
        certificate,
        MarketSnapshot("SPY", 100.0, 100.25, 151.0),
        now=151.0,
    )
    assert result.state is ExecutionState.FILLED
    registry.close()

    reopened = ExperimentRegistry(tmp_path / "recovery.db")
    reopened_journal = SandboxExecutionJournal(reopened)
    reopened_store = LiveExecutionIntentStore(reopened)
    assert reopened_journal.verify(intent.intent_id)
    assert reopened_journal.recover_state(intent.intent_id) is ExecutionState.FILLED
    assert reopened_store.get(intent.intent_id).status == "CONSUMED"
    reopened.close()


def test_recovery_rejects_tampered_persisted_event(tmp_path: Path):
    registry, store, certificate, intent, journal = _context(tmp_path)
    SandboxExecutionAdapter(store, journal).execute(
        intent,
        certificate,
        MarketSnapshot("SPY", 100.0, 100.25, 151.0),
        now=151.0,
    )
    registry._connection.execute(
        "UPDATE sandbox_execution_events SET detail = 'tampered' WHERE intent_id = ? AND sequence = 5",
        (intent.intent_id,),
    )
    registry._connection.commit()
    with pytest.raises(ValueError, match="integrity verification failed"):
        journal.recover_state(intent.intent_id)
    registry.close()


def test_legacy_journal_schema_fails_closed(tmp_path: Path):
    registry = ExperimentRegistry(tmp_path / "legacy.db")
    registry._connection.execute(
        """CREATE TABLE sandbox_execution_events (
            intent_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            state TEXT NOT NULL,
            timestamp REAL NOT NULL,
            event_fingerprint TEXT NOT NULL UNIQUE,
            detail TEXT NOT NULL DEFAULT '',
            previous_fingerprint TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (intent_id, sequence)
        )"""
    )
    registry._connection.commit()
    with pytest.raises(ValueError, match="explicit migration is required"):
        SandboxExecutionJournal(registry)
    registry.close()
