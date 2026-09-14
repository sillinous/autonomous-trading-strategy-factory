from pathlib import Path

import pytest

from atsf.live_execution_intent import build_execution_intent
from atsf.live_execution_intent_store import LiveExecutionIntentStore
from atsf.live_authorization import LiveAuthorization, LiveAuthorizationDecision
from atsf.live_risk_gateway import issue_live_risk_certificate
from atsf.registry import ExperimentRegistry
from atsf.sandbox_execution import (
    ExecutionMode,
    ExecutionState,
    MarketSnapshot,
    SandboxExecutionAdapter,
)
from atsf.sandbox_execution_journal import SandboxExecutionJournal


def make_context(tmp_path: Path):
    registry = ExperimentRegistry(tmp_path / "research.db")
    store = LiveExecutionIntentStore(registry)
    authorization = LiveAuthorization(
        strategy_id="strategy-1",
        decision=LiveAuthorizationDecision.AUTHORIZED,
        execution_authority=True,
        capital_fraction=0.05,
        reasons=(),
    )
    certificate = issue_live_risk_certificate(
        authorization, issued_at=100.0, expires_at=200.0, nonce="nonce-1"
    )
    intent = build_execution_intent(
        certificate,
        intent_id="intent-1",
        symbol="SPY",
        side="BUY",
        quantity_fraction=0.02,
        now=150.0,
    )
    store.record(intent, now=150.0)
    return registry, store, certificate, intent


def test_sandbox_fills_deterministically_from_market_snapshot(tmp_path: Path):
    registry, store, certificate, intent = make_context(tmp_path)
    journal = SandboxExecutionJournal(registry)
    adapter = SandboxExecutionAdapter(store, journal)
    market = MarketSnapshot("SPY", bid=100.0, ask=100.25, timestamp=151.0)

    result = adapter.execute(intent, certificate, market, now=151.0)

    assert result.mode is ExecutionMode.SANDBOX
    assert result.state is ExecutionState.FILLED
    assert result.execution_authority is False
    assert result.fill is not None
    assert result.fill.price == 100.25
    assert store.get(intent.intent_id).status == "CONSUMED"
    assert [event.state for event in journal.events(intent.intent_id)] == [
        ExecutionState.CREATED,
        ExecutionState.VALIDATED,
        ExecutionState.ADMITTED,
        ExecutionState.CONSUMED,
        ExecutionState.FILLED,
    ]
    assert journal.verify(intent.intent_id)
    registry.close()


def test_sandbox_rejects_live_mode(tmp_path: Path):
    registry, store, certificate, intent = make_context(tmp_path)
    result = SandboxExecutionAdapter(store).execute(
        intent,
        certificate,
        MarketSnapshot("SPY", 100.0, 100.25, 151.0),
        now=151.0,
        mode=ExecutionMode.LIVE,
    )
    assert result.state is ExecutionState.REJECTED
    assert "live execution mode is disabled" in result.reasons
    assert store.get(intent.intent_id).status == "CREATED"
    registry.close()


def test_sandbox_is_idempotently_rejected_after_consumption(tmp_path: Path):
    registry, store, certificate, intent = make_context(tmp_path)
    adapter = SandboxExecutionAdapter(store)
    market = MarketSnapshot("SPY", 100.0, 100.25, 151.0)
    first = adapter.execute(intent, certificate, market, now=151.0)
    second = adapter.execute(intent, certificate, market, now=152.0)
    assert first.state is ExecutionState.FILLED
    assert second.state is ExecutionState.REJECTED
    assert "already consumed" in second.reasons
    registry.close()


def test_sandbox_revalidates_kill_switch_before_consumption(tmp_path: Path):
    registry, store, certificate, intent = make_context(tmp_path)
    journal = SandboxExecutionJournal(registry)
    result = SandboxExecutionAdapter(store, journal).execute(
        intent,
        certificate,
        MarketSnapshot("SPY", 100.0, 100.25, 151.0),
        now=151.0,
        kill_switch_engaged=True,
    )
    assert result.state is ExecutionState.REJECTED
    assert any("kill switch" in reason for reason in result.reasons)
    assert store.get(intent.intent_id).status == "CREATED"
    assert journal.latest(intent.intent_id).state is ExecutionState.REJECTED
    assert journal.verify(intent.intent_id)
    registry.close()


def test_sandbox_rejects_expired_certificate(tmp_path: Path):
    registry, store, certificate, intent = make_context(tmp_path)
    journal = SandboxExecutionJournal(registry)
    result = SandboxExecutionAdapter(store, journal).execute(
        intent,
        certificate,
        MarketSnapshot("SPY", 100.0, 100.25, 201.0),
        now=201.0,
    )
    assert result.state is ExecutionState.REJECTED
    assert any("expired" in reason for reason in result.reasons)
    assert store.get(intent.intent_id).status == "CREATED"
    assert journal.latest(intent.intent_id).state is ExecutionState.REJECTED
    assert journal.verify(intent.intent_id)
    registry.close()


def test_market_symbol_mismatch_does_not_consume_intent(tmp_path: Path):
    registry, store, certificate, intent = make_context(tmp_path)
    journal = SandboxExecutionJournal(registry)
    result = SandboxExecutionAdapter(store, journal).execute(
        intent,
        certificate,
        MarketSnapshot("QQQ", 100.0, 100.25, 151.0),
        now=151.0,
    )
    assert result.state is ExecutionState.REJECTED
    assert any("symbol" in reason for reason in result.reasons)
    assert store.get(intent.intent_id).status == "CREATED"
    assert journal.latest(intent.intent_id).state is ExecutionState.REJECTED
    assert journal.verify(intent.intent_id)
    registry.close()
