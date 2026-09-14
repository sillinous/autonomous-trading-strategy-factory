from dataclasses import replace

import pytest

from atsf.live_authorization import LiveAuthorization, LiveAuthorizationDecision
from atsf.live_execution_intent import build_execution_intent
from atsf.live_execution_intent_store import LiveExecutionIntentStore
from atsf.live_risk_gateway import issue_live_risk_certificate
from atsf.registry import ExperimentRegistry


def certificate():
    authorization = LiveAuthorization(
        strategy_id="strategy-1",
        decision=LiveAuthorizationDecision.AUTHORIZED,
        execution_authority=True,
        capital_fraction=0.05,
        reasons=(),
    )
    return issue_live_risk_certificate(
        authorization, issued_at=100.0, expires_at=200.0, nonce="nonce-1"
    )


def intent():
    return build_execution_intent(
        certificate(), intent_id="intent-1", symbol="SPY", side="BUY",
        quantity_fraction=0.02, now=150.0
    )


def test_record_is_durable_and_idempotent(tmp_path):
    path = tmp_path / "registry.sqlite"
    first = intent()
    registry = ExperimentRegistry(path)
    store = LiveExecutionIntentStore(registry)
    assert store.record(first, now=151.0) == store.record(first, now=151.0)
    assert store.get(first.intent_id).status == "CREATED"
    registry._connection.close()

    reopened = ExperimentRegistry(path)
    assert LiveExecutionIntentStore(reopened).get(first.intent_id).intent_fingerprint == first.fingerprint


def test_record_rejects_mutation_under_same_intent_id():
    registry = ExperimentRegistry()
    store = LiveExecutionIntentStore(registry)
    first = intent()
    store.record(first, now=151.0)
    mutated = replace(first, quantity_fraction=0.03)
    with pytest.raises(ValueError, match="fingerprint is invalid"):
        store.record(mutated, now=151.0)


def test_record_rejects_same_fingerprint_under_different_id():
    registry = ExperimentRegistry()
    store = LiveExecutionIntentStore(registry)
    first = intent()
    store.record(first, now=151.0)
    duplicate_id = replace(first, intent_id="intent-2")
    with pytest.raises(ValueError, match="fingerprint is invalid"):
        store.record(duplicate_id, now=151.0)


def test_consume_is_atomic_and_replay_safe():
    registry = ExperimentRegistry()
    store = LiveExecutionIntentStore(registry)
    first = intent()
    store.record(first, now=151.0)
    consumed = store.consume(first, certificate(), now=152.0)
    assert consumed.status == "CONSUMED"
    with pytest.raises(ValueError, match="already consumed"):
        store.consume(first, certificate(), now=153.0)


def test_consume_requires_journal_entry():
    registry = ExperimentRegistry()
    store = LiveExecutionIntentStore(registry)
    with pytest.raises(ValueError, match="has not been journaled"):
        store.consume(intent(), certificate(), now=152.0)


def test_consume_revalidates_certificate_and_kill_switch():
    registry = ExperimentRegistry()
    store = LiveExecutionIntentStore(registry)
    first = intent()
    store.record(first, now=151.0)
    with pytest.raises(ValueError, match="kill switch"):
        store.consume(first, certificate(), now=152.0, kill_switch_engaged=True)


def test_consume_rejects_certificate_mismatch():
    registry = ExperimentRegistry()
    store = LiveExecutionIntentStore(registry)
    first = intent()
    store.record(first, now=151.0)
    other = issue_live_risk_certificate(
        LiveAuthorization(
            strategy_id="strategy-1",
            decision=LiveAuthorizationDecision.AUTHORIZED,
            execution_authority=True,
            capital_fraction=0.05,
            reasons=(),
        ),
        issued_at=100.0,
        expires_at=200.0,
        nonce="nonce-2",
    )
    with pytest.raises(ValueError, match="certificate fingerprint"):
        store.consume(first, other, now=152.0)


def test_consume_rejects_expired_certificate():
    registry = ExperimentRegistry()
    store = LiveExecutionIntentStore(registry)
    first = intent()
    store.record(first, now=151.0)
    with pytest.raises(ValueError, match="expired"):
        store.consume(first, certificate(), now=201.0)
