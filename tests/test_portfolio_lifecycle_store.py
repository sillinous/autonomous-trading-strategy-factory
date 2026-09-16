import sqlite3

import pytest

from atsf.portfolio_lifecycle_store import PortfolioLifecycleStore


def _record(store: PortfolioLifecycleStore, generation: int = 1):
    return store.new_record(
        portfolio_id="portfolio-a",
        generation=generation,
        strategy_ids=("strategy-a", "strategy-b"),
        weights={"strategy-a": 0.6, "strategy-b": 0.4},
        health_status="HEALTHY",
        decision="CONTINUE_PORTFOLIO",
        total_return=0.12,
        volatility=0.08,
        max_risk_fraction=0.58,
        breached_limits=(),
        replace_strategy_ids=(),
        created_at="2026-09-15T00:00:00+00:00",
    )


def test_round_trip_and_restart(tmp_path):
    path = tmp_path / "portfolio.db"
    connection = sqlite3.connect(path)
    store = PortfolioLifecycleStore(connection)
    original = _record(store)
    store.append(original)
    assert store.latest("portfolio-a") == original
    connection.close()

    reopened = PortfolioLifecycleStore(sqlite3.connect(path))
    assert reopened.latest("portfolio-a") == original
    reopened.verify("portfolio-a")


def test_fingerprint_is_deterministic():
    a = _record(PortfolioLifecycleStore(sqlite3.connect(":memory:")))
    b = _record(PortfolioLifecycleStore(sqlite3.connect(":memory:")))
    assert a.fingerprint == b.fingerprint


def test_tampering_is_detected(tmp_path):
    connection = sqlite3.connect(tmp_path / "portfolio.db")
    store = PortfolioLifecycleStore(connection)
    store.append(_record(store))
    connection.execute("UPDATE portfolio_lifecycle SET decision = 'RESEARCH_REPLACEMENT'")
    connection.commit()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        store.latest("portfolio-a")


def test_schema_missing_column_fails_closed():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE portfolio_lifecycle (sequence INTEGER PRIMARY KEY, portfolio_id TEXT NOT NULL)"
    )
    with pytest.raises(ValueError, match="explicit migration is required"):
        PortfolioLifecycleStore(connection)


def test_unsupported_schema_version_fails_closed():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE portfolio_lifecycle (
            sequence INTEGER PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            generation INTEGER NOT NULL,
            strategy_ids TEXT NOT NULL,
            weights TEXT NOT NULL,
            health_status TEXT NOT NULL,
            decision TEXT NOT NULL,
            total_return REAL NOT NULL,
            volatility REAL NOT NULL,
            max_risk_fraction REAL NOT NULL,
            breached_limits TEXT NOT NULL,
            replace_strategy_ids TEXT NOT NULL,
            execution_authority INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            schema_version INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO portfolio_lifecycle VALUES (1,'p',0,'[]','[]','HEALTHY','CONTINUE_PORTFOLIO',0,0,0,'[]','[]',0,'2026-09-15T00:00:00+00:00','x',999)"
    )
    connection.commit()
    with pytest.raises(ValueError, match="unsupported portfolio lifecycle schema version"):
        PortfolioLifecycleStore(connection)


def test_generation_must_increase():
    connection = sqlite3.connect(":memory:")
    store = PortfolioLifecycleStore(connection)
    store.append(_record(store, 2))
    with pytest.raises(ValueError, match="generation must increase"):
        store.append(_record(store, 2))
    with pytest.raises(ValueError, match="generation must increase"):
        store.append(_record(store, 1))


def test_execution_authority_is_never_persisted():
    connection = sqlite3.connect(":memory:")
    store = PortfolioLifecycleStore(connection)
    record = _record(store)
    assert record.execution_authority is False
    assert store.append(record).execution_authority is False


def test_nonfinite_record_is_rejected():
    connection = sqlite3.connect(":memory:")
    store = PortfolioLifecycleStore(connection)
    with pytest.raises(ValueError, match="non-finite"):
        store.new_record(
            portfolio_id="portfolio-a",
            generation=1,
            strategy_ids=("strategy-a",),
            weights={"strategy-a": 1.0},
            health_status="HEALTHY",
            decision="CONTINUE_PORTFOLIO",
            total_return=float("nan"),
            volatility=0.1,
            max_risk_fraction=1.0,
            breached_limits=(),
            replace_strategy_ids=(),
        )


def test_append_in_transaction_rolls_back_cleanly():
    connection = sqlite3.connect(":memory:")
    store = PortfolioLifecycleStore(connection)
    record = _record(store)
    connection.execute("BEGIN")
    store.append_in_transaction(record)
    assert store.latest("portfolio-a") == record
    connection.rollback()
    assert store.latest("portfolio-a") is None


def test_append_in_transaction_commits_with_outer_transaction():
    connection = sqlite3.connect(":memory:")
    store = PortfolioLifecycleStore(connection)
    record = _record(store)
    connection.execute("BEGIN")
    store.append_in_transaction(record)
    connection.commit()
    assert store.latest("portfolio-a") == record
    store.verify("portfolio-a")
