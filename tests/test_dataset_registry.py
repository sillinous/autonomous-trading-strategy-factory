import sqlite3

import pytest

from atsf.dataset_bundle import DatasetBundleIdentity
from atsf.dataset_registry import DatasetRegistry


def identity(version: str = "abc123", *, source: str = "unspecified", timeframe: str = "1d") -> DatasetBundleIdentity:
    return DatasetBundleIdentity(
        dataset_id="prices",
        version=version,
        symbols=("AAA", "BBB"),
        rows=20,
        start="2025-01-01T00:00:00",
        end="2025-01-20T00:00:00",
        source=source,
        timeframe=timeframe,
    )


def test_register_is_idempotent_for_identical_metadata():
    registry = DatasetRegistry(sqlite3.connect(":memory:"))
    first = registry.register(identity(), source="fixture")
    second = registry.register(identity(), source="fixture")
    assert first == second
    assert first.source == "fixture"
    assert first.timeframe == "1d"
    assert first.schema_version == "ohlcv.v1"
    assert registry.require("prices", "abc123") == first


def test_registration_is_immutable():
    registry = DatasetRegistry(sqlite3.connect(":memory:"))
    registry.register(identity(), source="fixture")
    with pytest.raises(ValueError, match="immutable"):
        registry.register(identity(), source="vendor")


def test_explicit_identity_source_must_match_registration():
    registry = DatasetRegistry(sqlite3.connect(":memory:"))
    with pytest.raises(ValueError, match="source"):
        registry.register(identity(source="vendor"), source="fixture")


def test_unknown_dataset_is_rejected():
    registry = DatasetRegistry(sqlite3.connect(":memory:"))
    with pytest.raises(KeyError, match="unknown dataset"):
        registry.require("prices", "missing")
