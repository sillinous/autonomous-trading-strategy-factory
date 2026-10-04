from atsf.registry import ExperimentRegistry


def test_external_snapshot_persists_idempotently(tmp_path):
    registry = ExperimentRegistry(tmp_path / "atsf.sqlite3")
    snapshot = {
        "fingerprint": "a" * 64,
        "source": "test-provider",
        "dataset": "daily_ohlcv",
        "fetched_at": "2026-10-04T00:00:00+00:00",
        "payload": {"symbol": "TEST", "records": [{"close": 123.45}]},
    }

    assert registry.save_external_snapshot(snapshot) == snapshot["fingerprint"]
    assert registry.save_external_snapshot(snapshot) == snapshot["fingerprint"]
    assert registry.get_external_snapshot(snapshot["fingerprint"]) == snapshot
    registry.close()


def test_external_snapshot_rejects_fingerprint_collision(tmp_path):
    registry = ExperimentRegistry(tmp_path / "atsf.sqlite3")
    snapshot = {
        "fingerprint": "b" * 64,
        "source": "test-provider",
        "dataset": "daily_ohlcv",
        "fetched_at": "2026-10-04T00:00:00+00:00",
        "payload": {"symbol": "TEST", "records": [{"close": 1.0}]},
    }
    registry.save_external_snapshot(snapshot)

    conflicting = {**snapshot, "payload": {"symbol": "TEST", "records": [{"close": 2.0}]}}
    try:
        registry.save_external_snapshot(conflicting)
    except ValueError as exc:
        assert "fingerprint collision" in str(exc)
    else:
        raise AssertionError("expected fingerprint collision rejection")
    registry.close()
