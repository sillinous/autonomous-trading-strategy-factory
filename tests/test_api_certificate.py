from fastapi.testclient import TestClient

from atsf.api import create_app
from atsf.registry import ExperimentRegistry
from tests.test_api import bars, seed


def test_certificate_endpoint_persists_and_retrieves_immutable_attestation(monkeypatch):
    monkeypatch.delenv("ATSF_API_KEY", raising=False)
    registry = ExperimentRegistry()
    strategy_id = seed(registry)
    client = TestClient(create_app(registry))

    response = client.post(
        "/portfolios/portfolio-1/paper-runs",
        json={"dataset_version": "v1", "data": {strategy_id: bars()}},
    )
    assert response.status_code == 201
    run_id = response.json()["run_id"]

    request = {"dataset_version": "v1", "data": {strategy_id: bars()}}
    first = client.post(f"/runs/{run_id}/certificate", json=request)
    second = client.post(f"/runs/{run_id}/certificate", json=request)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()

    stored = client.get(f"/runs/{run_id}/certificate")
    assert stored.status_code == 200
    assert stored.json() == first.json()

    registry._connection.execute(
        "UPDATE reproducibility_certificates SET certificate_json = ? WHERE run_id = ?",
        ('{"certificate_id":"tampered"}', run_id),
    )
    registry._connection.commit()
    retrieved = client.get(f"/runs/{run_id}/certificate")
    assert retrieved.status_code == 200
    assert retrieved.json()["certificate_id"] == "tampered"
    registry.close()
