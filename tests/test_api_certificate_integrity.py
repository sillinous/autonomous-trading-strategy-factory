from fastapi.testclient import TestClient

from atsf.api import create_app
from atsf.registry import ExperimentRegistry
from tests.test_api import bars, seed


def test_certificate_integrity_endpoint(monkeypatch):
    monkeypatch.delenv("ATSF_API_KEY", raising=False)
    registry = ExperimentRegistry()
    strategy_id = seed(registry)
    client = TestClient(create_app(registry))
    response = client.post("/portfolios/portfolio-1/paper-runs", json={"dataset_version": "v1", "data": {strategy_id: bars()}})
    assert response.status_code == 201
    run_id = response.json()["run_id"]
    request = {"dataset_version": "v1", "data": {strategy_id: bars()}}
    certificate = client.post(f"/runs/{run_id}/certificate", json=request)
    assert certificate.status_code == 200
    integrity = client.get(f"/runs/{run_id}/certificate/verify")
    assert integrity.status_code == 200
    assert integrity.json()["valid"] is True
    assert integrity.json()["certificate_id"] == certificate.json()["certificate_id"]
    registry.close()
