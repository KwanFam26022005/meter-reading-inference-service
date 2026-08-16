"""API tests for /health endpoint."""

from fastapi.testclient import TestClient


def test_health_endpoint(test_app: TestClient) -> None:
    response = test_app.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "meter-reading-inference-service"
    assert data["api_version"] == "v1"
