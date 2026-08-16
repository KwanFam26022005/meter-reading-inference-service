"""API tests for global error handling and concurrency limits."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from meter_reading_inference_service.main import create_app
from meter_reading_inference_service.settings import Settings


def test_pipeline_not_ready_returns_503(synthetic_valid_image_bytes: bytes) -> None:
    settings = Settings()
    app = create_app(settings)

    with TestClient(app) as client:
        app.state.coordinator = None  # Simulate uninitialized / failing pipeline
        # Status endpoint reports not_ready safely without 503
        status_res = client.get("/api/v1/models/status")
        assert status_res.status_code == 200
        assert status_res.json()["ready"] is False
        assert status_res.json()["status"] == "not_ready"

        # Inference endpoint fails closed with 503
        files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
        data = {"meter_type": "VSEE_VSE3T"}
        infer_res = client.post("/api/v1/meter-readings/infer", files=files, data=data)
        assert infer_res.status_code == 503
        assert infer_res.json()["error"]["code"] == "PIPELINE_NOT_READY"


def test_internal_server_error_envelope(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
) -> None:
    # Cause an unexpected exception in pipeline
    mock_pipeline.run.side_effect = RuntimeError("Unexpected internal crash")

    files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
    data = {"meter_type": "VSEE_VSE3T"}

    client = TestClient(test_app.app, raise_server_exceptions=False)
    response = client.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 500
    err = response.json()["error"]
    assert err["code"] == "INTERNAL_SERVICE_ERROR"
    # Ensure traceback string is not returned to client
    assert "Unexpected internal crash" not in err["message"]
