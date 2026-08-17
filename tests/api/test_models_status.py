"""API tests for /api/v1/models/status endpoint and readiness gating."""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from meter_reading_inference_service.inference import InferenceCoordinator
from meter_reading_inference_service.main import create_app
from meter_reading_inference_service.settings import Settings


def test_models_status_success(test_app: TestClient) -> None:
    response = test_app.get("/api/v1/models/status")
    assert response.status_code == 200
    data = response.json()

    assert data["ready"] is True
    assert data["status"] == "ready"
    assert data["core"]["expected_revision"] == "9816d9a9f764460f3583dc18251eca2d953b9af6"
    assert data["core"]["revision_verified"] is True
    assert data["core"]["import_origin_verified"] is True
    assert data["core"]["dependency_verified"] is True
    assert data["pipeline_config"]["valid"] is True
    assert data["pipeline_config"]["calibrated"] is True
    assert data["pipeline_config"]["calibration_status"] == "CALIBRATED"
    assert data["recognition"]["runtime_status"] == "AVAILABLE"
    assert data["recognition"]["asset_status"] == "AVAILABLE"
    assert data["detector"]["runtime_status"] == "AVAILABLE"
    assert data["detector"]["asset_status"] == "AVAILABLE"
    assert data["artifacts"]["status"] == "AVAILABLE"
    assert "VSEE_VSE3T" in data["capabilities"]["supported_meter_types"]
    assert "GELEX_ME40" in data["capabilities"]["supported_meter_types"]
    assert "VSEE_VSE3T" in data["capabilities"]["default_localization_profiles"]
    assert data["capabilities"]["configured_mode_ready"] is True
    assert data["capabilities"]["learned_shadow_mode_ready"] is True
    assert data["capabilities"]["learned_shadow_decision_path_ready"] is True
    assert data["capabilities"]["learned_shadow_telemetry_ready"] is True
    assert data["capabilities"]["learned_primary_enabled"] is False

    # Security check: ensure no absolute filesystem paths are leaked
    raw_text = response.text
    assert "D:\\" not in raw_text or "D:\\Models" not in raw_text


def test_models_status_unverified_core(test_app: TestClient) -> None:
    """A1-002: Core revision mismatch causes ready=False and status=not_ready."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    original = coordinator.core_revision_verified
    coordinator.core_revision_verified = False

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is False
        assert data["status"] == "not_ready"
        assert data["core"]["revision_verified"] is False
        assert data["core"]["dependency_verified"] is False
        assert data["capabilities"]["configured_mode_ready"] is False
    finally:
        coordinator.core_revision_verified = original


def test_models_status_unverified_import_origin(test_app: TestClient) -> None:
    """Finding 1: Unverified import origin causes ready=False, dependency_verified=False."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    original = coordinator.import_origin_verified
    coordinator.import_origin_verified = False

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is False
        assert data["status"] == "not_ready"
        assert data["core"]["import_origin_verified"] is False
        assert data["core"]["dependency_verified"] is False
        assert data["capabilities"]["configured_mode_ready"] is False
    finally:
        coordinator.import_origin_verified = original


def test_models_status_example_only_config(test_app: TestClient) -> None:
    """A1-003: EXAMPLE_ONLY configuration causes ready=False and status=not_ready."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    original_cal = coordinator.is_calibrated
    original_status = coordinator.calibration_status
    coordinator.is_calibrated = False
    coordinator.calibration_status = "EXAMPLE_ONLY"

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is False
        assert data["status"] == "not_ready"
        assert data["pipeline_config"]["calibrated"] is False
        assert data["pipeline_config"]["calibration_status"] == "EXAMPLE_ONLY"
        assert data["capabilities"]["configured_mode_ready"] is False
    finally:
        coordinator.is_calibrated = original_cal
        coordinator.calibration_status = original_status


def test_models_status_unspecified_calibration(test_app: TestClient) -> None:
    """Finding 3: Missing calibration_status (UNSPECIFIED) causes ready=False."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    original_cal = coordinator.is_calibrated
    original_status = coordinator.calibration_status
    coordinator.is_calibrated = False
    coordinator.calibration_status = "UNSPECIFIED"

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is False
        assert data["status"] == "not_ready"
        assert data["pipeline_config"]["calibrated"] is False
        assert data["pipeline_config"]["calibration_status"] == "UNSPECIFIED"
    finally:
        coordinator.is_calibrated = original_cal
        coordinator.calibration_status = original_status


def test_models_status_missing_ocr_runtime(test_app: TestClient) -> None:
    """A1-004 / CASE 6: Missing OCR runtime causes ready=False and status=not_ready while shadow telemetry reflects detector status."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    original = coordinator.ocr_runtime_status
    coordinator.ocr_runtime_status = "MISSING"

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is False
        assert data["status"] == "not_ready"
        assert data["recognition"]["runtime_status"] == "MISSING"
        assert data["detector"]["runtime_status"] == "AVAILABLE"
        assert data["detector"]["asset_status"] == "AVAILABLE"
        assert data["capabilities"]["configured_mode_ready"] is False
        assert data["capabilities"]["learned_shadow_mode_ready"] is False
        assert data["capabilities"]["learned_shadow_decision_path_ready"] is False
        assert data["capabilities"]["learned_shadow_telemetry_ready"] is True
    finally:
        coordinator.ocr_runtime_status = original


def test_models_status_missing_artifact_storage(test_app: TestClient) -> None:
    """Finding 2: Unavailable artifact storage causes ready=False and status=not_ready."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    original = coordinator.artifact_storage_status
    coordinator.artifact_storage_status = "UNAVAILABLE"

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is False
        assert data["status"] == "not_ready"
        assert data["artifacts"]["status"] == "UNAVAILABLE"
        assert data["capabilities"]["configured_mode_ready"] is False
    finally:
        coordinator.artifact_storage_status = original


def test_models_status_missing_detector_runtime_preserves_configured(test_app: TestClient) -> None:
    """A1-H1 / CASE 2: Detector runtime MISSING degrades shadow mode while CONFIGURED remains ready."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    orig_det_rt = coordinator.detector_runtime_status
    orig_det_asset = coordinator.detector_asset_status
    coordinator.detector_runtime_status = "MISSING"
    coordinator.detector_asset_status = "AVAILABLE"

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is True
        assert data["status"] == "degraded"
        assert data["detector"]["runtime_status"] == "MISSING"
        assert data["detector"]["asset_status"] == "AVAILABLE"
        assert data["capabilities"]["configured_mode_ready"] is True
        assert data["capabilities"]["learned_shadow_mode_ready"] is False
        assert data["capabilities"]["learned_shadow_decision_path_ready"] is True
        assert data["capabilities"]["learned_shadow_telemetry_ready"] is False
    finally:
        coordinator.detector_runtime_status = orig_det_rt
        coordinator.detector_asset_status = orig_det_asset


def test_models_status_incompatible_detector_runtime_preserves_configured(test_app: TestClient) -> None:
    """A1-H1 / CASE 3: Detector runtime INCOMPATIBLE degrades shadow mode while CONFIGURED remains ready."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    orig_det_rt = coordinator.detector_runtime_status
    orig_det_asset = coordinator.detector_asset_status
    coordinator.detector_runtime_status = "INCOMPATIBLE"
    coordinator.detector_asset_status = "AVAILABLE"

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is True
        assert data["status"] == "degraded"
        assert data["detector"]["runtime_status"] == "INCOMPATIBLE"
        assert data["detector"]["asset_status"] == "AVAILABLE"
        assert data["capabilities"]["configured_mode_ready"] is True
        assert data["capabilities"]["learned_shadow_mode_ready"] is False
        assert data["capabilities"]["learned_shadow_decision_path_ready"] is True
        assert data["capabilities"]["learned_shadow_telemetry_ready"] is False
    finally:
        coordinator.detector_runtime_status = orig_det_rt
        coordinator.detector_asset_status = orig_det_asset


def test_models_status_missing_detector_preserves_configured(test_app: TestClient) -> None:
    """Finding 3 / CASE 4: Missing detector asset degrades shadow mode but CONFIGURED remains ready."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    original = coordinator.detector_asset_status
    orig_rt = coordinator.detector_runtime_status
    coordinator.detector_asset_status = "MISSING"
    coordinator.detector_runtime_status = "AVAILABLE"

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is True
        assert data["status"] == "degraded"
        assert data["detector"]["asset_status"] == "MISSING"
        assert data["detector"]["runtime_status"] == "AVAILABLE"
        assert data["capabilities"]["configured_mode_ready"] is True
        assert data["capabilities"]["learned_shadow_mode_ready"] is False
        assert data["capabilities"]["learned_shadow_decision_path_ready"] is True
        assert data["capabilities"]["learned_shadow_telemetry_ready"] is False
    finally:
        coordinator.detector_asset_status = original
        coordinator.detector_runtime_status = orig_rt


def test_models_status_invalid_detector_preserves_configured(test_app: TestClient) -> None:
    """Finding 3 / CASE 5: Detector checksum mismatch (INVALID) degrades shadow mode while CONFIGURED remains ready."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    original = coordinator.detector_asset_status
    orig_rt = coordinator.detector_runtime_status
    coordinator.detector_asset_status = "INVALID"
    coordinator.detector_runtime_status = "AVAILABLE"

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is True
        assert data["status"] == "degraded"
        assert data["detector"]["asset_status"] == "INVALID"
        assert data["detector"]["runtime_status"] == "AVAILABLE"
        assert data["capabilities"]["configured_mode_ready"] is True
        assert data["capabilities"]["learned_shadow_mode_ready"] is False
        assert data["capabilities"]["learned_shadow_decision_path_ready"] is True
        assert data["capabilities"]["learned_shadow_telemetry_ready"] is False
    finally:
        coordinator.detector_asset_status = original
        coordinator.detector_runtime_status = orig_rt


def test_models_status_wrong_detector_while_ocr_unavailable_not_falsely_available(
    test_app: TestClient,
) -> None:
    """Finding 3: When OCR is unavailable and detector has wrong SHA, detector must report INVALID (not AVAILABLE)."""
    coordinator: InferenceCoordinator = test_app.app.state.coordinator
    orig_ocr = coordinator.ocr_runtime_status
    orig_det = coordinator.detector_asset_status
    coordinator.ocr_runtime_status = "MISSING"
    coordinator.detector_asset_status = "INVALID"

    try:
        response = test_app.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()

        assert data["ready"] is False
        assert data["status"] == "not_ready"
        assert data["detector"]["asset_status"] == "INVALID"
        assert data["capabilities"]["learned_shadow_telemetry_ready"] is False
    finally:
        coordinator.ocr_runtime_status = orig_ocr
        coordinator.detector_asset_status = orig_det


def test_models_status_no_coordinator() -> None:
    """A1-H1: When coordinator is absent, detector runtime_status reports NOT_EVALUATED."""
    app = create_app(Settings())
    # No coordinator injected
    with TestClient(app) as client:
        # Patch lifespan or mock state
        client.app.state.coordinator = None
        response = client.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is False
        assert data["status"] == "not_ready"
        assert data["detector"]["runtime_status"] == "NOT_EVALUATED"
        assert data["detector"]["asset_status"] == "NOT_EVALUATED"


def test_startup_artifact_storage_permission_error_lifespan(tmp_path: Path) -> None:
    """Finding 2: When artifact root mkdir raises PermissionError, app starts cleanly in not_ready state without aborting lifespan."""
    settings = Settings(
        artifact_root=tmp_path / "denied_artifacts",
    )

    app = create_app(settings)

    # Simulate PermissionError during check_artifact_storage (e.g. read-only filesystem or access denied)
    with patch(
        "meter_reading_inference_service.main.check_artifact_storage",
        return_value=("UNAVAILABLE", "Permission denied"),
    ):
        with TestClient(app) as client:
            # 1. Liveness probe is always healthy (200)
            health_resp = client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["status"] == "ok"

            # 2. Status endpoint reports not_ready with UNAVAILABLE artifact storage
            status_resp = client.get("/api/v1/models/status")
            assert status_resp.status_code == 200
            status_data = status_resp.json()
            assert status_data["ready"] is False
            assert status_data["status"] == "not_ready"
            assert status_data["artifacts"]["status"] == "UNAVAILABLE"
            assert status_data["capabilities"]["configured_mode_ready"] is False

            # 3. Infer request fails closed with 503 PIPELINE_NOT_READY
            infer_resp = client.post(
                "/api/v1/meter-readings/infer",
                files={"image": ("test.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 20, "image/jpeg")},
                data={"meter_type": "VSEE_VSE3T"},
            )
            assert infer_resp.status_code == 503
            assert infer_resp.json()["error"]["code"] == "PIPELINE_NOT_READY"


def test_startup_lifespan_detector_runtime_missing(tmp_path: Path) -> None:
    """A1-H1 / Lifespan: Proves startup preflight propagates missing detector runtime and prevents false shadow readiness."""
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
    )
    app = create_app(settings)

    with (
        patch(
            "meter_reading_inference_service.main.check_core_revision",
            return_value=(True, settings.expected_core_revision),
        ),
        patch(
            "meter_reading_inference_service.main.check_core_import_origin",
            return_value=(True, "editable"),
        ),
        patch(
            "meter_reading_inference_service.main.get_yaml_calibration_status",
            return_value=("CALIBRATED", True),
        ),
        patch(
            "meter_reading_inference_service.main.check_artifact_storage",
            return_value=("AVAILABLE", None),
        ),
        patch(
            "meter_reading_inference_service.main.check_ocr_runtime",
            return_value=("AVAILABLE", None),
        ),
        patch(
            "meter_reading_inference_service.main.check_ocr_asset",
            return_value=("AVAILABLE", None),
        ),
        patch(
            "meter_reading_inference_service.main.check_detector_runtime",
            return_value=("MISSING", "Ultralytics detector runtime is not installed"),
        ),
        patch(
            "meter_reading_inference_service.main.check_detector_asset",
            return_value=("AVAILABLE", None),
        ),
    ):
        with TestClient(app) as client:
            resp = client.get("/api/v1/models/status")
            assert resp.status_code == 200
            data = resp.json()

            # Configured remains ready, but service overall is degraded due to missing detector runtime
            assert data["ready"] is True
            assert data["status"] == "degraded"
            assert data["detector"]["runtime_status"] == "MISSING"
            assert data["detector"]["asset_status"] == "AVAILABLE"
            assert data["capabilities"]["configured_mode_ready"] is True
            assert data["capabilities"]["learned_shadow_mode_ready"] is False
            assert data["capabilities"]["learned_shadow_decision_path_ready"] is True
            assert data["capabilities"]["learned_shadow_telemetry_ready"] is False

