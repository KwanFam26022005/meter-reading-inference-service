"""API tests for /api/v1/meter-readings/infer endpoint."""

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from meter_reading_engine.contracts import (
    DecisionOutcome,
    LearnedLocalizationProvenance,
    LocatorMode,
    MeterType,
    PixelBBox,
)

from tests.conftest import create_sample_pipeline_result


def test_infer_success_configured_mode(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
    temp_artifact_dir: Path,
) -> None:
    mock_pipeline.run.side_effect = lambda input_data: create_sample_pipeline_result(
        outcome=DecisionOutcome.ACCEPT,
        run_id="run_accept_1",
        request_id=input_data.request_id,
        meter_type=input_data.meter_type,
        raw_ocr="00123456",
        confidence=0.98,
        artifact_root=temp_artifact_dir,
    )

    files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
    data = {
        "meter_type": "VSEE_VSE3T",
        "request_id": "req_unit_1",
        "locator_mode": "CONFIGURED",
    }

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    # Schema top-level
    assert res["schema_version"] == "a1.inference.v1"
    assert res["request_id"] == "req_unit_1"
    assert res["meter_type"] == "VSEE_VSE3T"

    # Summary
    assert res["summary"]["decision"]["outcome"] == "ACCEPT"
    assert res["summary"]["reading"]["raw_text"] == "00123456"
    assert res["summary"]["reading"]["is_valid"] is True

    # Trace
    assert len(res["trace"]["stages"]) == 9
    assert res["trace"]["stages"][0]["status"] == "PASS"
    assert res["trace"]["quality"] is not None
    assert len(res["trace"]["quality"]["metrics"]) > 0

    # Visualization
    assert res["visualization"]["source"]["coordinate_space_id"] == "source"
    assert res["visualization"]["display"]["region"] is not None
    assert res["visualization"]["working_display"]["transforms"] is not None
    assert res["visualization"]["reading_value"]["used_for_ocr"] is True

    # Check PipelineInput passed to Core
    args, _ = mock_pipeline.run.call_args
    passed_input = args[0]
    assert passed_input.image_bytes == synthetic_valid_image_bytes
    assert passed_input.meter_type == MeterType.VSEE_VSE3T
    assert passed_input.localization_profile_id == "vsee_vse3t_station_a_v1"


def test_infer_png_image_support(
    test_app: TestClient,
    synthetic_png_image_bytes: bytes,
    mock_pipeline: MagicMock,
    temp_artifact_dir: Path,
) -> None:
    files = {"image": ("test.png", synthetic_png_image_bytes, "image/png")}
    data = {"meter_type": "GELEX_ME40"}

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["meter_type"] == "GELEX_ME40"


def test_infer_learned_shadow_mode(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
    temp_artifact_dir: Path,
) -> None:
    shadow_prov = LearnedLocalizationProvenance(
        model_id="yuva_yolo",
        checkpoint_sha256="81C8DAAF2D68E7C87E6241A923B3E9724F0E5B5130AF70E6DD6CF112751E093C",
        input_space="working_display",
        conf=0.01,
        iou=0.50,
        max_det=20,
        proposal_count=5,
        selected_proposal_index=0,
        selected_detector_confidence=0.91,
        selected_bbox=PixelBBox(x=30, y=35, width=450, height=220),
        requested_locator=LocatorMode.LEARNED_SHADOW,
        effective_locator=LocatorMode.CONFIGURED,
        shadow_status="ACTIVE",
    )

    mock_pipeline.run.side_effect = lambda input_data: create_sample_pipeline_result(
        outcome=DecisionOutcome.ACCEPT,
        run_id="run_shadow_1",
        request_id=input_data.request_id,
        meter_type=input_data.meter_type,
        artifact_root=temp_artifact_dir,
        shadow_provenance=shadow_prov,
    )

    files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
    data = {
        "meter_type": "VSEE_VSE3T",
        "locator_mode": "LEARNED_SHADOW",
    }

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    shadow_vis = res["visualization"]["learned_shadow"]
    assert shadow_vis is not None
    assert shadow_vis["shadow_status"] == "ACTIVE"
    assert shadow_vis["configured"]["used_for_ocr"] is True
    assert shadow_vis["learned"]["used_for_ocr"] is False
    assert shadow_vis["learned"]["detector_confidence"] == 0.91
    assert shadow_vis["provenance"]["model_id"] == "yuva_yolo"


def test_infer_learned_primary_disabled_returns_403(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
) -> None:
    files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
    data = {
        "meter_type": "VSEE_VSE3T",
        "locator_mode": "LEARNED_PRIMARY",
    }

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 403
    err = response.json()["error"]
    assert err["code"] == "LEARNED_PRIMARY_DISABLED"


def test_infer_unsupported_media_type(test_app: TestClient) -> None:
    files = {"image": ("test.txt", b"plain text data", "text/plain")}
    data = {"meter_type": "VSEE_VSE3T"}

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_infer_corrupt_image_returns_400(test_app: TestClient) -> None:
    files = {"image": ("test.jpg", b"NOT_AN_IMAGE_HEADER_BYTES", "image/jpeg")}
    data = {"meter_type": "VSEE_VSE3T"}

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_infer_oversized_upload_returns_413(test_app: TestClient) -> None:
    # 2MB payload exceeds test limit of 1MB
    large_bytes = b"\xff\xd8\xff" + (b"\x00" * (2 * 1024 * 1024))
    files = {"image": ("test.jpg", large_bytes, "image/jpeg")}
    data = {"meter_type": "VSEE_VSE3T"}

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"


def test_infer_all_domain_outcomes_return_200(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
    temp_artifact_dir: Path,
) -> None:
    for outcome in (
        DecisionOutcome.ACCEPT,
        DecisionOutcome.REVIEW,
        DecisionOutcome.RECAPTURE,
        DecisionOutcome.ERROR,
    ):
        mock_pipeline.run.side_effect = lambda input_data, out=outcome: (
            create_sample_pipeline_result(
                outcome=out,
                run_id="run_" + out.value.lower(),
                request_id=input_data.request_id,
                meter_type=input_data.meter_type,
                artifact_root=temp_artifact_dir,
            )
        )

        files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
        data = {"meter_type": "VSEE_VSE3T"}

        response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
        assert response.status_code == 200
        assert response.json()["summary"]["decision"]["outcome"] == outcome.value


def test_infer_mismatched_declared_jpeg_actual_png_returns_415(
    test_app: TestClient,
    synthetic_png_image_bytes: bytes,
) -> None:
    """A1-006: PNG payload sent with image/jpeg header is rejected with 415."""
    files = {"image": ("test.jpg", synthetic_png_image_bytes, "image/jpeg")}
    data = {"meter_type": "VSEE_VSE3T"}

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_infer_mismatched_declared_png_actual_jpeg_returns_415(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
) -> None:
    """A1-006: JPEG payload sent with image/png header is rejected with 415."""
    files = {"image": ("test.png", synthetic_valid_image_bytes, "image/png")}
    data = {"meter_type": "VSEE_VSE3T"}

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_infer_exact_source_bytes_preserved(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
) -> None:
    """A1-006: Uploaded encoded byte sequence is passed untouched to Core."""
    files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
    data = {"meter_type": "VSEE_VSE3T"}

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 200

    args, _ = mock_pipeline.run.call_args
    passed_input = args[0]
    assert passed_input.image_bytes == synthetic_valid_image_bytes


def test_infer_unverified_core_returns_503(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
) -> None:
    """A1-002: Unverified Core causes 503 PIPELINE_NOT_READY without invoking pipeline.run."""
    mock_pipeline.run.reset_mock()
    test_app.app.state.coordinator.core_revision_verified = False

    files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
    data = {"meter_type": "VSEE_VSE3T"}

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "PIPELINE_NOT_READY"
    mock_pipeline.run.assert_not_called()


def test_infer_example_only_config_returns_503(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
) -> None:
    """A1-003: EXAMPLE_ONLY configuration causes 503 without invoking pipeline.run."""
    mock_pipeline.run.reset_mock()
    test_app.app.state.coordinator.is_calibrated = False
    test_app.app.state.coordinator.calibration_status = "EXAMPLE_ONLY"

    files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
    data = {"meter_type": "VSEE_VSE3T"}

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "PIPELINE_NOT_READY"
    mock_pipeline.run.assert_not_called()


def test_infer_missing_ocr_runtime_returns_503(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
) -> None:
    """A1-004: Missing OCR runtime causes 503 without invoking pipeline.run."""
    mock_pipeline.run.reset_mock()
    original = test_app.app.state.coordinator.ocr_runtime_status
    test_app.app.state.coordinator.ocr_runtime_status = "MISSING"

    try:
        files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
        data = {"meter_type": "VSEE_VSE3T"}

        response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "PIPELINE_NOT_READY"
        mock_pipeline.run.assert_not_called()
    finally:
        test_app.app.state.coordinator.ocr_runtime_status = original


def test_infer_oversized_content_length_header_returns_413(
    test_app: TestClient,
) -> None:
    """A1-006: Request with Content-Length exceeding limit is rejected early with 413."""
    headers = {
        "content-type": "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        "content-length": "20000000",  # 20MB exceeds 1MB limit
    }
    response = test_app.post(
        "/api/v1/meter-readings/infer",
        headers=headers,
        content=b"dummy",
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"


def test_infer_unverified_import_origin_returns_503(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
) -> None:
    """NEW: Unverified Core import origin causes 503 without invoking pipeline.run."""
    mock_pipeline.run.reset_mock()
    original = test_app.app.state.coordinator.import_origin_verified
    test_app.app.state.coordinator.import_origin_verified = False

    try:
        files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
        data = {"meter_type": "VSEE_VSE3T"}

        response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "PIPELINE_NOT_READY"
        mock_pipeline.run.assert_not_called()
    finally:
        test_app.app.state.coordinator.import_origin_verified = original


def test_infer_unavailable_artifact_storage_returns_503(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
) -> None:
    """NEW: Unavailable artifact storage causes 503 without invoking pipeline.run."""
    mock_pipeline.run.reset_mock()
    original = test_app.app.state.coordinator.artifact_storage_status
    test_app.app.state.coordinator.artifact_storage_status = "UNAVAILABLE"

    try:
        files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
        data = {"meter_type": "VSEE_VSE3T"}

        response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "PIPELINE_NOT_READY"
        mock_pipeline.run.assert_not_called()
    finally:
        test_app.app.state.coordinator.artifact_storage_status = original
