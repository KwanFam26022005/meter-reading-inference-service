"""API tests for AUTO locator_mode and reference comparison semantics."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from meter_reading_engine.contracts import (
    DecisionOutcome,
    LocalizationMethod,
)

import meter_reading_inference_service.inference as inference_module
from tests.conftest import create_sample_pipeline_result


def test_models_status_reports_auto_mode_ready(test_app: TestClient) -> None:
    response = test_app.get("/api/v1/models/status")
    assert response.status_code == 200
    res = response.json()
    assert "auto_mode_ready" in res["capabilities"]
    assert isinstance(res["capabilities"]["auto_mode_ready"], bool)


def test_infer_auto_mode_with_reference(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
    temp_artifact_dir: Path,
) -> None:
    # Set up mock pipeline result with AUTO value localization
    mock_pipeline.run.side_effect = lambda input_data: create_sample_pipeline_result(
        outcome=DecisionOutcome.ACCEPT,
        run_id="run_auto_1",
        request_id=input_data.request_id,
        meter_type=input_data.meter_type,
        raw_ocr="11532580",
        confidence=0.95,
        artifact_root=temp_artifact_dir,
    )

    files = {"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")}
    data = {
        "meter_type": "VSEE_VSE3T",
        "request_id": "req_auto_1",
        "locator_mode": "AUTO",
    }

    response = test_app.post("/api/v1/meter-readings/infer", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["request_id"] == "req_auto_1"
    assert res["provenance"]["requested_locator_mode"] == "AUTO"
    assert res["visualization"]["auto_localization"] is not None
    assert res["visualization"]["auto_localization"]["used_for_ocr"] is True
    assert res["visualization"]["auto_localization"]["auto_semantic"] == "ACTIVE_NUMERIC_SEQUENCE"
    assert res["visualization"]["auto_localization"]["comparison_semantics_match"] is None


def test_reference_panel_is_not_presented_as_auto_accuracy_metric(
    test_app: TestClient,
    synthetic_valid_image_bytes: bytes,
    mock_pipeline: MagicMock,
    temp_artifact_dir: Path,
    monkeypatch,
) -> None:
    registry_path = temp_artifact_dir / "reference_rois.json"
    registry_path.write_text(
        '{"test_reference_sha": {"sample_id": "test", "meter_type": "VSEE_VSE3T", '
        '"reference_type": "CONFIGURED", "coordinate_space": "source", '
        '"bbox": {"x": 1, "y": 2, "width": 3, "height": 4}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(inference_module, "Path", lambda _value: registry_path)

    def build_auto_result(input_data):
        result = create_sample_pipeline_result(
            outcome=DecisionOutcome.ACCEPT,
            run_id="run_auto_reference",
            request_id=input_data.request_id,
            meter_type=input_data.meter_type,
            artifact_root=temp_artifact_dir,
        )
        assert result.value_localization is not None
        auto_localization = replace(
            result.value_localization,
            method=LocalizationMethod.LEARNED,
            confidence=0.91,
            metadata={"strategy": "AUTO_MULTI_SCALE_YOLO_V1", "used_for_ocr": True},
        )
        return replace(
            result,
            source=replace(result.source, sha256="test_reference_sha"),
            value_localization=auto_localization,
        )

    mock_pipeline.run.side_effect = build_auto_result
    response = test_app.post(
        "/api/v1/meter-readings/infer",
        files={"image": ("test.jpg", synthetic_valid_image_bytes, "image/jpeg")},
        data={"meter_type": "VSEE_VSE3T", "request_id": "req_auto_reference", "locator_mode": "AUTO"},
    )

    assert response.status_code == 200
    auto = response.json()["visualization"]["auto_localization"]
    assert auto["reference_semantic"] == "DISPLAY_PANEL"
    assert auto["auto_semantic"] == "ACTIVE_NUMERIC_SEQUENCE"
    assert auto["comparison_semantics_match"] is False
    assert "IoU is not a correctness metric" in auto["comparison_message"]
