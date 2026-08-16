"""Unit tests for inference result mapping and sanitization."""

from meter_reading_engine.contracts import (
    Finding,
    FindingCategory,
    FindingSeverity,
    LearnedLocalizationProvenance,
    LocatorMode,
    PixelBBox,
    StageName,
    StageStatus,
)

from meter_reading_inference_service.inference import (
    map_stage_status,
    sanitize_finding,
)


def test_map_stage_status() -> None:
    assert map_stage_status(StageStatus.SUCCEEDED) == "PASS"
    assert map_stage_status(StageStatus.FAILED) == "FAIL"
    assert map_stage_status(StageStatus.NOT_RUN) == "NOT_RUN"


def test_sanitize_finding_removes_paths() -> None:
    finding = Finding(
        code="TEST_ERROR",
        category=FindingCategory.SYSTEM,
        severity=FindingSeverity.BLOCKING,
        stage=StageName.RECOGNITION,
        message=r"Error at D:\Models\secret\model.onnx failed to load",
        details={
            "model_path": r"D:\Models\secret\model.onnx",
            "safe_param": 42,
            "raw_exception": "Exception...",
        },
    )

    sanitized = sanitize_finding(finding)
    assert sanitized.code == "TEST_ERROR"
    assert "D:\\Models" not in sanitized.message
    assert "<path_omitted>" in sanitized.message
    assert "model_path" not in sanitized.details
    assert "raw_exception" not in sanitized.details
    assert sanitized.details["safe_param"] == 42


def test_learned_shadow_provenance_dataclass() -> None:
    prov = LearnedLocalizationProvenance(
        model_id="yuva_test",
        checkpoint_sha256="abc",
        input_space="working_display",
        conf=0.25,
        iou=0.45,
        max_det=10,
        proposal_count=3,
        selected_proposal_index=0,
        selected_detector_confidence=0.88,
        selected_bbox=PixelBBox(x=10, y=10, width=100, height=50),
        requested_locator=LocatorMode.LEARNED_SHADOW,
        effective_locator=LocatorMode.CONFIGURED,
        shadow_status="ACTIVE",
    )
    assert prov.shadow_status == "ACTIVE"
    assert prov.selected_detector_confidence == 0.88
