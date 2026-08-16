"""Test helpers for generating synthetic test pipeline results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from meter_reading_engine.contracts import (
    AcceptanceCheck,
    ArtifactIndex,
    ArtifactRef,
    ArtifactRole,
    CoordinateSpace,
    CoordinateUnit,
    DecisionOutcome,
    DecisionResult,
    GeometryInstruction,
    GeometryMode,
    GeometryResult,
    GeometrySource,
    LearnedLocalizationProvenance,
    LocalizationMethod,
    LocalizationResult,
    MeterType,
    ModelProvenance,
    NormalizationOperation,
    NormalizationResult,
    NormalizedBBox,
    PipelineResult,
    PixelBBox,
    QualityMetric,
    QualityReport,
    RecognitionResult,
    RecognitionSpan,
    RegionRef,
    RegionRole,
    RunContext,
    SourceIdentity,
    StageName,
    StageRecord,
    StageStatus,
    TransformKind,
    TransformRecord,
    ValidationCheck,
    ValidationReport,
)


def create_sample_pipeline_result(
    outcome: DecisionOutcome = DecisionOutcome.ACCEPT,
    run_id: str = "run_test_12345",
    request_id: str = "req_test_123",
    meter_type: MeterType = MeterType.VSEE_VSE3T,
    raw_ocr: str = "00123456",
    confidence: float = 0.95,
    artifact_root: Path | None = None,
    shadow_provenance: LearnedLocalizationProvenance | None = None,
) -> PipelineResult:
    """Helper to construct a realistic synthetic PipelineResult for testing."""
    now = datetime.now(timezone.utc)
    source_space = CoordinateSpace(id="source", width=640, height=480)
    working_space = CoordinateSpace(id="working_display", width=512, height=288, parent_id="source")

    # Create dummy artifact files if artifact_root provided
    art_index = None
    if artifact_root:
        run_rel = f"runs/{run_id}"
        run_dir = artifact_root / run_rel
        run_dir.mkdir(parents=True, exist_ok=True)

        for role_name in ("source.jpg", "display.jpg", "working_display.jpg", "reading_value.jpg"):
            f = run_dir / role_name
            f.write_bytes(b"FAKE_IMAGE_DATA")

        art_index = ArtifactIndex(
            store_id="file_store",
            run_relative_root=run_rel,
            artifacts=(
                ArtifactRef(
                    role=ArtifactRole.SOURCE,
                    relative_path="source.jpg",
                    media_type="image/jpeg",
                    sha256="abc123",
                    byte_length=15,
                ),
                ArtifactRef(
                    role=ArtifactRole.DISPLAY,
                    relative_path="display.jpg",
                    media_type="image/jpeg",
                    sha256="def456",
                    byte_length=15,
                ),
                ArtifactRef(
                    role=ArtifactRole.WORKING_DISPLAY,
                    relative_path="working_display.jpg",
                    media_type="image/jpeg",
                    sha256="ghi789",
                    byte_length=15,
                ),
                ArtifactRef(
                    role=ArtifactRole.READING_VALUE,
                    relative_path="reading_value.jpg",
                    media_type="image/jpeg",
                    sha256="jkl012",
                    byte_length=15,
                ),
            ),
        )

    # Stages
    stages = (
        StageRecord(
            stage=StageName.INPUT,
            status=StageStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            duration_ns=1_000_000,
            implementation_id="input_decoder_v1",
        ),
        StageRecord(
            stage=StageName.QUALITY,
            status=StageStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            duration_ns=2_000_000,
            implementation_id="standard_quality_v1",
        ),
        StageRecord(
            stage=StageName.DISPLAY_LOCALIZATION,
            status=StageStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            duration_ns=1_500_000,
            implementation_id="configured_display_v1",
        ),
        StageRecord(
            stage=StageName.GEOMETRY,
            status=StageStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            duration_ns=1_000_000,
            implementation_id="noop_geom_v1",
        ),
        StageRecord(
            stage=StageName.VALUE_LOCALIZATION,
            status=StageStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            duration_ns=2_500_000,
            implementation_id="configured_reading_v1",
        ),
        StageRecord(
            stage=StageName.RECOGNITION,
            status=StageStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            duration_ns=10_000_000,
            implementation_id="paddleocr_v6_reader_v1",
        ),
        StageRecord(
            stage=StageName.NORMALIZATION,
            status=StageStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            duration_ns=500_000,
            implementation_id="identity_normalizer_v1",
        ),
        StageRecord(
            stage=StageName.VALIDATION,
            status=StageStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            duration_ns=500_000,
            implementation_id="lcd_validator_v1",
        ),
        StageRecord(
            stage=StageName.PERSISTENCE,
            status=StageStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            duration_ns=1_000_000,
            implementation_id="file_persistence_v1",
        ),
    )

    quality = QualityReport(
        metrics=(
            QualityMetric(
                name="min_width",
                value=640,
                unit="pixels",
                threshold=100,
                check_version="1.0",
            ),
            QualityMetric(
                name="min_height",
                value=480,
                unit="pixels",
                threshold=100,
                check_version="1.0",
            ),
            QualityMetric(
                name="dark_pixel_fraction",
                value=0.10,
                unit="ratio",
                threshold=0.90,
                check_version="1.0",
            ),
            QualityMetric(
                name="bright_pixel_fraction",
                value=0.05,
                unit="ratio",
                threshold=0.90,
                check_version="1.0",
            ),
            QualityMetric(
                name="laplacian_variance",
                value=120.5,
                unit="variance",
                threshold=10.0,
                check_version="1.0",
            ),
        ),
        findings=(),
    )

    display_loc = LocalizationResult(
        role=RegionRole.DISPLAY,
        region=RegionRef(
            region_id="disp_1",
            role=RegionRole.DISPLAY,
            coordinate_space_id="source",
            supplied_unit=CoordinateUnit.NORMALIZED,
            supplied_bbox=NormalizedBBox(x=0.10, y=0.15, width=0.80, height=0.60),
            resolved_pixel_bbox=PixelBBox(x=64, y=72, width=512, height=288),
        ),
        method=LocalizationMethod.CONFIGURED,
        implementation_id="configured_display_v1",
        profile_id="vsee_vse3t_station_a_v1",
        profile_revision="1.0",
        confidence=1.0,
    )

    geom = GeometryResult(
        instruction=GeometryInstruction(
            mode=GeometryMode.NO_OP,
            source=GeometrySource.DEFAULT,
            input_space_id="display_crop",
        ),
        output_space=working_space,
        resolved_input_quad=None,
        transforms=(
            TransformRecord(
                transform_id="crop_1",
                kind=TransformKind.CROP_TRANSLATION,
                input_space_id="source",
                output_space_id="working_display",
                matrix_3x3_input_to_output=((1.0, 0.0, -64.0), (0.0, 1.0, -72.0), (0.0, 0.0, 1.0)),
                inverse_matrix_3x3=((1.0, 0.0, 64.0), (0.0, 1.0, 72.0), (0.0, 0.0, 1.0)),
                input_shape=(480, 640),
                output_shape=(288, 512),
            ),
        ),
        findings=(),
        implementation_id="noop_geom_v1",
    )

    val_meta = {}
    if shadow_provenance:
        val_meta["learned_localization"] = shadow_provenance

    val_loc = LocalizationResult(
        role=RegionRole.READING_VALUE,
        region=RegionRef(
            region_id="val_1",
            role=RegionRole.READING_VALUE,
            coordinate_space_id="working_display",
            supplied_unit=CoordinateUnit.NORMALIZED,
            supplied_bbox=NormalizedBBox(x=0.05, y=0.10, width=0.90, height=0.80),
            resolved_pixel_bbox=PixelBBox(x=25, y=28, width=460, height=230),
        ),
        method=LocalizationMethod.CONFIGURED,
        implementation_id="configured_reading_v1",
        profile_id="vsee_vse3t_station_a_v1",
        profile_revision="1.0",
        confidence=1.0,
        metadata=val_meta,
    )

    rec = RecognitionResult(
        raw_text=raw_ocr,
        confidence=confidence,
        spans=(RecognitionSpan(text=raw_ocr, confidence=confidence),),
        model=ModelProvenance(
            reader_id="paddleocr_v6_reader_v1",
            logical_model_id="PP-OCRv6-medium",
            resolved_model_version="1.0",
            model_sha256="fake_sha",
            paddleocr_version="3.7.0",
            paddle_version="3.3.1",
            device="cpu",
            inference_parameters={},
        ),
        findings=(),
        started_at=now,
        finished_at=now,
        duration_ns=10_000_000,
    )

    norm = NormalizationResult(
        raw_text=raw_ocr,
        normalized_text=raw_ocr,
        normalizer_id="identity_normalizer_v1",
        normalizer_version="1.0",
        operations=(NormalizationOperation(operation="identity"),),
        findings=(),
    )

    val_report = ValidationReport(
        is_valid=(outcome == DecisionOutcome.ACCEPT),
        acceptance_profile_complete=True,
        profile_id="vsee_vse3t_val_v1",
        profile_revision="1.0",
        normalized_text=raw_ocr,
        parsed_value=raw_ocr,
        min_ocr_confidence_for_accept=0.85,
        checks=(
            ValidationCheck(
                rule_id="full_pattern_match",
                passed=(outcome == DecisionOutcome.ACCEPT),
                observed=raw_ocr,
                expected="^[0-9]{6,8}(\\.[0-9]{1,2})?$",
            ),
        ),
        findings=(),
    )

    decision = DecisionResult(
        outcome=outcome,
        reason_codes=() if outcome == DecisionOutcome.ACCEPT else ("VALIDATION_FORMAT_MISMATCH",),
        policy_id="decision_v1",
        policy_version="1.0",
        decided_at=now,
        acceptance_checks=(
            AcceptanceCheck(
                check_id="validation_complete",
                passed=(outcome == DecisionOutcome.ACCEPT),
                reason_codes=()
                if outcome == DecisionOutcome.ACCEPT
                else ("VALIDATION_FORMAT_MISMATCH",),
            ),
        ),
    )

    return PipelineResult(
        schema_version="1.0",
        run=RunContext(
            run_id=run_id,
            request_id=request_id,
            started_at=now,
            meter_type=meter_type,
            config_revision="1.0.0",
            config_sha256="fake_config_sha256",
        ),
        source=SourceIdentity(
            sha256="fake_source_sha256",
            byte_length=len(b"test"),
            media_type="image/jpeg",
            encoded_width=640,
            encoded_height=480,
            exif_orientation=1,
            orientation_operation="none",
            decoder_id="pil",
            decoder_version="10.0.0",
            width=640,
            height=480,
            channels=3,
        ),
        coordinate_spaces=(source_space, working_space),
        stages=stages,
        quality=quality,
        display_localization=display_loc,
        geometry=geom,
        value_localization=val_loc,
        recognition=rec,
        normalization=norm,
        validation=val_report,
        decision=decision,
        artifacts=art_index,
        finished_at=now,
        duration_ns=20_000_000,
        transforms=geom.transforms,
    )
