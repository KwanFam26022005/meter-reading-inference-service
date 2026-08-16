"""Shared pytest fixtures for unit and API test suites."""

from __future__ import annotations

import io
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from meter_reading_engine.contracts import (
    ArtifactPolicy,
    CoordinateUnit,
    DecimalPolicy,
    DecisionOutcome,
    DecisionPolicy,
    DetectorPolicy,
    GeometryInstruction,
    GeometryMode,
    GeometrySource,
    LocalizationProfile,
    MeterType,
    NormalizedBBox,
    PipelineConfig,
    QualityPolicy,
    ReadingProfile,
    RecognitionPolicy,
    RegionSpec,
    ValidationProfile,
    freeze_config,
)
from PIL import Image

from meter_reading_inference_service.artifacts import InMemoryArtifactRegistry
from meter_reading_inference_service.inference import InferenceCoordinator
from meter_reading_inference_service.main import create_app
from meter_reading_inference_service.settings import Settings
from tests.helpers import create_sample_pipeline_result


@pytest.fixture
def temp_artifact_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary artifact root directory."""
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def synthetic_valid_image_bytes() -> bytes:
    """Generates a valid test JPEG image."""
    arr = np.ones((240, 320, 3), dtype=np.uint8) * 180
    # Draw simple patterns
    arr[50:190, 50:270] = 50
    arr[70:170, 70:250] = 220
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def synthetic_png_image_bytes() -> bytes:
    """Generates a valid test PNG image."""
    arr = np.ones((240, 320, 3), dtype=np.uint8) * 150
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_pipeline_config(temp_artifact_dir: Path) -> PipelineConfig:
    """Provides a valid PipelineConfig for testing."""
    vsee_loc = LocalizationProfile(
        profile_id="vsee_vse3t_station_a_v1",
        revision="1.0",
        meter_type=MeterType.VSEE_VSE3T,
        capture_context="station_a",
        reference_source_width=640,
        reference_source_height=480,
        aspect_ratio_tolerance=0.10,
        display_roi=RegionSpec(
            coordinate_space_id="source",
            unit=CoordinateUnit.NORMALIZED,
            bbox=NormalizedBBox(x=0.10, y=0.15, width=0.80, height=0.60),
        ),
        geometry=GeometryInstruction(
            mode=GeometryMode.NO_OP,
            source=GeometrySource.DEFAULT,
            input_space_id="display_crop",
        ),
        reading_value_roi=RegionSpec(
            coordinate_space_id="working_display",
            unit=CoordinateUnit.NORMALIZED,
            bbox=NormalizedBBox(x=0.05, y=0.10, width=0.90, height=0.80),
        ),
        acceptance_complete=True,
    )

    gelex_loc = LocalizationProfile(
        profile_id="gelex_me40_station_b_v1",
        revision="1.0",
        meter_type=MeterType.GELEX_ME40,
        capture_context="station_b",
        reference_source_width=640,
        reference_source_height=480,
        aspect_ratio_tolerance=0.10,
        display_roi=RegionSpec(
            coordinate_space_id="source",
            unit=CoordinateUnit.NORMALIZED,
            bbox=NormalizedBBox(x=0.10, y=0.15, width=0.80, height=0.60),
        ),
        geometry=GeometryInstruction(
            mode=GeometryMode.NO_OP,
            source=GeometrySource.DEFAULT,
            input_space_id="display_crop",
        ),
        reading_value_roi=RegionSpec(
            coordinate_space_id="working_display",
            unit=CoordinateUnit.NORMALIZED,
            bbox=NormalizedBBox(x=0.05, y=0.10, width=0.90, height=0.80),
        ),
        acceptance_complete=True,
    )

    vsee_val = ValidationProfile(
        profile_id="vsee_vse3t_val_v1",
        revision="1.0",
        meter_type=MeterType.VSEE_VSE3T,
        acceptance_complete=True,
        full_match_pattern=r"^[0-9]{6,8}(\.[0-9]{1,2})?$",
        decimal_policy=DecimalPolicy.OPTIONAL,
        integer_digits_min=6,
        integer_digits_max=8,
        fractional_digits_min=0,
        fractional_digits_max=2,
        min_ocr_confidence_for_accept=0.85,
    )

    gelex_val = ValidationProfile(
        profile_id="gelex_me40_val_v1",
        revision="1.0",
        meter_type=MeterType.GELEX_ME40,
        acceptance_complete=True,
        full_match_pattern=r"^[0-9]{5,8}(\.[0-9]{1,2})?$",
        decimal_policy=DecimalPolicy.OPTIONAL,
        integer_digits_min=5,
        integer_digits_max=8,
        fractional_digits_min=0,
        fractional_digits_max=2,
        min_ocr_confidence_for_accept=0.85,
    )

    return freeze_config(
        PipelineConfig(
            schema_version="1.0",
            config_id="test_config_v1",
            revision="1.0.0",
            quality=QualityPolicy(
                profile_id="quality_v1",
                revision="1.0",
                acceptance_complete=True,
                min_width=100,
                min_height=100,
                dark_pixel_threshold=15,
                max_dark_fraction=0.90,
                bright_pixel_threshold=240,
                max_bright_fraction=0.90,
                blur_metric_id="laplacian_variance_v1",
                min_blur_metric=10.0,
            ),
            localization_profiles={
                vsee_loc.profile_id: vsee_loc,
                gelex_loc.profile_id: gelex_loc,
            },
            validation_profiles={
                vsee_val.profile_id: vsee_val,
                gelex_val.profile_id: gelex_val,
            },
            reading_profiles={
                "vsee_vse3t_reading_v1": ReadingProfile(
                    profile_id="vsee_vse3t_reading_v1",
                    meter_type=MeterType.VSEE_VSE3T,
                    validation_profile_id=vsee_val.profile_id,
                ),
                "gelex_me40_reading_v1": ReadingProfile(
                    profile_id="gelex_me40_reading_v1",
                    meter_type=MeterType.GELEX_ME40,
                    validation_profile_id=gelex_val.profile_id,
                ),
            },
            meter_type_reading_profiles={
                MeterType.VSEE_VSE3T: "vsee_vse3t_reading_v1",
                MeterType.GELEX_ME40: "gelex_me40_reading_v1",
            },
            decision=DecisionPolicy(
                policy_id="decision_v1",
                version="1.0",
                supported_meter_types=(MeterType.VSEE_VSE3T, MeterType.GELEX_ME40),
            ),
            recognition=RecognitionPolicy(
                reader_id="mock_ocr",
                logical_model_id="PP-OCRv6-medium",
                resolved_model_version="1.0",
                model_path="",
            ),
            detector=DetectorPolicy(
                logical_model_id="yuva_yolo",
                resolved_model_version="v1",
                model_path="",
            ),
            artifacts=ArtifactPolicy(
                policy_id="artifacts_v1",
                artifact_root=str(temp_artifact_dir),
            ),
        )
    )


@pytest.fixture
def mock_pipeline(sample_pipeline_config: PipelineConfig, temp_artifact_dir: Path) -> MagicMock:
    """Fixture providing a mock MeterReadingPipeline."""
    mock = MagicMock()
    mock.config = sample_pipeline_config
    mock.run.side_effect = lambda input_data: create_sample_pipeline_result(
        outcome=DecisionOutcome.ACCEPT,
        run_id="run_" + input_data.request_id[:8],
        request_id=input_data.request_id,
        meter_type=input_data.meter_type,
        artifact_root=temp_artifact_dir,
    )
    return mock


@pytest.fixture
def test_app(
    sample_pipeline_config: PipelineConfig,
    mock_pipeline: MagicMock,
    temp_artifact_dir: Path,
) -> Generator[TestClient, None, None]:
    """TestClient fixture with mock pipeline and registry pre-injected."""
    settings = Settings(
        artifact_root=temp_artifact_dir,
        enable_learned_primary=False,
        max_upload_bytes=1048576,  # 1MB for tests
        default_localization_profiles={
            "VSEE_VSE3T": "vsee_vse3t_station_a_v1",
            "GELEX_ME40": "gelex_me40_station_b_v1",
        },
    )

    app = create_app(settings)
    registry = InMemoryArtifactRegistry(artifact_root=temp_artifact_dir)

    coordinator = InferenceCoordinator(
        pipeline=mock_pipeline,
        registry=registry,
        settings=settings,
        core_revision_verified=True,
        core_current_revision=settings.expected_core_revision,
        import_origin_verified=True,
        is_calibrated=True,
        calibration_status="CALIBRATED",
        ocr_runtime_status="AVAILABLE",
        ocr_asset_status="AVAILABLE",
        detector_runtime_status="AVAILABLE",
        detector_asset_status="AVAILABLE",
        artifact_storage_status="AVAILABLE",
    )

    app.state.settings = settings
    app.state.registry = registry
    app.state.pipeline = mock_pipeline
    app.state.coordinator = coordinator

    with TestClient(app) as client:
        yield client
