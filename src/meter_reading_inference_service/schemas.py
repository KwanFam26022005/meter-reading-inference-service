"""Pydantic schemas and DTOs for meter-reading-inference-service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Health & Status Schemas
# -----------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "meter-reading-inference-service"
    api_version: str = "v1"


class CoreStatus(BaseModel):
    expected_revision: str
    current_revision: str | None = None
    revision_verified: bool
    import_origin_verified: bool = False
    dependency_verified: bool = False  # True when BOTH revision_verified AND import_origin_verified


class PipelineConfigStatus(BaseModel):
    config_id: str
    revision: str
    valid: bool
    calibrated: bool = False
    calibration_status: str = "EXAMPLE_ONLY"


class RecognitionStatus(BaseModel):
    model_id: str
    runtime_status: str
    asset_status: str = "MISSING"


class DetectorStatus(BaseModel):
    model_id: str
    runtime_status: str = "NOT_EVALUATED"
    asset_status: str
    required_for_default: bool = False


class CapabilitiesStatus(BaseModel):
    supported_meter_types: list[str]
    default_localization_profiles: dict[str, str]
    configured_mode_ready: bool
    learned_shadow_mode_ready: bool
    learned_shadow_decision_path_ready: bool = False
    learned_shadow_telemetry_ready: bool = False
    learned_primary_enabled: bool


class ArtifactsStatus(BaseModel):
    status: str  # AVAILABLE | UNAVAILABLE


class ModelsStatusResponse(BaseModel):
    status: Literal["ready", "degraded", "not_ready"]
    ready: bool
    core: CoreStatus
    pipeline_config: PipelineConfigStatus
    recognition: RecognitionStatus
    detector: DetectorStatus
    artifacts: ArtifactsStatus
    capabilities: CapabilitiesStatus


# -----------------------------------------------------------------------------
# Inference Findings and Metrics
# -----------------------------------------------------------------------------


class SanitizedFinding(BaseModel):
    code: str
    category: str
    severity: str
    stage: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class QualityMetricItem(BaseModel):
    name: str
    value: int | float | bool
    unit: str | None = None
    threshold: int | float | bool | None = None
    check_version: str


class QualityGroupedChecks(BaseModel):
    DIMENSIONS: list[QualityMetricItem] = Field(default_factory=list)
    DARK_EXPOSURE: list[QualityMetricItem] = Field(default_factory=list)
    BRIGHT_EXPOSURE: list[QualityMetricItem] = Field(default_factory=list)
    SHARPNESS: list[QualityMetricItem] = Field(default_factory=list)
    other_metrics: list[QualityMetricItem] = Field(default_factory=list)


class QualityTrace(BaseModel):
    metrics: list[QualityMetricItem] = Field(default_factory=list)
    grouped_checks: QualityGroupedChecks = Field(default_factory=QualityGroupedChecks)
    findings: list[SanitizedFinding] = Field(default_factory=list)


class StageTraceItem(BaseModel):
    stage: str
    status: Literal["PASS", "FAIL", "NOT_RUN"]
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: float | None = None
    implementation_id: str | None = None
    findings: list[SanitizedFinding] = Field(default_factory=list)
    not_run_reason: str | None = None


class RecognitionTrace(BaseModel):
    raw_text: str | None = None
    confidence: float | None = None
    duration_ms: float | None = None
    reader_id: str | None = None
    logical_model_id: str | None = None
    resolved_model_version: str | None = None
    findings: list[SanitizedFinding] = Field(default_factory=list)


class NormalizationTrace(BaseModel):
    raw_text: str | None = None
    normalized_text: str | None = None
    normalizer_id: str | None = None
    normalizer_version: str | None = None
    operations: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[SanitizedFinding] = Field(default_factory=list)


class ValidationCheckItem(BaseModel):
    rule_id: str
    passed: bool
    observed: Any = None
    expected: Any = None


class ValidationTrace(BaseModel):
    is_valid: bool | None = None
    acceptance_profile_complete: bool | None = None
    profile_id: str | None = None
    profile_revision: str | None = None
    normalized_text: str | None = None
    parsed_value: str | None = None
    min_ocr_confidence_for_accept: float | None = None
    checks: list[ValidationCheckItem] = Field(default_factory=list)
    findings: list[SanitizedFinding] = Field(default_factory=list)


class TracePayload(BaseModel):
    stages: list[StageTraceItem] = Field(default_factory=list)
    quality: QualityTrace | None = None
    recognition: RecognitionTrace | None = None
    normalization: NormalizationTrace | None = None
    validation: ValidationTrace | None = None


# -----------------------------------------------------------------------------
# Summary DTO
# -----------------------------------------------------------------------------


class AcceptanceCheckSummary(BaseModel):
    check_id: str
    passed: bool
    reason_codes: list[str] = Field(default_factory=list)


class DecisionSummary(BaseModel):
    outcome: Literal["ACCEPT", "REVIEW", "RECAPTURE", "ERROR"]
    reason_codes: list[str] = Field(default_factory=list)
    policy_id: str
    policy_version: str
    acceptance_checks: list[AcceptanceCheckSummary] = Field(default_factory=list)


class ReadingSummary(BaseModel):
    raw_text: str | None = None
    normalized_text: str | None = None
    parsed_value: str | None = None
    confidence: float | None = None
    is_valid: bool = False


class InferenceSummary(BaseModel):
    started_at: str
    finished_at: str
    duration_ms: float
    decision: DecisionSummary
    reading: ReadingSummary


# -----------------------------------------------------------------------------
# Visualization DTOs
# -----------------------------------------------------------------------------


class BBoxRegion(BaseModel):
    coordinate_space_id: str
    unit: str
    x: float
    y: float
    width: float
    height: float
    pixel_bbox: dict[str, int] | None = None


class SourceVisual(BaseModel):
    coordinate_space_id: str = "source"
    artifact_url: str | None = None
    width: int | None = None
    height: int | None = None


class DisplayVisual(BaseModel):
    coordinate_space_id: str = "source"
    artifact_url: str | None = None
    method: str | None = None
    implementation_id: str | None = None
    profile_id: str | None = None
    profile_revision: str | None = None
    confidence: float | None = None
    region: BBoxRegion | None = None


class TransformVisual(BaseModel):
    transform_id: str
    kind: str
    input_space_id: str
    output_space_id: str
    matrix: list[list[float]] | None = None
    inverse_matrix: list[list[float]] | None = None
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None


class WorkingDisplayVisual(BaseModel):
    coordinate_space_id: str = "working_display"
    artifact_url: str | None = None
    geometry_mode: str | None = None
    geometry_source: str | None = None
    implementation_id: str | None = None
    quadrilateral: list[dict[str, float]] | None = None
    transforms: list[TransformVisual] = Field(default_factory=list)


class ReadingValueVisual(BaseModel):
    coordinate_space_id: str = "working_display"
    artifact_url: str | None = None
    method: str | None = None
    implementation_id: str | None = None
    profile_id: str | None = None
    profile_revision: str | None = None
    region: BBoxRegion | None = None
    used_for_ocr: bool = True


class ShadowROI(BaseModel):
    coordinate_space_id: str
    bbox: dict[str, float]
    used_for_ocr: bool
    detector_confidence: float | None = None


class LearnedShadowVisual(BaseModel):
    shadow_status: str | None = None
    configured: ShadowROI | None = None
    learned: ShadowROI | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class VisualizationPayload(BaseModel):
    source: SourceVisual
    display: DisplayVisual | None = None
    working_display: WorkingDisplayVisual | None = None
    reading_value: ReadingValueVisual | None = None
    learned_shadow: LearnedShadowVisual | None = None


# -----------------------------------------------------------------------------
# Provenance DTO
# -----------------------------------------------------------------------------


class SourceIdentityProvenance(BaseModel):
    sha256: str
    byte_length: int
    media_type: str
    encoded_width: int | None = None
    encoded_height: int | None = None
    exif_orientation: int | None = None
    orientation_operation: str = "none"
    decoder_id: str = "pil"
    decoder_version: str = ""
    width: int | None = None
    height: int | None = None


class InferenceProvenance(BaseModel):
    core_repository: str = "https://github.com/KwanFam26022005/meter-reading-engine-v2"
    core_revision: str
    revision_verified: bool
    pipeline_schema_version: str = "1.0"
    config_id: str
    config_revision: str
    config_sha256: str
    source: SourceIdentityProvenance
    requested_locator_mode: str
    effective_locator_mode: str
    recognition_model_id: str | None = None
    recognition_model_version: str | None = None
    available_artifact_roles: list[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Top-Level Inference Response
# -----------------------------------------------------------------------------


class InferenceResponse(BaseModel):
    schema_version: Literal["a1.inference.v1"] = "a1.inference.v1"
    request_id: str
    run_id: str
    meter_type: str
    summary: InferenceSummary
    trace: TracePayload
    visualization: VisualizationPayload
    provenance: InferenceProvenance
