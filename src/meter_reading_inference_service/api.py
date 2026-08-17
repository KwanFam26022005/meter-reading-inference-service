"""FastAPI API routes for meter-reading-inference-service."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

from meter_reading_inference_service.artifacts import InMemoryArtifactRegistry
from meter_reading_inference_service.errors import (
    InvalidRequestError,
    PipelineNotReadyError,
    UploadTooLargeError,
)
from meter_reading_inference_service.inference import InferenceCoordinator
from meter_reading_inference_service.schemas import (
    ArtifactsStatus,
    CapabilitiesStatus,
    CoreStatus,
    DetectorStatus,
    HealthResponse,
    InferenceResponse,
    ModelsStatusResponse,
    PipelineConfigStatus,
    RecognitionStatus,
)
from meter_reading_inference_service.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Liveness probe. Returns ok without running inference."""
    return HealthResponse()


@router.get("/api/v1/models/status", response_model=ModelsStatusResponse, tags=["Status"])
async def models_status(request: Request) -> ModelsStatusResponse:
    """Readiness and model capability status."""
    coordinator: InferenceCoordinator | None = getattr(request.app.state, "coordinator", None)
    settings: Settings = getattr(request.app.state, "settings", Settings())

    if coordinator is None:
        return ModelsStatusResponse(
            status="not_ready",
            ready=False,
            core=CoreStatus(
                expected_revision=settings.expected_core_revision,
                current_revision=getattr(request.app.state, "core_current_revision", None),
                revision_verified=getattr(request.app.state, "core_revision_verified", False),
                import_origin_verified=False,
                dependency_verified=False,
            ),
            pipeline_config=PipelineConfigStatus(
                config_id="none",
                revision="none",
                valid=False,
                calibrated=False,
                calibration_status="NOT_LOADED",
            ),
            recognition=RecognitionStatus(
                model_id="PP-OCRv6-medium",
                runtime_status="MISSING",
                asset_status="MISSING",
            ),
            detector=DetectorStatus(
                model_id="none",
                runtime_status="NOT_EVALUATED",
                asset_status="NOT_EVALUATED",
                required_for_default=False,
            ),
            artifacts=ArtifactsStatus(status="UNAVAILABLE"),
            capabilities=CapabilitiesStatus(
                supported_meter_types=["VSEE_VSE3T", "GELEX_ME40"],
                default_localization_profiles=settings.default_localization_profiles,
                configured_mode_ready=False,
                learned_shadow_mode_ready=False,
                learned_shadow_decision_path_ready=False,
                learned_shadow_telemetry_ready=False,
                learned_primary_enabled=settings.enable_learned_primary,
            ),
        )

    pipeline = coordinator.pipeline
    config = pipeline.config if pipeline else None

    core_verified = coordinator.core_revision_verified
    import_origin_verified = getattr(coordinator, "import_origin_verified", False)
    dependency_verified = core_verified and import_origin_verified
    is_calibrated = coordinator.is_calibrated
    cal_status = coordinator.calibration_status
    rec_runtime = coordinator.ocr_runtime_status
    rec_asset = coordinator.ocr_asset_status
    det_runtime = getattr(coordinator, "detector_runtime_status", "NOT_EVALUATED")
    det_asset = coordinator.detector_asset_status
    art_storage = getattr(coordinator, "artifact_storage_status", "UNAVAILABLE")

    configured_ready = coordinator.is_ready
    det_asset_ready = det_asset == "AVAILABLE"
    det_runtime_ready = det_runtime == "AVAILABLE"
    shadow_telemetry_ready = det_asset_ready and det_runtime_ready
    shadow_mode_ready = configured_ready and shadow_telemetry_ready

    overall_status: str
    if not configured_ready:
        overall_status = "not_ready"
    elif not shadow_telemetry_ready:
        overall_status = "degraded"
    else:
        overall_status = "ready"

    supported_mtypes = (
        [m.value for m in config.decision.supported_meter_types]
        if config
        else ["VSEE_VSE3T", "GELEX_ME40"]
    )
    rec_model_id = config.recognition.logical_model_id if config else "PP-OCRv6-medium"
    det_model_id = (
        (config.detector.logical_model_id if config.detector else "none") if config else "none"
    )

    auto_mode_ready = (
        dependency_verified
        and (rec_runtime == "AVAILABLE")
        and (rec_asset == "AVAILABLE")
        and (det_runtime == "AVAILABLE")
        and (det_asset == "AVAILABLE")
        and (art_storage == "AVAILABLE")
    )

    return ModelsStatusResponse(
        status=overall_status,
        ready=configured_ready,
        core=CoreStatus(
            expected_revision=settings.expected_core_revision,
            current_revision=coordinator.core_current_revision,
            revision_verified=core_verified,
            import_origin_verified=import_origin_verified,
            dependency_verified=dependency_verified,
        ),
        pipeline_config=PipelineConfigStatus(
            config_id=config.config_id if config else "none",
            revision=config.revision if config else "none",
            valid=(config is not None),
            calibrated=is_calibrated,
            calibration_status=cal_status,
        ),
        recognition=RecognitionStatus(
            model_id=rec_model_id,
            runtime_status=rec_runtime,
            asset_status=rec_asset,
        ),
        detector=DetectorStatus(
            model_id=det_model_id,
            runtime_status=det_runtime,
            asset_status=det_asset,
            required_for_default=False,
        ),
        artifacts=ArtifactsStatus(status=art_storage),
        capabilities=CapabilitiesStatus(
            supported_meter_types=supported_mtypes,
            default_localization_profiles=settings.default_localization_profiles,
            configured_mode_ready=configured_ready,
            learned_shadow_mode_ready=shadow_mode_ready,
            learned_shadow_decision_path_ready=configured_ready,
            learned_shadow_telemetry_ready=shadow_telemetry_ready,
            learned_primary_enabled=settings.enable_learned_primary,
            auto_mode_ready=auto_mode_ready,
        ),
    )


@router.post(
    "/api/v1/meter-readings/infer",
    response_model=InferenceResponse,
    tags=["Inference"],
)
async def infer_meter_reading(request: Request) -> InferenceResponse:
    """Execute pipeline inference on an uploaded image with bounded pre-parsing checks."""
    coordinator: InferenceCoordinator | None = getattr(request.app.state, "coordinator", None)
    settings: Settings = getattr(request.app.state, "settings", Settings())

    if coordinator is None:
        raise PipelineNotReadyError("Pipeline coordinator is not initialized")

    # 1. Early Content-Length check
    content_length_header = request.headers.get("content-length")
    if content_length_header:
        try:
            content_length = int(content_length_header)
            # Allow 8KB extra overhead for multipart boundary headers
            if content_length > settings.max_upload_bytes + 8192:
                raise UploadTooLargeError(
                    message=f"Request Content-Length ({content_length} bytes) exceeds maximum limit of {settings.max_upload_bytes} bytes"
                )
        except ValueError:
            pass

    # 2. Explicitly bounded multipart parsing
    try:
        form = await request.form(
            max_files=1,
            max_fields=10,
            max_part_size=settings.max_upload_bytes,
        )
    except Exception as exc:
        if "too large" in str(exc).lower() or "limit" in str(exc).lower():
            raise UploadTooLargeError(
                message=f"Uploaded payload exceeds limit of {settings.max_upload_bytes} bytes"
            ) from exc
        raise InvalidRequestError(message=f"Malformed multipart form request: {exc}") from exc

    image_field = form.get("image")
    if not isinstance(image_field, UploadFile):
        raise InvalidRequestError("Missing required file field 'image' in multipart upload")

    meter_type_field = form.get("meter_type")
    if not meter_type_field or not isinstance(meter_type_field, str):
        raise InvalidRequestError("Missing required form parameter 'meter_type'")

    request_id_field = form.get("request_id")
    reading_profile_field = form.get("reading_profile")
    validation_profile_field = form.get("validation_profile_id")
    localization_profile_field = form.get("localization_profile_id")
    locator_mode_field = form.get("locator_mode") or "CONFIGURED"

    # Read uploaded bytes with bounded enforcement
    max_bytes = settings.max_upload_bytes
    image_bytes = await image_field.read(max_bytes + 1)
    if len(image_bytes) > max_bytes:
        raise UploadTooLargeError(message=f"Uploaded image size exceeds limit of {max_bytes} bytes")

    declared_content_type = image_field.content_type or "image/jpeg"

    return await coordinator.infer(
        image_bytes=image_bytes,
        media_type=declared_content_type,
        meter_type_str=meter_type_field,
        request_id=request_id_field if isinstance(request_id_field, str) else None,
        reading_profile=reading_profile_field if isinstance(reading_profile_field, str) else None,
        validation_profile_id=validation_profile_field
        if isinstance(validation_profile_field, str)
        else None,
        localization_profile_id=localization_profile_field
        if isinstance(localization_profile_field, str)
        else None,
        locator_mode_str=locator_mode_field
        if isinstance(locator_mode_field, str)
        else "CONFIGURED",
    )


@router.get(
    "/api/v1/runs/{run_id}/artifacts/{role}",
    response_class=FileResponse,
    tags=["Artifacts"],
)
async def get_run_artifact(
    request: Request,
    run_id: str,
    role: str,
) -> FileResponse:
    """Serve a pipeline artifact image safely by run_id and role."""
    registry: InMemoryArtifactRegistry | None = getattr(request.app.state, "registry", None)
    if registry is None:
        raise PipelineNotReadyError("Artifact registry is not initialized")

    resolved_path, media_type = registry.resolve_artifact_file(run_id=run_id, role=role)

    return FileResponse(
        path=str(resolved_path),
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )
