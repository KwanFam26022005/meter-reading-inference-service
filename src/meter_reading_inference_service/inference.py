"""Inference coordinator, concurrency guard, and PipelineResult DTO mapper."""

from __future__ import annotations

import asyncio
import io
import logging
import re
import uuid
from typing import Any

from meter_reading_engine.contracts import (
    Finding,
    LearnedLocalizationProvenance,
    LocatorMode,
    MeterType,
    NormalizedBBox,
    PipelineInput,
    PipelineResult,
    PixelBBox,
    RegionRef,
    StageStatus,
    format_utc_rfc3339,
)
from meter_reading_engine.pipeline import MeterReadingPipeline
from PIL import Image

from meter_reading_inference_service.artifacts import InMemoryArtifactRegistry
from meter_reading_inference_service.errors import (
    InvalidRequestError,
    LearnedPrimaryDisabledError,
)
from meter_reading_inference_service.schemas import (
    AcceptanceCheckSummary,
    BBoxRegion,
    DecisionSummary,
    DisplayVisual,
    InferenceProvenance,
    InferenceResponse,
    InferenceSummary,
    LearnedShadowVisual,
    NormalizationTrace,
    QualityGroupedChecks,
    QualityMetricItem,
    QualityTrace,
    ReadingSummary,
    ReadingValueVisual,
    RecognitionTrace,
    SanitizedFinding,
    ShadowROI,
    SourceIdentityProvenance,
    SourceVisual,
    StageTraceItem,
    TracePayload,
    TransformVisual,
    ValidationCheckItem,
    ValidationTrace,
    VisualizationPayload,
    WorkingDisplayVisual,
)
from meter_reading_inference_service.settings import Settings

logger = logging.getLogger(__name__)


def sanitize_finding(finding: Finding) -> SanitizedFinding:
    """Sanitize finding details to avoid leaking filesystem internals or unhandled errors."""
    sanitized_details: dict[str, Any] = {}
    if finding.details:
        for k, v in finding.details.items():
            # Filter out machine path keys or raw tracebacks
            if k in ("model_path", "traceback", "exception_obj", "raw_exception"):
                continue
            if isinstance(v, (int, float, str, bool)):
                # Strip potential windows / unix filepaths from string values
                if isinstance(v, str) and (":\\" in v or (v.startswith("/") and len(v) > 20)):
                    sanitized_details[k] = "<path_omitted>"
                else:
                    sanitized_details[k] = v
            elif isinstance(v, (list, tuple)):
                sanitized_details[k] = [x for x in v if isinstance(x, (int, float, str, bool))][:20]
            elif isinstance(v, dict):
                sanitized_details[k] = {
                    str(dk): dv for dk, dv in v.items() if isinstance(dv, (int, float, str, bool))
                }

    category_str = (
        finding.category.value if hasattr(finding.category, "value") else str(finding.category)
    )
    severity_str = (
        finding.severity.value if hasattr(finding.severity, "value") else str(finding.severity)
    )
    stage_str = finding.stage.value if hasattr(finding.stage, "value") else str(finding.stage)

    # Sanitize message if it contains long absolute paths
    msg = finding.message
    if ":\\" in msg or "/home/" in msg or "/tmp/" in msg:
        msg = re.sub(r"[A-Za-z]:\\[^ '\"\n]+", "<path_omitted>", msg)
        msg = re.sub(r"/[a-zA-Z0-9_\-\.\/]+", "<path_omitted>", msg)

    return SanitizedFinding(
        code=finding.code,
        category=category_str,
        severity=severity_str,
        stage=stage_str,
        message=msg,
        details=sanitized_details,
    )


def map_stage_status(status: StageStatus | str) -> str:
    s = status.value if hasattr(status, "value") else str(status)
    if s == "SUCCEEDED":
        return "PASS"
    if s == "FAILED":
        return "FAIL"
    return "NOT_RUN"


def bbox_to_region_dto(region: RegionRef | None) -> BBoxRegion | None:
    if region is None:
        return None

    supplied = region.supplied_bbox
    unit_str = (
        region.supplied_unit.value
        if hasattr(region.supplied_unit, "value")
        else str(region.supplied_unit)
    )
    x = float(supplied.x)
    y = float(supplied.y)
    w = float(supplied.width)
    h = float(supplied.height)

    resolved = region.resolved_pixel_bbox
    pixel_bbox_dict = None
    if resolved:
        pixel_bbox_dict = {
            "x": int(resolved.x),
            "y": int(resolved.y),
            "width": int(resolved.width),
            "height": int(resolved.height),
        }

    return BBoxRegion(
        coordinate_space_id=region.coordinate_space_id,
        unit=unit_str,
        x=x,
        y=y,
        width=w,
        height=h,
        pixel_bbox=pixel_bbox_dict,
    )


class InferenceCoordinator:
    """Coordinates validation, concurrency-guarded execution, and DTO assembly."""

    def __init__(
        self,
        pipeline: MeterReadingPipeline | None,
        registry: InMemoryArtifactRegistry,
        settings: Settings,
        core_revision_verified: bool = False,
        core_current_revision: str | None = None,
        import_origin_verified: bool = False,
        is_calibrated: bool = False,
        calibration_status: str = "UNSPECIFIED",
        ocr_runtime_status: str = "MISSING",
        ocr_asset_status: str = "MISSING",
        detector_runtime_status: str = "MISSING",
        detector_asset_status: str = "MISSING",
        artifact_storage_status: str = "UNAVAILABLE",
        readiness_reason: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.registry = registry
        self.settings = settings
        self.core_revision_verified = core_revision_verified
        self.import_origin_verified = import_origin_verified
        self.core_current_revision = core_current_revision or settings.expected_core_revision
        self.is_calibrated = is_calibrated
        self.calibration_status = calibration_status
        self.ocr_runtime_status = ocr_runtime_status
        self.ocr_asset_status = ocr_asset_status
        self.detector_runtime_status = detector_runtime_status
        self.detector_asset_status = detector_asset_status
        self.artifact_storage_status = artifact_storage_status
        self.readiness_reason = readiness_reason
        self.semaphore = asyncio.Semaphore(settings.max_inference_concurrency)

    @property
    def is_ready(self) -> bool:
        """CONFIGURED inference readiness equation.

        ALL of the following must pass:
          - Core import origin verified (imported package == configured Core)
          - Core git revision verified (HEAD == expected frozen SHA)
          - Config valid AND calibration_status == CALIBRATED
          - OCR runtime available
          - OCR asset available
          - Artifact storage available
          - Pipeline constructed

        Detector is NOT required for CONFIGURED readiness.
        """
        if not self.import_origin_verified:
            return False
        if not self.core_revision_verified:
            return False
        if not self.is_calibrated:
            return False
        if self.ocr_runtime_status != "AVAILABLE":
            return False
        if self.ocr_asset_status != "AVAILABLE":
            return False
        if self.artifact_storage_status != "AVAILABLE":
            return False
        if self.pipeline is None:
            return False
        return True

    async def infer(
        self,
        image_bytes: bytes,
        media_type: str,
        meter_type_str: str,
        request_id: str | None = None,
        reading_profile: str | None = None,
        validation_profile_id: str | None = None,
        localization_profile_id: str | None = None,
        locator_mode_str: str = "CONFIGURED",
    ) -> InferenceResponse:
        """Process an inference request end-to-end with strict validation."""
        # 0. Gate inference on configured readiness
        if not self.is_ready:
            from meter_reading_inference_service.errors import PipelineNotReadyError

            reason = self.readiness_reason or "Pipeline is not ready for inference"
            if not self.import_origin_verified:
                reason = "Core package import origin could not be verified against the configured Core repository"
            elif not self.core_revision_verified:
                reason = "Core git revision is unverified or mismatched"
            elif not self.is_calibrated:
                reason = f"Pipeline configuration is marked {self.calibration_status} and is not calibrated for real inference"
            elif self.ocr_runtime_status != "AVAILABLE":
                reason = (
                    f"OCR Python runtime is unavailable or incompatible ({self.ocr_runtime_status})"
                )
            elif self.ocr_asset_status != "AVAILABLE":
                reason = f"Configured local OCR model bundle is missing or invalid ({self.ocr_asset_status})"
            elif self.artifact_storage_status != "AVAILABLE":
                reason = "Artifact storage is not available for inference results"
            elif self.pipeline is None:
                reason = "MeterReadingPipeline engine is not initialized"
            raise PipelineNotReadyError(message=f"PIPELINE_NOT_READY: {reason}")

        # 1. Validate image size
        if len(image_bytes) > self.settings.max_upload_bytes:
            from meter_reading_inference_service.errors import UploadTooLargeError

            raise UploadTooLargeError(
                message=f"Uploaded image size ({len(image_bytes)} bytes) exceeds limit ({self.settings.max_upload_bytes} bytes)"
            )

        # 2. Validate media type
        clean_media = media_type.split(";")[0].strip().lower()
        if clean_media not in ("image/jpeg", "image/png", "image/jpg"):
            from meter_reading_inference_service.errors import UnsupportedMediaTypeError

            raise UnsupportedMediaTypeError(
                message=f"Unsupported media type '{clean_media}'. Allowed types: image/jpeg, image/png"
            )
        if clean_media == "image/jpg":
            clean_media = "image/jpeg"

        # 3. Validate image magic header signature and decode consistency
        from meter_reading_inference_service.errors import UnsupportedMediaTypeError

        is_png_sig = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpeg_sig = image_bytes.startswith(b"\xff\xd8\xff")
        if clean_media == "image/jpeg" and is_png_sig:
            raise UnsupportedMediaTypeError(
                message="Declared media type 'image/jpeg' does not match actual encoded PNG signature"
            )
        if clean_media == "image/png" and is_jpeg_sig:
            raise UnsupportedMediaTypeError(
                message="Declared media type 'image/png' does not match actual encoded JPEG signature"
            )

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                detected_format = img.format
                img.verify()
            if clean_media == "image/jpeg" and detected_format != "JPEG":
                raise UnsupportedMediaTypeError(
                    message=f"Declared media type 'image/jpeg' mismatches decoded image format '{detected_format}'"
                )
            if clean_media == "image/png" and detected_format != "PNG":
                raise UnsupportedMediaTypeError(
                    message=f"Declared media type 'image/png' mismatches decoded image format '{detected_format}'"
                )
        except UnsupportedMediaTypeError:
            raise
        except Exception as exc:
            raise InvalidRequestError(
                message=f"Uploaded image is corrupt or invalid: {exc}"
            ) from exc

        # 4. Validate MeterType
        try:
            meter_type = MeterType(meter_type_str)
        except ValueError as exc:
            supported = [m.value for m in MeterType]
            raise InvalidRequestError(
                message=f"Unknown meter_type '{meter_type_str}'. Supported types: {supported}"
            ) from exc

        # 5. Validate LocatorMode
        try:
            locator_mode = LocatorMode(locator_mode_str)
        except ValueError as exc:
            supported_loc = [m.value for m in LocatorMode]
            raise InvalidRequestError(
                message=f"Unknown locator_mode '{locator_mode_str}'. Supported modes: {supported_loc}"
            ) from exc

        # 6. Enforce ENABLE_LEARNED_PRIMARY policy
        if locator_mode == LocatorMode.LEARNED_PRIMARY and not self.settings.enable_learned_primary:
            raise LearnedPrimaryDisabledError()

        # 7. Explicit localization profile default mapping
        effective_loc_profile_id = localization_profile_id
        if not effective_loc_profile_id:
            effective_loc_profile_id = self.settings.default_localization_profiles.get(
                meter_type.value
            )
            if not effective_loc_profile_id:
                raise InvalidRequestError(
                    message=f"No explicit localization profile provided and no default mapping configured for meter_type '{meter_type.value}'"
                )

        # 8. Build Core PipelineInput (preserving exact uploaded image_bytes)
        req_id = request_id.strip() if request_id and request_id.strip() else uuid.uuid4().hex
        pipeline_input = PipelineInput(
            schema_version="1.0",
            request_id=req_id,
            image_bytes=image_bytes,
            media_type=clean_media,
            meter_type=meter_type,
            localization_profile_id=effective_loc_profile_id,
            reading_profile=reading_profile,
            validation_profile_id=validation_profile_id,
            locator_mode=locator_mode,
            localization_overrides=None,
            source_uri=None,
            caller_metadata={},
            allow_configured_fallback=True,
        )

        # 9. Concurrency-guarded execution (cancellation safe)
        await self.semaphore.acquire()
        try:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(None, self.pipeline.run, pipeline_input)
            try:
                result: PipelineResult = await asyncio.shield(fut)
            except asyncio.CancelledError:
                logger.warning(
                    "Client cancelled request %s; retaining inference slot until pipeline worker finishes.",
                    pipeline_input.request_id,
                )
                await asyncio.wrap_future(fut)
                raise
        finally:
            self.semaphore.release()

        # 10. Register run artifacts if produced
        run_id = result.run.run_id
        if result.artifacts:
            self.registry.register_index(run_id=run_id, artifact_index=result.artifacts)

        # 11. Assemble DTO
        return self._assemble_response(result, pipeline_input)

    def _assemble_response(
        self, result: PipelineResult, input_data: PipelineInput
    ) -> InferenceResponse:
        run_id = result.run.run_id

        # Stages trace
        stage_trace_items: list[StageTraceItem] = []
        for s in result.stages:
            duration_ms = (s.duration_ns / 1_000_000.0) if s.duration_ns is not None else None
            if duration_ms is None and s.started_at and s.finished_at:
                duration_ms = (s.finished_at - s.started_at).total_seconds() * 1000.0

            stage_trace_items.append(
                StageTraceItem(
                    stage=s.stage.value if hasattr(s.stage, "value") else str(s.stage),
                    status=map_stage_status(s.status),
                    started_at=format_utc_rfc3339(s.started_at) if s.started_at else None,
                    finished_at=format_utc_rfc3339(s.finished_at) if s.finished_at else None,
                    duration_ms=round(duration_ms, 3) if duration_ms is not None else None,
                    implementation_id=s.implementation_id,
                    findings=[sanitize_finding(f) for f in s.findings],
                    not_run_reason=s.not_run_reason,
                )
            )

        # Quality trace
        quality_trace: QualityTrace | None = None
        if result.quality:
            metrics_list: list[QualityMetricItem] = []
            grouped = QualityGroupedChecks()
            for m in result.quality.metrics:
                item = QualityMetricItem(
                    name=m.name,
                    value=m.value,
                    unit=m.unit,
                    threshold=m.threshold,
                    check_version=m.check_version,
                )
                metrics_list.append(item)
                name_upper = m.name.upper()
                if (
                    "WIDTH" in name_upper
                    or "HEIGHT" in name_upper
                    or "DIMENSION" in name_upper
                    or "ASPECT" in name_upper
                ):
                    grouped.DIMENSIONS.append(item)
                elif "DARK" in name_upper:
                    grouped.DARK_EXPOSURE.append(item)
                elif "BRIGHT" in name_upper or "GLARE" in name_upper:
                    grouped.BRIGHT_EXPOSURE.append(item)
                elif "BLUR" in name_upper or "LAPLACIAN" in name_upper or "SHARP" in name_upper:
                    grouped.SHARPNESS.append(item)
                else:
                    grouped.other_metrics.append(item)

            quality_trace = QualityTrace(
                metrics=metrics_list,
                grouped_checks=grouped,
                findings=[sanitize_finding(f) for f in result.quality.findings],
            )

        # Recognition trace
        recognition_trace: RecognitionTrace | None = None
        if result.recognition:
            rec = result.recognition
            rec_ms = rec.duration_ns / 1_000_000.0 if rec.duration_ns else None
            recognition_trace = RecognitionTrace(
                raw_text=rec.raw_text,
                confidence=rec.confidence,
                duration_ms=round(rec_ms, 3) if rec_ms is not None else None,
                reader_id=rec.model.reader_id if rec.model else None,
                logical_model_id=rec.model.logical_model_id if rec.model else None,
                resolved_model_version=rec.model.resolved_model_version if rec.model else None,
                findings=[sanitize_finding(f) for f in rec.findings],
            )

        # Normalization trace
        normalization_trace: NormalizationTrace | None = None
        if result.normalization:
            norm = result.normalization
            normalization_trace = NormalizationTrace(
                raw_text=norm.raw_text,
                normalized_text=norm.normalized_text,
                normalizer_id=norm.normalizer_id,
                normalizer_version=norm.normalizer_version,
                operations=[
                    {"operation": op.operation, "parameters": dict(op.parameters)}
                    for op in norm.operations
                ],
                findings=[sanitize_finding(f) for f in norm.findings],
            )

        # Validation trace
        validation_trace: ValidationTrace | None = None
        if result.validation:
            val = result.validation
            checks_list = [
                ValidationCheckItem(
                    rule_id=c.rule_id,
                    passed=c.passed,
                    observed=c.observed,
                    expected=c.expected,
                )
                for c in val.checks
            ]
            validation_trace = ValidationTrace(
                is_valid=val.is_valid,
                acceptance_profile_complete=val.acceptance_profile_complete,
                profile_id=val.profile_id,
                profile_revision=val.profile_revision,
                normalized_text=val.normalized_text,
                parsed_value=val.parsed_value,
                min_ocr_confidence_for_accept=val.min_ocr_confidence_for_accept,
                checks=checks_list,
                findings=[sanitize_finding(f) for f in val.findings],
            )

        trace_payload = TracePayload(
            stages=stage_trace_items,
            quality=quality_trace,
            recognition=recognition_trace,
            normalization=normalization_trace,
            validation=validation_trace,
        )

        # Decision & Reading Summary
        dec = result.decision
        decision_summary = DecisionSummary(
            outcome=dec.outcome.value if hasattr(dec.outcome, "value") else str(dec.outcome),
            reason_codes=list(dec.reason_codes),
            policy_id=dec.policy_id,
            policy_version=dec.policy_version,
            acceptance_checks=[
                AcceptanceCheckSummary(
                    check_id=chk.check_id,
                    passed=chk.passed,
                    reason_codes=list(chk.reason_codes),
                )
                for chk in dec.acceptance_checks
            ],
        )

        reading_summary = ReadingSummary(
            raw_text=result.recognition.raw_text if result.recognition else None,
            normalized_text=result.normalization.normalized_text if result.normalization else None,
            parsed_value=result.validation.parsed_value if result.validation else None,
            confidence=result.recognition.confidence if result.recognition else None,
            is_valid=result.validation.is_valid if result.validation else False,
        )

        total_duration_ms = (
            (result.duration_ns / 1_000_000.0)
            if result.duration_ns
            else (result.finished_at - result.run.started_at).total_seconds() * 1000.0
        )

        summary = InferenceSummary(
            started_at=format_utc_rfc3339(result.run.started_at),
            finished_at=format_utc_rfc3339(result.finished_at),
            duration_ms=round(total_duration_ms, 3),
            decision=decision_summary,
            reading=reading_summary,
        )

        # Visualizations & Artifact URLs
        available_roles = self.registry.get_available_roles(run_id)

        source_url = (
            f"/api/v1/runs/{run_id}/artifacts/source" if "source" in available_roles else None
        )
        display_url = (
            f"/api/v1/runs/{run_id}/artifacts/display" if "display" in available_roles else None
        )
        working_url = (
            f"/api/v1/runs/{run_id}/artifacts/working_display"
            if "working_display" in available_roles
            else None
        )
        reading_url = (
            f"/api/v1/runs/{run_id}/artifacts/reading_value"
            if "reading_value" in available_roles
            else None
        )

        source_vis = SourceVisual(
            coordinate_space_id="source",
            artifact_url=source_url,
            width=result.source.width,
            height=result.source.height,
        )

        display_vis: DisplayVisual | None = None
        if result.display_localization:
            dloc = result.display_localization
            method_str = dloc.method.value if hasattr(dloc.method, "value") else str(dloc.method)
            display_vis = DisplayVisual(
                coordinate_space_id="source",
                artifact_url=display_url,
                method=method_str,
                implementation_id=dloc.implementation_id,
                profile_id=dloc.profile_id,
                profile_revision=dloc.profile_revision,
                confidence=dloc.confidence,
                region=bbox_to_region_dto(dloc.region),
            )

        working_vis: WorkingDisplayVisual | None = None
        if result.geometry:
            geom = result.geometry
            mode_str = (
                geom.instruction.mode.value
                if hasattr(geom.instruction.mode, "value")
                else str(geom.instruction.mode)
            )
            source_str = (
                geom.instruction.source.value
                if hasattr(geom.instruction.source, "value")
                else str(geom.instruction.source)
            )

            quad_points = None
            if geom.resolved_input_quad:
                quad_points = [{"x": p.x, "y": p.y} for p in geom.resolved_input_quad.points]

            transforms_vis = [
                TransformVisual(
                    transform_id=t.transform_id,
                    kind=t.kind.value if hasattr(t.kind, "value") else str(t.kind),
                    input_space_id=t.input_space_id,
                    output_space_id=t.output_space_id,
                    matrix=[list(row) for row in t.matrix_3x3_input_to_output]
                    if t.matrix_3x3_input_to_output
                    else None,
                    inverse_matrix=[list(row) for row in t.inverse_matrix_3x3]
                    if t.inverse_matrix_3x3
                    else None,
                    input_shape=list(t.input_shape) if t.input_shape else None,
                    output_shape=list(t.output_shape) if t.output_shape else None,
                )
                for t in geom.transforms
            ]

            working_vis = WorkingDisplayVisual(
                coordinate_space_id="working_display",
                artifact_url=working_url,
                geometry_mode=mode_str,
                geometry_source=source_str,
                implementation_id=geom.implementation_id,
                quadrilateral=quad_points,
                transforms=transforms_vis,
            )

        reading_vis: ReadingValueVisual | None = None
        if result.value_localization:
            vloc = result.value_localization
            method_str = vloc.method.value if hasattr(vloc.method, "value") else str(vloc.method)
            reading_vis = ReadingValueVisual(
                coordinate_space_id="working_display",
                artifact_url=reading_url,
                method=method_str,
                implementation_id=vloc.implementation_id,
                profile_id=vloc.profile_id,
                profile_revision=vloc.profile_revision,
                region=bbox_to_region_dto(vloc.region),
                used_for_ocr=True,
            )

        # Learned shadow extraction
        shadow_vis: LearnedShadowVisual | None = None
        if (
            result.value_localization
            and "learned_localization" in result.value_localization.metadata
        ):
            prov_data = result.value_localization.metadata["learned_localization"]
            shadow_vis = self._extract_learned_shadow(prov_data, result)

        vis_payload = VisualizationPayload(
            source=source_vis,
            display=display_vis,
            working_display=working_vis,
            reading_value=reading_vis,
            learned_shadow=shadow_vis,
        )

        # Provenance
        source_prov = SourceIdentityProvenance(
            sha256=result.source.sha256,
            byte_length=result.source.byte_length,
            media_type=result.source.media_type,
            encoded_width=result.source.encoded_width,
            encoded_height=result.source.encoded_height,
            exif_orientation=result.source.exif_orientation,
            orientation_operation=result.source.orientation_operation,
            decoder_id=result.source.decoder_id,
            decoder_version=result.source.decoder_version,
            width=result.source.width,
            height=result.source.height,
        )

        effective_loc_mode = input_data.locator_mode
        if isinstance(effective_loc_mode, LocatorMode):
            effective_loc_mode = effective_loc_mode.value

        provenance = InferenceProvenance(
            core_repository="https://github.com/KwanFam26022005/meter-reading-engine-v2",
            core_revision=self.core_current_revision,
            revision_verified=self.core_revision_verified,
            pipeline_schema_version=result.schema_version,
            config_id=self.pipeline.config.config_id,
            config_revision=self.pipeline.config.revision,
            config_sha256=self.pipeline.config.config_sha256,
            source=source_prov,
            requested_locator_mode=input_data.locator_mode.value
            if hasattr(input_data.locator_mode, "value")
            else str(input_data.locator_mode),
            effective_locator_mode=str(effective_loc_mode),
            recognition_model_id=result.recognition.model.logical_model_id
            if result.recognition
            else None,
            recognition_model_version=result.recognition.model.resolved_model_version
            if result.recognition
            else None,
            available_artifact_roles=available_roles,
        )

        return InferenceResponse(
            schema_version="a1.inference.v1",
            request_id=result.run.request_id,
            run_id=run_id,
            meter_type=result.run.meter_type.value
            if hasattr(result.run.meter_type, "value")
            else str(result.run.meter_type),
            summary=summary,
            trace=trace_payload,
            visualization=vis_payload,
            provenance=provenance,
        )

    def _extract_learned_shadow(self, prov_obj: Any, result: PipelineResult) -> LearnedShadowVisual:
        """Extract safe shadow visual DTO from LearnedLocalizationProvenance."""
        if isinstance(prov_obj, LearnedLocalizationProvenance):
            p = prov_obj
        elif isinstance(prov_obj, dict):
            # Dict representation
            shadow_status = prov_obj.get("shadow_status")
            configured_bbox_dict = {}
            if result.value_localization and result.value_localization.region:
                r = result.value_localization.region.resolved_pixel_bbox
                configured_bbox_dict = {"x": r.x, "y": r.y, "width": r.width, "height": r.height}

            learned_bbox_dict = None
            sel_bbox = prov_obj.get("selected_bbox")
            if sel_bbox:
                if isinstance(sel_bbox, (PixelBBox, NormalizedBBox)):
                    learned_bbox_dict = {
                        "x": sel_bbox.x,
                        "y": sel_bbox.y,
                        "width": sel_bbox.width,
                        "height": sel_bbox.height,
                    }
                elif isinstance(sel_bbox, dict):
                    learned_bbox_dict = sel_bbox

            return LearnedShadowVisual(
                shadow_status=shadow_status,
                configured=ShadowROI(
                    coordinate_space_id="working_display",
                    bbox=configured_bbox_dict,
                    used_for_ocr=True,
                )
                if configured_bbox_dict
                else None,
                learned=ShadowROI(
                    coordinate_space_id="working_display",
                    bbox=learned_bbox_dict or {},
                    used_for_ocr=False,
                    detector_confidence=prov_obj.get("selected_detector_confidence"),
                )
                if learned_bbox_dict
                else None,
                provenance={
                    "model_id": prov_obj.get("model_id"),
                    "checkpoint_sha256": prov_obj.get("checkpoint_sha256"),
                    "input_space": prov_obj.get("input_space"),
                    "conf": prov_obj.get("conf"),
                    "iou": prov_obj.get("iou"),
                    "max_det": prov_obj.get("max_det"),
                    "proposal_count": prov_obj.get("proposal_count"),
                    "selected_proposal_index": prov_obj.get("selected_proposal_index"),
                    "fallback_reason": prov_obj.get("fallback_reason"),
                    "failure_reason": prov_obj.get("failure_reason"),
                    "failure_type": prov_obj.get("failure_type"),
                },
            )

        # Dataclass object
        cfg_roi = None
        if result.value_localization and result.value_localization.region:
            r = result.value_localization.region.resolved_pixel_bbox
            cfg_roi = ShadowROI(
                coordinate_space_id="working_display",
                bbox={"x": r.x, "y": r.y, "width": r.width, "height": r.height},
                used_for_ocr=True,
            )

        learned_roi = None
        if p.selected_bbox:
            learned_roi = ShadowROI(
                coordinate_space_id="working_display",
                bbox={
                    "x": p.selected_bbox.x,
                    "y": p.selected_bbox.y,
                    "width": p.selected_bbox.width,
                    "height": p.selected_bbox.height,
                },
                used_for_ocr=False,
                detector_confidence=p.selected_detector_confidence,
            )

        return LearnedShadowVisual(
            shadow_status=p.shadow_status,
            configured=cfg_roi,
            learned=learned_roi,
            provenance={
                "model_id": p.model_id,
                "checkpoint_sha256": p.checkpoint_sha256,
                "input_space": p.input_space,
                "conf": p.conf,
                "iou": p.iou,
                "max_det": p.max_det,
                "proposal_count": p.proposal_count,
                "selected_proposal_index": p.selected_proposal_index,
                "fallback_reason": p.fallback_reason,
                "failure_reason": p.failure_reason,
                "failure_type": p.failure_type,
            },
        )
