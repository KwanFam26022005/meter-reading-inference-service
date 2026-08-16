"""FastAPI Application Entrypoint and Lifespan."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from meter_reading_inference_service.api import router
from meter_reading_inference_service.artifacts import InMemoryArtifactRegistry
from meter_reading_inference_service.core import (
    check_artifact_storage,
    check_core_import_origin,
    check_core_revision,
    check_detector_asset,
    check_ocr_asset,
    check_ocr_runtime,
    get_yaml_calibration_status,
)
from meter_reading_inference_service.errors import (
    UploadTooLargeError,
    format_error_response,
    register_exception_handlers,
)
from meter_reading_inference_service.inference import InferenceCoordinator
from meter_reading_inference_service.settings import Settings

logger = logging.getLogger("meter_reading_inference_service")


class RequestSizeLimitMiddleware:
    """ASGI middleware to reject requests whose body exceeds MAX_UPLOAD_BYTES early."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                cl = int(content_length.decode("latin1"))
                if cl > self.max_bytes + 8192:
                    res_body = json.dumps(
                        format_error_response(
                            code="UPLOAD_TOO_LARGE",
                            message=f"Request Content-Length ({cl} bytes) exceeds limit of {self.max_bytes} bytes",
                        )
                    ).encode("utf-8")
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 413,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(res_body)).encode("latin1")),
                            ],
                        }
                    )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": res_body,
                        }
                    )
                    return
            except ValueError:
                pass

        received_bytes = 0

        async def custom_receive() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                received_bytes += len(body)
                if received_bytes > self.max_bytes + 8192:
                    raise UploadTooLargeError(
                        message=f"Uploaded payload stream exceeded limit of {self.max_bytes} bytes"
                    )
            return message

        try:
            await self.app(scope, custom_receive, send)
        except UploadTooLargeError as exc:
            res_body = json.dumps(format_error_response(code=exc.code, message=exc.message)).encode(
                "utf-8"
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(res_body)).encode("latin1")),
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": res_body,
                }
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager creating pipeline and registry once at startup."""
    settings: Settings = getattr(app.state, "settings", Settings())
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.info("Initializing meter-reading-inference-service...")

    # Initialize Artifact Registry if not already present (does not require eager mkdir)
    if not hasattr(app.state, "registry") or app.state.registry is None:
        registry = InMemoryArtifactRegistry(
            artifact_root=settings.artifact_root,
            capacity=settings.artifact_registry_capacity,
            ttl_seconds=settings.artifact_registry_ttl_seconds,
        )
        app.state.registry = registry
    else:
        registry = app.state.registry

    # 1. Verify Core git revision
    is_verified, current_sha = check_core_revision(
        core_repo_path=settings.core_repo_path,
        expected_revision=settings.expected_core_revision,
    )
    app.state.core_revision_verified = is_verified
    app.state.core_current_revision = current_sha
    logger.info(
        "Core revision verification: expected=%s, current=%s, verified=%s",
        settings.expected_core_revision,
        current_sha,
        is_verified,
    )

    # 2. Verify Core import origin (imported package == configured Core repo)
    import_origin_verified, import_origin_method = check_core_import_origin(
        core_repo_path=settings.core_repo_path,
        expected_revision=settings.expected_core_revision,
    )
    logger.info(
        "Core import origin verification: verified=%s, method=%s",
        import_origin_verified,
        import_origin_method,
    )

    # 3. Check calibration status from YAML
    cal_status, is_calibrated = get_yaml_calibration_status(settings.pipeline_config_path)
    logger.info(
        "Pipeline config calibration check: status=%s, calibrated=%s", cal_status, is_calibrated
    )

    # 4. Authoritative Artifact storage readiness preflight (safe probe, no unhandled exceptions)
    art_storage_status, art_storage_err = check_artifact_storage(settings.artifact_root)
    logger.info(
        "Artifact storage preflight: status=%s, error=%s", art_storage_status, art_storage_err
    )

    # 5. Check OCR runtime & OCR model asset
    ocr_runtime_status, ocr_rt_err = check_ocr_runtime()
    ocr_asset_status, ocr_asset_err = check_ocr_asset(settings.ocr_model_path)

    # 6. Load PipelineConfig and derive Detector readiness from config policy
    loaded_config = None
    det_asset_status = "NOT_EVALUATED"
    det_asset_err: str | None = None
    config_load_err: str | None = None

    try:
        from meter_reading_inference_service.core import load_pipeline_config_from_yaml

        loaded_config = load_pipeline_config_from_yaml(
            yaml_path=settings.pipeline_config_path,
            ocr_model_path_override=settings.ocr_model_path if settings.ocr_model_path else None,
            yolo_model_path_override=settings.yolo_model_path if settings.yolo_model_path else None,
            artifact_root_override=settings.artifact_root,
        )
        if loaded_config.detector:
            det_expected_sha = loaded_config.detector.expected_model_sha256
            det_path = loaded_config.detector.model_path or settings.yolo_model_path
            det_asset_status, det_asset_err = check_detector_asset(
                det_path, expected_sha256=det_expected_sha
            )
        else:
            det_asset_status = "NOT_CONFIGURED"
    except Exception as exc:
        config_load_err = str(exc)
        logger.warning("PipelineConfig load failed during preflight: %s", exc)
        det_asset_status = "NOT_EVALUATED"

    logger.info(
        "Readiness preflight: OCR runtime=%s, OCR asset=%s, Detector asset=%s",
        ocr_runtime_status,
        ocr_asset_status,
        det_asset_status,
    )

    # 7. Initialize Core Pipeline singleton if not already present
    if not hasattr(app.state, "coordinator") or app.state.coordinator is None:
        pipeline = None
        readiness_err = None

        if not import_origin_verified:
            readiness_err = "Core package import origin could not be verified against the configured Core repository"
        elif not is_verified:
            readiness_err = f"Core git revision verification failed (current: {current_sha}, expected: {settings.expected_core_revision})"
        elif not is_calibrated:
            readiness_err = (
                f"Pipeline configuration is marked {cal_status} (not calibrated for real inference)"
            )
        elif art_storage_status != "AVAILABLE":
            readiness_err = f"Artifact storage is not available: {art_storage_err}"
        elif ocr_runtime_status != "AVAILABLE":
            readiness_err = f"OCR Python runtime is unavailable or incompatible: {ocr_rt_err}"
        elif ocr_asset_status != "AVAILABLE":
            readiness_err = (
                f"Configured local OCR model bundle is missing or invalid: {ocr_asset_err}"
            )
        elif loaded_config is None:
            readiness_err = f"Pipeline configuration failed to load: {config_load_err}"
        else:
            try:
                from meter_reading_engine.pipeline import MeterReadingPipeline

                pipeline = MeterReadingPipeline(config=loaded_config)
                logger.info("MeterReadingPipeline successfully constructed.")
            except Exception as exc:
                readiness_err = f"Pipeline construction failed: {exc}"
                logger.warning("Pipeline construction failed: %s", exc)

        coordinator = InferenceCoordinator(
            pipeline=pipeline,
            registry=registry,
            settings=settings,
            core_revision_verified=is_verified,
            core_current_revision=current_sha,
            import_origin_verified=import_origin_verified,
            is_calibrated=is_calibrated,
            calibration_status=cal_status,
            ocr_runtime_status=ocr_runtime_status,
            ocr_asset_status=ocr_asset_status,
            detector_asset_status=det_asset_status,
            artifact_storage_status=art_storage_status,
            readiness_reason=readiness_err,
        )
        app.state.pipeline = pipeline
        app.state.coordinator = coordinator
    else:
        logger.info("Pipeline coordinator already injected.")

    yield

    logger.info("Shutting down meter-reading-inference-service...")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory for FastAPI app."""
    app_settings = settings or Settings()

    app = FastAPI(
        title="Meter Reading Inference Service",
        description="FastAPI Inference and Inspection Adapter for frozen meter-reading-engine-v2",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.settings = app_settings

    # Request size limit guard
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=app_settings.max_upload_bytes)

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    register_exception_handlers(app)

    # Routes
    app.include_router(router)

    return app


app = create_app()
