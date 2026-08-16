"""HTTP and domain error definitions for the inference service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    """Base application exception with machine-readable code and HTTP status."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class InvalidRequestError(AppError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="INVALID_REQUEST",
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class LearnedPrimaryDisabledError(AppError):
    def __init__(
        self,
        message: str = "LEARNED_PRIMARY mode is disabled in service configuration",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="LEARNED_PRIMARY_DISABLED",
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
        )


class ArtifactNotFoundError(AppError):
    def __init__(
        self, message: str = "Artifact not found", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            code="ARTIFACT_NOT_FOUND",
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class UploadTooLargeError(AppError):
    def __init__(
        self,
        message: str = "Uploaded image exceeds maximum allowed size",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="UPLOAD_TOO_LARGE",
            message=message,
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            details=details,
        )


class UnsupportedMediaTypeError(AppError):
    def __init__(
        self,
        message: str = "Unsupported media type. Allowed types: image/jpeg, image/png",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="UNSUPPORTED_MEDIA_TYPE",
            message=message,
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            details=details,
        )


class InferenceBusyError(AppError):
    def __init__(
        self,
        message: str = "Inference service is busy, please retry later",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="INFERENCE_BUSY",
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
        )


class PipelineNotReadyError(AppError):
    def __init__(
        self, message: str = "Pipeline is not ready", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            code="PIPELINE_NOT_READY",
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


class InternalServiceError(AppError):
    def __init__(
        self, message: str = "Internal server error occurred", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            code="INTERNAL_SERVICE_ERROR",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


def format_error_response(
    code: str, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Register uniform error handlers for FastAPI application."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=format_error_response(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=format_error_response(
                code="INVALID_REQUEST",
                message="Request validation failed",
                details={
                    "validation_errors": [
                        {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
                        for e in errors
                    ]
                },
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code_map = {
            400: "INVALID_REQUEST",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "UPLOAD_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            429: "INFERENCE_BUSY",
            503: "PIPELINE_NOT_READY",
        }
        code = code_map.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content=format_error_response(code=code, message=str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=format_error_response(
                code="INTERNAL_SERVICE_ERROR",
                message="An unexpected error occurred processing the request",
                details={},
            ),
        )
