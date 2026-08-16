"""Configuration and runtime settings for meter-reading-inference-service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the inference service."""

    # Path to YAML pipeline config
    pipeline_config_path: Path = Path("config/pipeline.demo.example.yaml")

    # Path to frozen Core repository (for revision verification)
    core_repo_path: Path = Path(r"D:\Projects\meter-reading-engine-v2")

    # Expected frozen Core git commit sha
    expected_core_revision: str = "b185479f023eb8f0aeb8330183b4e2c564c4a465"

    # Optional model path overrides
    ocr_model_path: str = ""
    yolo_model_path: str = r"D:\Models\meter-reading-engine\lcd-reading-value-yolo-v1\best.pt"

    # Artifact storage and cache
    artifact_root: Path = Path("local_artifacts")
    artifact_registry_capacity: int = 256
    artifact_registry_ttl_seconds: int = 86400  # 24 hours

    # Limits and concurrency
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_inference_concurrency: int = 1
    enable_learned_primary: bool = False

    # CORS
    allowed_origins: list[str] = ["http://localhost:3000"]

    # Explicit default localization profiles mapping: meter_type -> localization_profile_id
    default_localization_profiles: dict[str, str] = {
        "VSEE_VSE3T": "vsee_vse3t_station_a_v1",
        "GELEX_ME40": "gelex_me40_station_b_v1",
    }

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("default_localization_profiles", mode="before")
    @classmethod
    def _parse_default_profiles(cls, v: Any) -> dict[str, str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("{"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
        return v
