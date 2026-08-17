"""Unit tests for service configuration and settings."""

from meter_reading_inference_service.settings import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.expected_core_revision == "9816d9a9f764460f3583dc18251eca2d953b9af6"
    assert settings.max_inference_concurrency == 1
    assert not settings.enable_learned_primary
    assert "VSEE_VSE3T" in settings.default_localization_profiles
    assert "GELEX_ME40" in settings.default_localization_profiles


def test_settings_json_parsing() -> None:
    settings = Settings(
        allowed_origins='["http://localhost:3000", "http://127.0.0.1:3000"]',
        default_localization_profiles='{"VSEE_VSE3T": "custom_profile_1"}',
    )
    assert settings.allowed_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    assert settings.default_localization_profiles == {"VSEE_VSE3T": "custom_profile_1"}
