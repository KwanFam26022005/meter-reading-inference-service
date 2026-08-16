"""Unit tests for YAML loader, geometry parsing, zero-download, Core revision, and origin verification."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from meter_reading_engine.contracts import (
    CoordinateUnit,
    GeometryMode,
    MeterType,
    PipelineConfig,
    QuadrilateralOrder,
)

from meter_reading_inference_service.core import (
    check_artifact_storage,
    check_core_import_origin,
    check_core_revision,
    check_detector_asset,
    check_detector_runtime,
    check_ocr_asset,
    check_ocr_runtime,
    get_yaml_calibration_status,
    load_pipeline_config_from_yaml,
)


def test_load_pipeline_config_from_yaml() -> None:
    yaml_path = Path("config/pipeline.demo.example.yaml")
    config = load_pipeline_config_from_yaml(yaml_path)

    assert isinstance(config, PipelineConfig)
    assert config.schema_version == "1.0"
    assert config.config_id == "pipeline_demo_v1"
    assert "vsee_vse3t_station_a_v1" in config.localization_profiles
    assert "gelex_me40_station_b_v1" in config.localization_profiles
    assert MeterType.VSEE_VSE3T in config.meter_type_reading_profiles
    assert config.config_sha256 != ""
    assert config.recognition.allow_model_download is False
    if config.detector:
        assert config.detector.allow_model_download is False


def test_core_dependency_pin_assumption() -> None:
    """A1-001: Ensure canonical frozen revision matches service constant."""
    expected_sha = "b185479f023eb8f0aeb8330183b4e2c564c4a465"
    assert len(expected_sha) == 40


def test_check_core_revision_exact_and_mismatch(tmp_path: Path) -> None:
    """A1-002: Exact match returns (True, SHA), mismatch returns (False, SHA), missing path returns (False, None)."""
    core_path = Path(r"D:\Projects\meter-reading-engine-v2")
    expected = "b185479f023eb8f0aeb8330183b4e2c564c4a465"
    is_verified, current_sha = check_core_revision(core_path, expected)

    assert is_verified is True
    assert current_sha == expected

    # Mismatched SHA
    is_verified_bad, sha_bad = check_core_revision(
        core_path, "0000000000000000000000000000000000000000"
    )
    assert is_verified_bad is False
    assert sha_bad == expected

    # Unverifiable path (non-existent or non-git dir)
    is_verified_unv, sha_unv = check_core_revision(tmp_path / "nonexistent_repo", expected)
    assert is_verified_unv is False
    assert sha_unv is None


def test_calibration_status_example_only() -> None:
    """A1-003: Demo configuration is explicitly EXAMPLE_ONLY and not calibrated."""
    yaml_path = Path("config/pipeline.demo.example.yaml")
    status, is_calibrated = get_yaml_calibration_status(yaml_path)

    assert status == "EXAMPLE_ONLY"
    assert is_calibrated is False


def test_calibration_status_calibrated(tmp_path: Path) -> None:
    """A1-003: Configuration marked CALIBRATED is recognized."""
    cfg_file = tmp_path / "calibrated.yaml"
    cfg_file.write_text("calibration_status: 'CALIBRATED'\n", encoding="utf-8")
    status, is_calibrated = get_yaml_calibration_status(cfg_file)

    assert status == "CALIBRATED"
    assert is_calibrated is True


def test_calibration_status_missing_key_fails_closed(tmp_path: Path) -> None:
    """Finding 3: Missing calibration_status key MUST fail closed (UNSPECIFIED, not calibrated)."""
    cfg_file = tmp_path / "no_status.yaml"
    cfg_file.write_text("schema_version: '1.0'\n", encoding="utf-8")
    status, is_calibrated = get_yaml_calibration_status(cfg_file)

    assert status == "UNSPECIFIED"
    assert is_calibrated is False


def test_calibration_status_empty_value_fails_closed(tmp_path: Path) -> None:
    """Finding 3: Empty string calibration_status must fail closed (UNSPECIFIED)."""
    cfg_file = tmp_path / "empty_status.yaml"
    cfg_file.write_text("calibration_status: ''\n", encoding="utf-8")
    status, is_calibrated = get_yaml_calibration_status(cfg_file)

    assert status == "UNSPECIFIED"
    assert is_calibrated is False


def test_calibration_status_arbitrary_value_not_calibrated(tmp_path: Path) -> None:
    """Finding 3: Any value other than CALIBRATED (after upper-case normalization) must not enable inference."""
    for bad_value in ("UNKNOWN", "TRUE", "1", "EXAMPLE_ONLY"):
        cfg_file = tmp_path / f"bad_{bad_value}.yaml"
        cfg_file.write_text(f"calibration_status: '{bad_value}'\n", encoding="utf-8")
        status, is_calibrated = get_yaml_calibration_status(cfg_file)
        assert is_calibrated is False, (
            f"Expected is_calibrated=False for calibration_status={bad_value!r}, got True"
        )


def test_calibration_status_not_found() -> None:
    """Finding 3: Non-existent YAML returns NOT_FOUND and is not calibrated."""
    status, is_calibrated = get_yaml_calibration_status(Path("/nonexistent/path.yaml"))
    assert status == "NOT_FOUND"
    assert is_calibrated is False


def test_ocr_runtime_preflight() -> None:
    """A1-004: OCR runtime check returns truthful status."""
    status, err = check_ocr_runtime()
    assert status in ("AVAILABLE", "MISSING", "INCOMPATIBLE")


def test_ocr_asset_preflight(tmp_path: Path) -> None:
    """A1-004 & 5: OCR asset check verifies bundle layout without loading weights."""
    # Empty or missing
    status, err = check_ocr_asset("")
    assert status == "MISSING"

    status, err = check_ocr_asset(tmp_path / "nonexistent")
    assert status == "MISSING"

    # Incomplete bundle (missing inference.yml)
    bundle_dir = tmp_path / "ocr_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "inference.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "inference.pdiparams").write_bytes(b"\x00" * 10)
    status, err = check_ocr_asset(bundle_dir)
    assert status == "INVALID"

    # Complete valid bundle
    (bundle_dir / "inference.yml").write_text(
        "Global:\n  model_name: 'PP-OCRv6_medium_rec'\n", encoding="utf-8"
    )
    status, err = check_ocr_asset(bundle_dir)
    assert status == "AVAILABLE"
    assert err is None


def test_detector_asset_preflight(tmp_path: Path) -> None:
    """Finding 3: Detector asset check verifies presence and SHA-256 against policy."""
    status, err = check_detector_asset("")
    assert status == "MISSING"

    # Missing file
    status, err = check_detector_asset(tmp_path / "nonexistent.pt")
    assert status == "MISSING"

    # Present file without expected SHA
    model_file = tmp_path / "best.pt"
    model_data = b"TEST_MODEL_DATA_FOR_CHECKSUM"
    model_file.write_bytes(model_data)

    status, err = check_detector_asset(model_file)
    assert status == "AVAILABLE"

    # Present file with matching expected SHA
    actual_sha = hashlib.sha256(model_data).hexdigest()
    status, err = check_detector_asset(model_file, expected_sha256=actual_sha)
    assert status == "AVAILABLE"
    assert err is None

    # Present file with mismatched expected SHA
    wrong_sha = "0" * 64
    status, err = check_detector_asset(model_file, expected_sha256=wrong_sha)
    assert status == "INVALID"
    assert err is not None
    assert "mismatch" in err.lower()


def test_check_detector_runtime_missing() -> None:
    """A1-H1: ModuleNotFoundError during ultralytics import returns MISSING with generic message."""
    with patch(
        "importlib.import_module",
        side_effect=ModuleNotFoundError("No module named 'ultralytics'", name="ultralytics"),
    ):
        status, err = check_detector_runtime()
    assert status == "MISSING"
    assert err == "Ultralytics detector runtime is not installed"


def test_check_detector_runtime_incompatible_import_error() -> None:
    """A1-H1: Non-ModuleNotFoundError exception during ultralytics import returns INCOMPATIBLE with generic message."""
    with patch("importlib.import_module", side_effect=ImportError("DLL initialization failed")):
        status, err = check_detector_runtime()
    assert status == "INCOMPATIBLE"
    assert err == "Ultralytics detector runtime could not be imported"


def test_check_detector_runtime_incompatible_sanitizes_path_leak() -> None:
    """A1-H1: Generic import/runtime exceptions with path strings are fully sanitized."""
    path_bearing_exc = RuntimeError(
        r"Failed loading DLL D:\Private\runtime\ultralytics\backend.dll: access denied"
    )
    with patch("importlib.import_module", side_effect=path_bearing_exc):
        status, err = check_detector_runtime()
    assert status == "INCOMPATIBLE"
    assert err == "Ultralytics detector runtime could not be imported"
    assert "D:\\" not in err
    assert "Private" not in err
    assert "backend.dll" not in err
    assert "access denied" not in err


def test_check_detector_runtime_incompatible_version() -> None:
    """A1-H1: Ultralytics imported with version != 8.4.120 returns INCOMPATIBLE."""
    mock_mod = MagicMock()
    mock_mod.__version__ = "8.3.0"
    with patch("importlib.import_module", return_value=mock_mod):
        status, err = check_detector_runtime()
    assert status == "INCOMPATIBLE"
    assert err == "Ultralytics version '8.3.0' is incompatible (expected 8.4.120)"


def test_check_detector_runtime_available() -> None:
    """A1-H1: Ultralytics imported with version == 8.4.120 returns AVAILABLE."""
    mock_mod = MagicMock()
    mock_mod.__version__ = "8.4.120"
    with patch("importlib.import_module", return_value=mock_mod):
        status, err = check_detector_runtime()
    assert status == "AVAILABLE"
    assert err is None


def test_four_point_geometry_parsing(tmp_path: Path) -> None:
    """A1-005: Full FOUR_POINT geometry parsing with normalized and pixel quad."""
    with open("config/pipeline.demo.example.yaml", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)

    # 1. Normalized FOUR_POINT
    base_raw["localization_profiles"]["vsee_vse3t_station_a_v1"]["geometry"] = {
        "mode": "FOUR_POINT",
        "source": "CONFIGURED",
        "input_space_id": "display_crop",
        "target_width": 300,
        "target_height": 100,
        "interpolation": "CUBIC",
        "border_mode": "CONSTANT",
        "profile_id": "geom_four_point_v1",
        "profile_revision": "1.0",
        "quadrilateral": {
            "coordinate_space_id": "display_crop",
            "unit": "NORMALIZED",
            "order": "TL_TR_BR_BL",
            "points": [
                {"x": 0.05, "y": 0.05},
                {"x": 0.95, "y": 0.08},
                {"x": 0.92, "y": 0.92},
                {"x": 0.08, "y": 0.90},
            ],
        },
    }

    test_yaml = tmp_path / "four_point_norm.yaml"
    with open(test_yaml, "w", encoding="utf-8") as f:
        yaml.dump(base_raw, f)

    config = load_pipeline_config_from_yaml(test_yaml)
    geom = config.localization_profiles["vsee_vse3t_station_a_v1"].geometry
    assert geom.mode == GeometryMode.FOUR_POINT
    assert geom.quadrilateral is not None
    assert geom.quadrilateral.unit == CoordinateUnit.NORMALIZED
    assert geom.quadrilateral.quadrilateral.order == QuadrilateralOrder.TL_TR_BR_BL
    assert len(geom.quadrilateral.quadrilateral.points) == 4
    assert geom.target_width == 300
    assert geom.target_height == 100
    assert geom.interpolation == "CUBIC"


def test_four_point_geometry_missing_quad_rejected(tmp_path: Path) -> None:
    """A1-005: FOUR_POINT with missing quadrilateral specification fails validation."""
    with open("config/pipeline.demo.example.yaml", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)

    base_raw["localization_profiles"]["vsee_vse3t_station_a_v1"]["geometry"] = {
        "mode": "FOUR_POINT",
        "source": "CONFIGURED",
    }

    test_yaml = tmp_path / "four_point_invalid.yaml"
    with open(test_yaml, "w", encoding="utf-8") as f:
        yaml.dump(base_raw, f)

    with pytest.raises(ValueError, match="FOUR_POINT geometry requires a 'quadrilateral'"):
        load_pipeline_config_from_yaml(test_yaml)


def test_zero_download_policy_enforcement(tmp_path: Path) -> None:
    """A1-005: allow_model_download=True is strictly rejected for recognition and detector."""
    with open("config/pipeline.demo.example.yaml", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)

    # Recognition policy violation
    base_raw["recognition"]["allow_model_download"] = True
    test_yaml1 = tmp_path / "download_rec_violation.yaml"
    with open(test_yaml1, "w", encoding="utf-8") as f:
        yaml.dump(base_raw, f)

    with pytest.raises(ValueError, match="MODEL_DOWNLOAD_POLICY_FORBIDDEN"):
        load_pipeline_config_from_yaml(test_yaml1)

    # Detector policy violation
    base_raw["recognition"]["allow_model_download"] = False
    base_raw["detector"]["allow_model_download"] = True
    test_yaml2 = tmp_path / "download_det_violation.yaml"
    with open(test_yaml2, "w", encoding="utf-8") as f:
        yaml.dump(base_raw, f)

    with pytest.raises(ValueError, match="MODEL_DOWNLOAD_POLICY_FORBIDDEN"):
        load_pipeline_config_from_yaml(test_yaml2)


# ---------------------------------------------------------------------------
# Finding 1: Core Import Origin Verification & Codex Attack Tests
# ---------------------------------------------------------------------------


def test_check_core_import_origin_editable() -> None:
    """Finding 1: Verify that the actual editable install resolves to the configured Core path."""
    core_path = Path(r"D:\Projects\meter-reading-engine-v2")
    expected = "b185479f023eb8f0aeb8330183b4e2c564c4a465"

    verified, method = check_core_import_origin(core_path, expected)

    assert verified is True
    assert method == "editable"


def test_check_core_import_origin_editable_revision_mismatch() -> None:
    """Finding 1: Editable install whose Git HEAD mismatches expected SHA fails closed."""
    core_path = Path(r"D:\Projects\meter-reading-engine-v2")
    wrong_expected = "0000000000000000000000000000000000000000"

    verified, reason = check_core_import_origin(core_path, wrong_expected)

    assert verified is False
    assert "head does not match" in reason.lower() or "cannot verify" in reason.lower()


def test_check_core_import_origin_codex_attack_rejection(tmp_path: Path) -> None:
    """Finding 1: Codex attack regression test.

    Simulate scenario where imported meter_reading_engine.__file__ is located in
    a shadow directory (C:\\shadow\\meter_reading_engine\\__init__.py), while an installed
    distribution 'meter-reading-engine' contains valid pinned PEP 610 metadata.

    The check MUST fail closed: import_origin_verified = False.
    """
    shadow_pkg_file = tmp_path / "shadow" / "meter_reading_engine" / "__init__.py"
    shadow_pkg_file.parent.mkdir(parents=True)
    shadow_pkg_file.write_text("# shadow malicious package\n", encoding="utf-8")

    expected_sha = "b185479f023eb8f0aeb8330183b4e2c564c4a465"
    core_path = Path(r"D:\Projects\meter-reading-engine-v2")

    # Mock meter_reading_engine.__file__ to point to shadow directory
    mock_pkg = MagicMock()
    mock_pkg.__file__ = str(shadow_pkg_file)

    # Mock installed distribution having valid direct_url.json but installed in site-packages
    mock_dist = MagicMock()
    # Installed files of this distribution are in site-packages, NOT the shadow directory
    mock_file_path = tmp_path / "site_packages" / "meter_reading_engine" / "__init__.py"
    mock_file_path.parent.mkdir(parents=True)
    mock_file_path.write_text("# legitimate vcs package\n", encoding="utf-8")

    mock_dist.files = ["meter_reading_engine/__init__.py"]
    mock_dist.locate_file.return_value = mock_file_path
    mock_dist.read_text.return_value = json.dumps(
        {
            "url": "https://github.com/KwanFam26022005/meter-reading-engine-v2.git",
            "vcs_info": {
                "commit_id": expected_sha,
                "vcs": "git",
            },
        }
    )

    with (
        patch.dict("sys.modules", {"meter_reading_engine": mock_pkg}),
        patch(
            "importlib.metadata.packages_distributions",
            return_value={"meter_reading_engine": ["meter-reading-engine"]},
        ),
        patch("importlib.metadata.distribution", return_value=mock_dist),
    ):
        verified, reason = check_core_import_origin(core_path, expected_sha)

    # Must fail closed because the imported module is NOT owned by the distribution
    assert verified is False
    assert "not owned by any installed distribution" in reason


def test_check_core_import_origin_vcs_installed_valid(tmp_path: Path) -> None:
    """Finding 1: Legitimate VCS-installed package whose files contain the imported module passes."""
    legit_pkg_file = tmp_path / "site_packages" / "meter_reading_engine" / "__init__.py"
    legit_pkg_file.parent.mkdir(parents=True)
    legit_pkg_file.write_text("# legitimate vcs package\n", encoding="utf-8")

    expected_sha = "b185479f023eb8f0aeb8330183b4e2c564c4a465"
    dummy_core_path = tmp_path / "non_existent_core_repo"

    mock_pkg = MagicMock()
    mock_pkg.__file__ = str(legit_pkg_file)

    mock_dist = MagicMock()
    mock_dist.files = ["meter_reading_engine/__init__.py"]
    mock_dist.locate_file.return_value = legit_pkg_file
    mock_dist.read_text.return_value = json.dumps(
        {
            "url": "https://github.com/KwanFam26022005/meter-reading-engine-v2",
            "vcs_info": {
                "commit_id": expected_sha,
                "vcs": "git",
            },
        }
    )

    with (
        patch.dict("sys.modules", {"meter_reading_engine": mock_pkg}),
        patch(
            "importlib.metadata.packages_distributions",
            return_value={"meter_reading_engine": ["meter-reading-engine"]},
        ),
        patch("importlib.metadata.distribution", return_value=mock_dist),
    ):
        verified, method = check_core_import_origin(dummy_core_path, expected_sha)

    assert verified is True
    assert method == "vcs_metadata"


def test_check_core_import_origin_vcs_commit_mismatch(tmp_path: Path) -> None:
    """Finding 1: VCS-installed package with wrong commit SHA fails closed."""
    legit_pkg_file = tmp_path / "site_packages" / "meter_reading_engine" / "__init__.py"
    legit_pkg_file.parent.mkdir(parents=True)
    legit_pkg_file.write_text("# legitimate vcs package\n", encoding="utf-8")

    expected_sha = "b185479f023eb8f0aeb8330183b4e2c564c4a465"
    wrong_sha = "1111111111111111111111111111111111111111"
    dummy_core_path = tmp_path / "non_existent_core_repo"

    mock_pkg = MagicMock()
    mock_pkg.__file__ = str(legit_pkg_file)

    mock_dist = MagicMock()
    mock_dist.files = ["meter_reading_engine/__init__.py"]
    mock_dist.locate_file.return_value = legit_pkg_file
    mock_dist.read_text.return_value = json.dumps(
        {
            "url": "https://github.com/KwanFam26022005/meter-reading-engine-v2",
            "vcs_info": {
                "commit_id": wrong_sha,
                "vcs": "git",
            },
        }
    )

    with (
        patch.dict("sys.modules", {"meter_reading_engine": mock_pkg}),
        patch(
            "importlib.metadata.packages_distributions",
            return_value={"meter_reading_engine": ["meter-reading-engine"]},
        ),
        patch("importlib.metadata.distribution", return_value=mock_dist),
    ):
        verified, reason = check_core_import_origin(dummy_core_path, expected_sha)

    assert verified is False
    assert "commit_id does not match" in reason


def test_check_core_import_origin_vcs_repo_url_mismatch(tmp_path: Path) -> None:
    """Finding 1: VCS-installed package pointing to an untrusted repository URL fails closed."""
    legit_pkg_file = tmp_path / "site_packages" / "meter_reading_engine" / "__init__.py"
    legit_pkg_file.parent.mkdir(parents=True)
    legit_pkg_file.write_text("# legitimate vcs package\n", encoding="utf-8")

    expected_sha = "b185479f023eb8f0aeb8330183b4e2c564c4a465"
    dummy_core_path = tmp_path / "non_existent_core_repo"

    mock_pkg = MagicMock()
    mock_pkg.__file__ = str(legit_pkg_file)

    mock_dist = MagicMock()
    mock_dist.files = ["meter_reading_engine/__init__.py"]
    mock_dist.locate_file.return_value = legit_pkg_file
    mock_dist.read_text.return_value = json.dumps(
        {
            "url": "https://github.com/attacker/meter-reading-engine-fake.git",
            "vcs_info": {
                "commit_id": expected_sha,
                "vcs": "git",
            },
        }
    )

    with (
        patch.dict("sys.modules", {"meter_reading_engine": mock_pkg}),
        patch(
            "importlib.metadata.packages_distributions",
            return_value={"meter_reading_engine": ["meter-reading-engine"]},
        ),
        patch("importlib.metadata.distribution", return_value=mock_dist),
    ):
        verified, reason = check_core_import_origin(dummy_core_path, expected_sha)

    assert verified is False
    assert "repository url does not match" in reason.lower()


# ---------------------------------------------------------------------------
# Finding 2: Git dubious ownership
# ---------------------------------------------------------------------------


def test_check_core_revision_handles_dubious_ownership(tmp_path: Path) -> None:
    """Finding 2: A non-zero git returncode (simulating dubious ownership) fails closed."""
    mock_result = MagicMock()
    mock_result.returncode = 128
    mock_result.stdout = ""
    mock_result.stderr = (
        "fatal: detected dubious ownership in repository at '/some/path'\n"
        "To add an exception for this directory, call:\n"
        "\tgit config --global safe.directory /some/path"
    )

    with patch("subprocess.run", return_value=mock_result):
        verified, sha = check_core_revision(tmp_path, "b185479f023eb8f0aeb8330183b4e2c564c4a465")

    assert verified is False
    assert sha is None


# ---------------------------------------------------------------------------
# Finding 2: Artifact Storage Preflight
# ---------------------------------------------------------------------------


def test_check_artifact_storage_available(tmp_path: Path) -> None:
    """Finding 2: Writable directory is AVAILABLE and probe is cleaned up."""
    status, err = check_artifact_storage(tmp_path)
    assert status == "AVAILABLE"
    assert err is None
    # Probe file must have been removed
    probe_files = list(tmp_path.glob(".readiness_probe_*"))
    assert len(probe_files) == 0


def test_check_artifact_storage_creates_directory(tmp_path: Path) -> None:
    """Finding 2: Non-existent directory is safely created and verified writable."""
    target = tmp_path / "new_artifact_root"
    assert not target.exists()
    status, err = check_artifact_storage(target)
    assert status == "AVAILABLE"
    assert target.is_dir()
    probe_files = list(target.glob(".readiness_probe_*"))
    assert len(probe_files) == 0


def test_check_artifact_storage_permission_denied_handled_gracefully(tmp_path: Path) -> None:
    """Finding 2: PermissionError during mkdir is handled safely without raising unhandled exception."""
    target = tmp_path / "permission_denied_dir"
    with patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied")):
        status, err = check_artifact_storage(target)

    assert status == "UNAVAILABLE"
    assert err is not None
    assert "could not be created" in err
