"""Core integration adapter, YAML configuration loader, and revision verification."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as importlib_meta
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from meter_reading_engine.contracts import (
    ArtifactPolicy,
    CoordinateUnit,
    DecimalPolicy,
    DecisionPolicy,
    DetectorPolicy,
    GeometryInstruction,
    GeometryMode,
    GeometrySource,
    LeadingZeroPolicy,
    LocalizationProfile,
    MeterType,
    NormalizedBBox,
    PipelineConfig,
    PixelBBox,
    Point,
    Quadrilateral,
    QuadrilateralOrder,
    QuadSpec,
    QualityPolicy,
    ReadingProfile,
    RecognitionPolicy,
    RegionSpec,
    ValidationProfile,
    freeze_config,
)
from meter_reading_engine.pipeline import MeterReadingPipeline

from meter_reading_inference_service.errors import PipelineNotReadyError
from meter_reading_inference_service.settings import Settings

logger = logging.getLogger(__name__)

# Canonical Core repository URL expected by the service
_EXPECTED_CORE_REPO_URL = "https://github.com/KwanFam26022005/meter-reading-engine-v2"


def check_core_revision(core_repo_path: Path, expected_revision: str) -> tuple[bool, str | None]:
    """Verify that Core repository Git HEAD matches the canonical frozen revision.

    Returns (is_verified, current_sha).
    Fails closed on any git or ownership error.
    """
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(core_repo_path),
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if res.returncode != 0:
            stderr = res.stderr.strip()
            logger.warning("git rev-parse failed (rc=%s): %s", res.returncode, stderr)
            # Surface dubious-ownership hint without leaking full path
            if "dubious ownership" in stderr or "safe.directory" in stderr:
                logger.warning(
                    "Git dubious ownership detected for Core repository. "
                    "Run: git config --global safe.directory %s",
                    str(core_repo_path),
                )
            return False, None
        current_sha = res.stdout.strip()
        is_verified = current_sha.lower() == expected_revision.lower()
        return is_verified, current_sha
    except Exception as exc:
        logger.warning("Could not verify Core revision via git: %s", exc)
        return False, None


def check_core_import_origin(
    core_repo_path: Path,
    expected_revision: str,
) -> tuple[bool, str]:
    """Verify that the imported meter_reading_engine package resolves from the
    configured CORE_REPO_PATH (editable install) or from trustworthy PEP 610
    VCS metadata of the specific distribution that actually provides the imported files.

    Returns (import_origin_verified, reason_str).
    Never exposes absolute package paths through the returned reason string that
    would be surfaced via HTTP.
    """
    try:
        import meter_reading_engine as _engine_pkg
    except ImportError:
        return False, "meter_reading_engine package is not importable"

    pkg_file = getattr(_engine_pkg, "__file__", None)
    if not pkg_file:
        return False, "meter_reading_engine has no __file__ attribute"

    pkg_path = Path(pkg_file).resolve()
    core_root = Path(core_repo_path).resolve()

    # --- Strategy 1: Editable local install ---
    # The package file must be contained under the configured Core source tree.
    try:
        pkg_path.relative_to(core_root)
        # Verify the Git revision of THAT exact checkout
        is_verified, current_sha = check_core_revision(core_root, expected_revision)
        if is_verified:
            return True, "editable"
        return False, "Editable Core repository HEAD does not match expected revision"
    except ValueError:
        pass  # Not under core_root — evaluate installed distribution ownership

    # --- Strategy 2: PEP 610 direct_url.json VCS metadata bound to owning distribution ---
    try:
        # 1. Identify which installed distributions claim to provide meter_reading_engine
        pkg_dists = importlib_meta.packages_distributions().get("meter_reading_engine", [])
        if not pkg_dists:
            return (
                False,
                "No installed distribution provides the imported meter_reading_engine package",
            )

        # 2. Find distributions that physically contain the imported pkg_path
        owning_distributions = []
        for dist_name in set(pkg_dists):
            try:
                dist = importlib_meta.distribution(dist_name)
                if dist.files:
                    for df in dist.files:
                        try:
                            if dist.locate_file(df).resolve() == pkg_path:
                                owning_distributions.append(dist)
                                break
                        except Exception:
                            continue
            except Exception:
                continue

        if not owning_distributions:
            return (
                False,
                "Imported meter_reading_engine is not owned by any installed distribution",
            )

        if len(owning_distributions) > 1:
            return (
                False,
                "Ambiguous package ownership: multiple distributions contain the imported package",
            )

        owning_dist = owning_distributions[0]

        # 3. Read direct_url.json ONLY from the verified owning distribution
        direct_url_text = owning_dist.read_text("direct_url.json")
        if not direct_url_text:
            return (
                False,
                "Owning distribution does not contain direct_url.json metadata",
            )

        import json

        du = json.loads(direct_url_text)
        vcs_info = du.get("vcs_info")
        if not isinstance(vcs_info, dict) or not vcs_info:
            return (
                False,
                "Owning distribution direct_url.json has no valid vcs_info block",
            )

        commit_id = str(vcs_info.get("commit_id", "")).strip().lower()
        requested_rev = str(vcs_info.get("requested_revision", "")).strip().lower()
        expected_rev = expected_revision.strip().lower()

        if commit_id != expected_rev and requested_rev != expected_rev:
            return (
                False,
                "PEP 610 VCS metadata commit_id does not match expected Core revision",
            )

        # 4. Normalize and verify repository URL
        url = str(du.get("url", "")).strip()
        if not url:
            return False, "PEP 610 VCS metadata missing repository URL"

        def _normalize_url(u: str) -> tuple[str, str]:
            parsed = urlparse(u)
            netloc = parsed.netloc.lower()
            path = parsed.path.rstrip("/").removesuffix(".git").lower()
            return netloc, path

        if _normalize_url(url) != _normalize_url(_EXPECTED_CORE_REPO_URL):
            return (
                False,
                "PEP 610 VCS repository URL does not match expected Core repository origin",
            )

        return True, "vcs_metadata"

    except Exception as exc:
        logger.debug("PEP 610 direct_url.json check failed: %s", exc)

    return (
        False,
        "Cannot verify that the imported meter_reading_engine originates from the configured Core repository",
    )


def get_yaml_calibration_status(yaml_path: Path) -> tuple[str, bool]:
    """Read calibration_status from YAML file.

    Returns (calibration_status_str, is_calibrated).

    FAIL CLOSED: only an explicit exact value ``calibration_status: CALIBRATED``
    may produce ``is_calibrated = True``.  Missing, unknown, empty, or any other
    value produces ``is_calibrated = False``.

    Known typed states: CALIBRATED | EXAMPLE_ONLY | UNSPECIFIED | INVALID
    """
    if not yaml_path.is_file():
        return "NOT_FOUND", False
    try:
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw_value = raw.get("calibration_status")  # Default is None — intentionally missing
        if raw_value is None:
            # Missing key: fail closed as UNSPECIFIED
            return "UNSPECIFIED", False
        status = str(raw_value).upper().strip()
        if not status:
            return "UNSPECIFIED", False
        # Only the exact string CALIBRATED enables real inference
        is_calibrated = status == "CALIBRATED"
        return status, is_calibrated
    except Exception:
        return "INVALID", False


def check_ocr_runtime() -> tuple[str, str | None]:
    """Check if the required PaddleOCR runtime packages are installed with supported versions.

    Returns (runtime_status, error_message):
    status is 'AVAILABLE', 'MISSING', or 'INCOMPATIBLE'.
    """
    try:
        paddleocr_mod = importlib.import_module("paddleocr")
        paddlex_mod = importlib.import_module("paddlex")
        paddle_mod = importlib.import_module("paddle")
    except ModuleNotFoundError as exc:
        return "MISSING", f"Required OCR runtime package not installed: {exc.name}"
    except Exception as exc:
        return "MISSING", f"Failed to import OCR runtime: {exc}"

    ocr_ver = getattr(paddleocr_mod, "__version__", "unknown")
    if ocr_ver != "3.7.0":
        return "INCOMPATIBLE", f"PaddleOCR version {ocr_ver!r} is incompatible (expected 3.7.0)"

    px_ver = str(getattr(paddlex_mod, "__version__", "unknown"))
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", px_ver)
    if not match:
        return "INCOMPATIBLE", f"PaddleX version {px_ver!r} cannot be parsed"
    px_tuple = tuple(int(x) for x in match.groups())
    if not ((3, 7, 0) <= px_tuple < (3, 8, 0)):
        return (
            "INCOMPATIBLE",
            f"PaddleX version {px_ver!r} is incompatible (expected >=3.7.0,<3.8.0)",
        )

    pd_ver = str(getattr(paddle_mod, "__version__", "unknown"))
    if pd_ver != "3.3.1":
        return "INCOMPATIBLE", f"PaddlePaddle version {pd_ver!r} is incompatible (expected 3.3.1)"

    return "AVAILABLE", None


def check_ocr_asset(model_path: str | Path | None) -> tuple[str, str | None]:
    """Check local OCR model bundle layout without initializing weights.

    Returns (asset_status, error_message):
    status is 'AVAILABLE', 'MISSING', or 'INVALID'.
    """
    if not model_path:
        return "MISSING", "OCR model path is empty or not configured"

    p = Path(str(model_path).strip())
    if not p.exists() or not p.is_dir():
        return "MISSING", "OCR model directory does not exist"

    topology_candidates = {"inference.json", "inference.pdmodel"}
    param_candidates = {"inference.pdiparams", "inference.params"}
    config_file = p / "inference.yml"

    has_topo = any((p / f).is_file() and (p / f).stat().st_size > 0 for f in topology_candidates)
    has_param = any((p / f).is_file() and (p / f).stat().st_size > 0 for f in param_candidates)
    has_config = config_file.is_file() and config_file.stat().st_size > 0

    if not (has_topo and has_param and has_config):
        return "INVALID", "OCR model bundle is missing required files"

    try:
        with config_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "Global" not in data:
            return "INVALID", "inference.yml missing Global block"
        global_block = data["Global"]
        if (
            not isinstance(global_block, dict)
            or global_block.get("model_name") != "PP-OCRv6_medium_rec"
        ):
            return "INVALID", "inference.yml Global.model_name is not PP-OCRv6_medium_rec"
    except Exception:
        return "INVALID", "Failed to parse inference.yml"

    return "AVAILABLE", None


def check_detector_runtime() -> tuple[str, str | None]:
    """Check if the required Ultralytics detector runtime package is installed with supported version.

    Returns (runtime_status, error_message):
    status is 'AVAILABLE', 'MISSING', or 'INCOMPATIBLE'.
    """
    try:
        ultralytics_mod = importlib.import_module("ultralytics")
    except ModuleNotFoundError:
        return "MISSING", "Ultralytics detector runtime is not installed"
    except Exception as exc:
        logger.warning("Failed to import detector runtime: %s", exc)
        return "INCOMPATIBLE", "Ultralytics detector runtime could not be imported"

    det_ver = getattr(ultralytics_mod, "__version__", "unknown")
    if det_ver != "8.4.120":
        return "INCOMPATIBLE", f"Ultralytics version {det_ver!r} is incompatible (expected 8.4.120)"

    return "AVAILABLE", None


def check_detector_asset(
    model_path: str | Path | None,
    expected_sha256: str | None = None,
) -> tuple[str, str | None]:
    """Check detector weight file presence and SHA-256 against PipelineConfig policy.

    ``expected_sha256`` must come from ``PipelineConfig.detector.expected_model_sha256``
    (NOT from a hard-coded or environment-level override). When the policy requires
    a SHA and it cannot be verified, INVALID is returned.

    Returns (asset_status, error_message):
    status is 'AVAILABLE', 'MISSING', or 'INVALID'.

    Missing/invalid detector does NOT disable CONFIGURED inference — it only
    affects learned shadow telemetry readiness.
    """
    if not model_path:
        return "MISSING", "Detector model path is not configured"

    p = Path(str(model_path).strip())
    if not p.exists() or not p.is_file() or p.stat().st_size == 0:
        return "MISSING", "Detector model weight file does not exist"

    if expected_sha256:
        expected_sha256 = expected_sha256.strip().lower()
        try:
            h = hashlib.sha256()
            with p.open("rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            actual = h.hexdigest().lower()
            if actual != expected_sha256:
                return (
                    "INVALID",
                    "Detector model SHA-256 checksum mismatch (expected by PipelineConfig policy)",
                )
        except Exception:
            return "INVALID", "Failed to verify detector model checksum"

    return "AVAILABLE", None


def check_artifact_storage(artifact_root: Path) -> tuple[str, str | None]:
    """Authoritative readiness preflight verifying artifact root writability.

    Safely tests directory creation, write, flush, close, and cleanup with a
    probe file. Never raises unhandled exceptions. Fails closed with 'UNAVAILABLE'.
    Never leaks absolute local paths in the returned error message.
    """
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("Artifact storage mkdir failed: %s", exc)
        return "UNAVAILABLE", "Artifact storage directory could not be created or accessed"

    if not artifact_root.is_dir():
        return "UNAVAILABLE", "Artifact storage path does not resolve to a directory"

    probe_path: Path | None = None
    try:
        fd, probe_str = tempfile.mkstemp(prefix=".readiness_probe_", dir=str(artifact_root))
        probe_path = Path(probe_str)
        try:
            with open(fd, "wb", closefd=True) as f:
                f.write(b"probe\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as write_exc:
            logger.warning("Artifact storage probe write failed: %s", write_exc)
            return "UNAVAILABLE", "Artifact storage is not writable by the service"
        finally:
            if probe_path.exists():
                try:
                    probe_path.unlink()
                except Exception as del_exc:
                    logger.warning("Artifact storage probe cleanup failed: %s", del_exc)
        return "AVAILABLE", None
    except Exception as exc:
        logger.warning("Artifact storage writability probe failed: %s", exc)
        if probe_path and probe_path.exists():
            try:
                probe_path.unlink()
            except Exception:
                pass
        return "UNAVAILABLE", "Artifact storage is not writable by the service"


def _parse_region_spec(data: dict[str, Any]) -> RegionSpec:
    coord_space = data.get("coordinate_space_id", "source")
    unit_str = data.get("unit", "NORMALIZED").upper()
    unit = CoordinateUnit(unit_str)

    bbox_data = data.get("bbox", {})
    if unit == CoordinateUnit.NORMALIZED:
        bbox = NormalizedBBox(
            x=float(bbox_data["x"]),
            y=float(bbox_data["y"]),
            width=float(bbox_data["width"]),
            height=float(bbox_data["height"]),
        )
    else:
        bbox = PixelBBox(
            x=int(bbox_data["x"]),
            y=int(bbox_data["y"]),
            width=int(bbox_data["width"]),
            height=int(bbox_data["height"]),
        )
    return RegionSpec(coordinate_space_id=coord_space, unit=unit, bbox=bbox)


def _parse_geometry_instruction(data: dict[str, Any]) -> GeometryInstruction:
    mode_str = data.get("mode", "NO_OP").upper()
    try:
        mode = GeometryMode(mode_str)
    except ValueError as exc:
        raise ValueError(f"Invalid geometry mode: '{mode_str}'") from exc

    source_str = data.get("source", "DEFAULT").upper()
    try:
        source = GeometrySource(source_str)
    except ValueError as exc:
        raise ValueError(f"Invalid geometry source: '{source_str}'") from exc

    quad_spec: QuadSpec | None = None
    if mode == GeometryMode.FOUR_POINT:
        quad_data = data.get("quadrilateral")
        if not quad_data or not isinstance(quad_data, dict):
            raise ValueError(
                "FOUR_POINT geometry requires a 'quadrilateral' specification dictionary"
            )

        q_coord_space = quad_data.get(
            "coordinate_space_id", data.get("input_space_id", "display_crop")
        )
        q_unit_str = quad_data.get("unit", "NORMALIZED").upper()
        try:
            q_unit = CoordinateUnit(q_unit_str)
        except ValueError as exc:
            raise ValueError(f"Invalid quadrilateral coordinate unit: '{q_unit_str}'") from exc

        raw_pts = quad_data.get("points")
        if not raw_pts or not isinstance(raw_pts, (list, tuple)) or len(raw_pts) != 4:
            raise ValueError(
                f"FOUR_POINT quadrilateral requires exactly 4 points, got {len(raw_pts) if isinstance(raw_pts, (list, tuple)) else raw_pts}"
            )

        points_list = []
        for i, pt in enumerate(raw_pts):
            if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
                raise ValueError(f"Quad point at index {i} must be a dict containing 'x' and 'y'")
            points_list.append(Point(x=float(pt["x"]), y=float(pt["y"])))

        order_str = quad_data.get("order", "TL_TR_BR_BL").upper()
        try:
            order = QuadrilateralOrder(order_str)
        except ValueError as exc:
            raise ValueError(f"Invalid quadrilateral order: '{order_str}'") from exc

        quad = Quadrilateral(
            points=(points_list[0], points_list[1], points_list[2], points_list[3]),
            order=order,
        )
        quad_spec = QuadSpec(coordinate_space_id=q_coord_space, unit=q_unit, quadrilateral=quad)

    target_w = int(data["target_width"]) if data.get("target_width") is not None else None
    target_h = int(data["target_height"]) if data.get("target_height") is not None else None

    return GeometryInstruction(
        mode=mode,
        source=source,
        input_space_id=data.get("input_space_id", "display_crop"),
        quadrilateral=quad_spec,
        target_width=target_w,
        target_height=target_h,
        interpolation=data.get("interpolation", "LINEAR"),
        border_mode=data.get("border_mode", "CONSTANT"),
        profile_id=data.get("profile_id"),
        profile_revision=data.get("profile_revision"),
    )


def load_pipeline_config_from_yaml(
    yaml_path: Path,
    ocr_model_path_override: str | None = None,
    yolo_model_path_override: str | None = None,
    artifact_root_override: Path | None = None,
) -> PipelineConfig:
    """Service-owned YAML loader converting configuration dictionary into PipelineConfig."""
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Pipeline config file not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # 1. Quality Policy
    q_data = raw.get("quality", {})
    quality_policy = QualityPolicy(
        profile_id=q_data.get("profile_id", "quality_default_v1"),
        revision=q_data.get("revision", "1.0"),
        acceptance_complete=q_data.get("acceptance_complete", True),
        min_width=int(q_data.get("min_width", 320)),
        min_height=int(q_data.get("min_height", 240)),
        dark_pixel_threshold=int(q_data.get("dark_pixel_threshold", 15)),
        max_dark_fraction=float(q_data.get("max_dark_fraction", 0.90)),
        bright_pixel_threshold=int(q_data.get("bright_pixel_threshold", 240)),
        max_bright_fraction=float(q_data.get("max_bright_fraction", 0.90)),
        blur_metric_id=q_data.get("blur_metric_id", "laplacian_variance_v1"),
        min_blur_metric=float(q_data.get("min_blur_metric", 10.0)),
    )

    # 2. Localization Profiles
    loc_profiles: dict[str, LocalizationProfile] = {}
    for pid, pdata in raw.get("localization_profiles", {}).items():
        loc_profiles[pid] = LocalizationProfile(
            profile_id=pdata.get("profile_id", pid),
            revision=pdata.get("revision", "1.0"),
            meter_type=MeterType(pdata["meter_type"]),
            capture_context=pdata.get("capture_context", "default"),
            reference_source_width=int(pdata.get("reference_source_width", 640)),
            reference_source_height=int(pdata.get("reference_source_height", 480)),
            aspect_ratio_tolerance=float(pdata.get("aspect_ratio_tolerance", 0.10)),
            display_roi=_parse_region_spec(pdata["display_roi"]),
            geometry=_parse_geometry_instruction(pdata.get("geometry", {})),
            reading_value_roi=_parse_region_spec(pdata["reading_value_roi"]),
            acceptance_complete=pdata.get("acceptance_complete", True),
        )

    # 3. Validation Profiles
    val_profiles: dict[str, ValidationProfile] = {}
    for pid, pdata in raw.get("validation_profiles", {}).items():
        dec_pol = pdata.get("decimal_policy", "OPTIONAL").upper()
        lead_pol = pdata.get("leading_zero_policy", "ALLOW_ANY").upper()
        val_profiles[pid] = ValidationProfile(
            profile_id=pdata.get("profile_id", pid),
            revision=pdata.get("revision", "1.0"),
            meter_type=MeterType(pdata["meter_type"]),
            acceptance_complete=pdata.get("acceptance_complete", True),
            full_match_pattern=pdata["full_match_pattern"],
            decimal_policy=DecimalPolicy(dec_pol),
            integer_digits_min=int(pdata.get("integer_digits_min", 1)),
            integer_digits_max=int(pdata.get("integer_digits_max", 10)),
            fractional_digits_min=int(pdata.get("fractional_digits_min", 0)),
            fractional_digits_max=int(pdata.get("fractional_digits_max", 4)),
            leading_zero_policy=LeadingZeroPolicy(lead_pol),
            numeric_min=pdata.get("numeric_min"),
            numeric_max=pdata.get("numeric_max"),
            min_ocr_confidence_for_accept=float(pdata.get("min_ocr_confidence_for_accept", 0.85)),
        )

    # 4. Reading Profiles
    reading_profiles: dict[str, ReadingProfile] = {}
    for pid, pdata in raw.get("reading_profiles", {}).items():
        reading_profiles[pid] = ReadingProfile(
            profile_id=pdata.get("profile_id", pid),
            meter_type=MeterType(pdata["meter_type"]),
            validation_profile_id=pdata["validation_profile_id"],
        )

    # 5. Meter Type Reading Profiles mapping
    meter_type_reading_profiles: dict[MeterType, str] = {}
    for mtype_str, rpid in raw.get("meter_type_reading_profiles", {}).items():
        meter_type_reading_profiles[MeterType(mtype_str)] = rpid

    # 6. Decision Policy
    d_data = raw.get("decision", {})
    supported_mtypes = tuple(
        MeterType(m) for m in d_data.get("supported_meter_types", ["VSEE_VSE3T", "GELEX_ME40"])
    )
    decision_policy = DecisionPolicy(
        policy_id=d_data.get("policy_id", "decision_default_v1"),
        version=d_data.get("version", "1.0"),
        supported_meter_types=supported_mtypes,
        manual_localization_acceptance_enabled=d_data.get(
            "manual_localization_acceptance_enabled", True
        ),
        learned_evidence_acceptance_enabled=d_data.get(
            "learned_evidence_acceptance_enabled", False
        ),
        persistence_required_for_accept=d_data.get("persistence_required_for_accept", True),
    )

    # 7. Recognition Policy - zero download strictly enforced
    rec_data = raw.get("recognition", {})
    if rec_data.get("allow_model_download") is True:
        raise ValueError(
            "MODEL_DOWNLOAD_POLICY_FORBIDDEN: allow_model_download=True is forbidden for recognition policy"
        )
    model_path = (
        ocr_model_path_override if ocr_model_path_override else rec_data.get("model_path", "")
    )
    recognition_policy = RecognitionPolicy(
        reader_id=rec_data.get("reader_id", "paddleocr_v6_reader_v1"),
        logical_model_id=rec_data.get("logical_model_id", "PP-OCRv6-medium"),
        resolved_model_version=rec_data.get("resolved_model_version", "1.0"),
        model_path=model_path,
        allow_model_download=False,
        device=rec_data.get("device", "cpu"),
    )

    # 8. Detector Policy - zero download strictly enforced
    detector_policy: DetectorPolicy | None = None
    det_data = raw.get("detector")
    if det_data:
        if det_data.get("allow_model_download") is True:
            raise ValueError(
                "MODEL_DOWNLOAD_POLICY_FORBIDDEN: allow_model_download=True is forbidden for detector policy"
            )
        det_model_path = (
            yolo_model_path_override if yolo_model_path_override else det_data.get("model_path", "")
        )
        detector_policy = DetectorPolicy(
            model_path=det_model_path,
            expected_model_sha256=det_data.get("expected_model_sha256"),
            logical_model_id=det_data.get("logical_model_id", "yuva_reading_value_yolo11n"),
            resolved_model_version=det_data.get("resolved_model_version", "yolo11n_v1"),
            input_space=det_data.get("input_space", "working_display"),
            target_class_id=int(det_data.get("target_class_id", 0)),
            target_class_name=det_data.get("target_class_name", "item"),
            conf_threshold=float(det_data.get("conf_threshold", 0.01)),
            iou_threshold=float(det_data.get("iou_threshold", 0.50)),
            max_det=int(det_data.get("max_det", 20)),
            allow_model_download=False,
            device=det_data.get("device", "cpu"),
            allow_configured_fallback=det_data.get("allow_configured_fallback", True),
        )

    # 9. Artifact Policy
    art_data = raw.get("artifacts", {})
    art_root = (
        str(artifact_root_override)
        if artifact_root_override
        else art_data.get("artifact_root", "local_artifacts")
    )
    artifact_policy = ArtifactPolicy(
        policy_id=art_data.get("policy_id", "artifacts_default_v1"),
        artifact_root=art_root,
        persistence_required=art_data.get("persistence_required", True),
        retain_source=art_data.get("retain_source", True),
        retain_display=art_data.get("retain_display", True),
        retain_working_display=art_data.get("retain_working_display", True),
        retain_reading_value=art_data.get("retain_reading_value", True),
    )

    config = PipelineConfig(
        schema_version=raw.get("schema_version", "1.0"),
        config_id=raw.get("config_id", "pipeline_demo_v1"),
        revision=raw.get("revision", "1.0.0"),
        quality=quality_policy,
        localization_profiles=loc_profiles,
        validation_profiles=val_profiles,
        reading_profiles=reading_profiles,
        meter_type_reading_profiles=meter_type_reading_profiles,
        decision=decision_policy,
        recognition=recognition_policy,
        artifacts=artifact_policy,
        detector=detector_policy,
    )
    return freeze_config(config)


def create_pipeline(settings: Settings) -> MeterReadingPipeline:
    """Instantiate the singleton MeterReadingPipeline using service settings."""
    try:
        config = load_pipeline_config_from_yaml(
            yaml_path=settings.pipeline_config_path,
            ocr_model_path_override=settings.ocr_model_path if settings.ocr_model_path else None,
            yolo_model_path_override=settings.yolo_model_path if settings.yolo_model_path else None,
            artifact_root_override=settings.artifact_root,
        )
        return MeterReadingPipeline(config=config)
    except Exception as exc:
        logger.error("Failed to construct MeterReadingPipeline: %s", exc)
        raise PipelineNotReadyError(
            message=f"Failed to initialize pipeline from config '{settings.pipeline_config_path}': {exc}"
        ) from exc
