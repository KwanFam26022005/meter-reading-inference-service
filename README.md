# Meter Reading Inference Service (A1)

FastAPI Inference and Inspection Adapter for the frozen **meter-reading-engine-v2** Core pipeline.

This service acts strictly as an adapter and visualization backend for the upcoming **A2 Mobile Web Visualizer (Next.js)** frontend.

> **CRITICAL ARCHITECTURAL BOUNDARY:**
> - **NO** camera capture or hardware integration.
> - **NO** data collection or dataset export.
> - **NO** login, authentication, or session management.
> - **NO** database (PostgreSQL/MySQL/MongoDB) or message broker (Redis/Celery/Kafka).
> - **NO** business workflows, task assignment, or check-in/out logic.
> - `meter-reading-engine-v2` is completely **FROZEN & READ-ONLY**.

---

## 1. Installation & Dependency Pinning

### Frozen Core Pin (A1-001)
- **Repository:** `https://github.com/KwanFam26022005/meter-reading-engine-v2`
- **Frozen Branch:** `feature/ocr-q1-fixed-decimal-normalization`
- **Canonical Revision:** `a68bd884ae2b023fdeadd1ccefe25524a995f080`

### Reproducible Install
The service pins the exact frozen Core revision in `pyproject.toml` using PEP 508 git dependency specification:
```bash
pip install "meter-reading-engine @ git+https://github.com/KwanFam26022005/meter-reading-engine-v2.git@a68bd884ae2b023fdeadd1ccefe25524a995f080"
pip install -e ".[dev]"
```

### Local Editable Development
For local development against a local clone of the frozen Core:
```bash
pip install -e D:\Projects\meter-reading-engine-v2 -e ".[dev]"
```
> **Note:** Runtime revision verification remains mandatory in all modes. An arbitrary or mismatched installed version of `meter-reading-engine` will fail closed at startup and during inference.

---

## 2. Readiness: Code Ready vs Real Demo Ready

### A1 Code Ready
The service codebase is fully implemented, verified, and test-covered (all unit and API integration tests passing with zero network or external weight requirements).

### Real Demo Ready Prerequisites
To achieve **`READY`** status for actual live inference on a real device/meter, the following prerequisites must all be satisfied simultaneously:
1. **Verified Frozen Core:** Git HEAD of Core repository matches `a68bd884ae2b023fdeadd1ccefe25524a995f080`.
2. **Calibrated Configuration:** `PIPELINE_CONFIG_PATH` points to a genuinely calibrated YAML configuration (`calibration_status: "CALIBRATED"`).
   > **Important:** `config/pipeline.demo.example.yaml` is **EXAMPLE ONLY / NOT CALIBRATED / NOT REAL-INFERENCE READY**. It contains synthetic ROI fixtures for syntax validation and fake-pipeline testing only.
3. **Compatible OCR Runtime:** Installed Python packages match `paddleocr==3.7.0`, `paddlex>=3.7.0,<3.8.0`, and `paddlepaddle==3.3.1`.
4. **Provisioned Local OCR Model Bundle:** Local offline `PP-OCRv6-medium` bundle directory is present containing valid `inference.json` (or `.pdmodel`), `inference.pdiparams` (or `.params`), and `inference.yml` (`Global.model_name == PP-OCRv6_medium_rec`).
5. **Zero-Download Policy:** Model auto-downloading is disabled (`allow_model_download: false`).
6. **Usable Artifact Storage:** `ARTIFACT_ROOT` directory is writable.

If any prerequisite is missing:
- `GET /health` returns `200` (`status: "ok"`).
- `GET /api/v1/models/status` returns `ready: false`, `status: "not_ready"`.
- `POST /api/v1/meter-readings/infer` fails closed with `503 PIPELINE_NOT_READY`.

---

## 3. Environment Configuration

Copy `.env.example` to `.env` or set environment variables:

| Variable | Type | Default | Description |
|---|---|---|---|
| `PIPELINE_CONFIG_PATH` | Path | `config/pipeline.demo.example.yaml` | Service YAML pipeline configuration |
| `CORE_REPO_PATH` | Path | `D:\Projects\meter-reading-engine-v2` | Path to frozen Core repository |
| `OCR_MODEL_PATH` | String | `""` | Local offline OCR model weights directory |
| `YOLO_MODEL_PATH` | Path | `D:\Models\meter-reading-engine\lcd-reading-value-yolo-v1\best.pt` | YOLOv11 checkpoint for learned locator |
| `ARTIFACT_ROOT` | Path | `local_artifacts` | Directory where run artifact images are persisted |
| `MAX_UPLOAD_BYTES` | Integer | `10485760` (10MB) | Maximum allowed image upload size |
| `MAX_INFERENCE_CONCURRENCY` | Integer | `1` | Max concurrent pipeline executions |
| `ENABLE_LEARNED_PRIMARY` | Boolean | `false` | Enable `LEARNED_PRIMARY` mode (forbidden by default) |
| `ALLOWED_ORIGINS` | JSON / CSV | `["http://localhost:3000"]` | Allowed CORS origins for A2 frontend |
| `DEFAULT_LOCALIZATION_PROFILES` | JSON | `{"VSEE_VSE3T": "...", "GELEX_ME40": "..."}` | Explicit mapping of meter type to default profile ID |
| `LOG_LEVEL` | String | `INFO` | Logging level |

---

## 4. API Surface

### 1. `GET /health`
Liveness probe. Does not invoke inference. Always returns 200.
```json
{
  "status": "ok",
  "service": "meter-reading-inference-service",
  "api_version": "v1"
}
```

### 2. `GET /api/v1/models/status`
Readiness probe and capability snapshot.
```json
{
  "status": "ready",
  "ready": true,
  "core": {
    "expected_revision": "a68bd884ae2b023fdeadd1ccefe25524a995f080",
    "current_revision": "a68bd884ae2b023fdeadd1ccefe25524a995f080",
    "revision_verified": true
  },
  "pipeline_config": {
    "config_id": "pipeline_demo_v1",
    "revision": "1.0.0",
    "valid": true,
    "calibrated": true,
    "calibration_status": "CALIBRATED"
  },
  "recognition": {
    "model_id": "PP-OCRv6-medium",
    "runtime_status": "AVAILABLE",
    "asset_status": "AVAILABLE"
  },
  "detector": {
    "model_id": "yuva_reading_value_yolo11n",
    "asset_status": "AVAILABLE",
    "required_for_default": false
  },
  "capabilities": {
    "supported_meter_types": ["VSEE_VSE3T", "GELEX_ME40"],
    "default_localization_profiles": {
      "VSEE_VSE3T": "vsee_vse3t_station_a_v1",
      "GELEX_ME40": "gelex_me40_station_b_v1"
    },
    "configured_mode_ready": true,
    "learned_shadow_mode_ready": true,
    "learned_shadow_decision_path_ready": true,
    "learned_shadow_telemetry_ready": true,
    "learned_primary_enabled": false
  }
}
```

### 3. `POST /api/v1/meter-readings/infer`
Execute inference on an uploaded meter image.

**Form Parameters:**
- `image` *(File, required)*: JPEG or PNG binary bytes.
- `meter_type` *(Form string, required)*: `VSEE_VSE3T` or `GELEX_ME40`.
- `request_id` *(Form string, optional)*: Client trace UUID.
- `reading_profile` *(Form string, optional)*.
- `validation_profile_id` *(Form string, optional)*.
- `localization_profile_id` *(Form string, optional)*.
- `locator_mode` *(Form string, optional)*: `CONFIGURED` (default) or `LEARNED_SHADOW`.

### 4. `GET /api/v1/runs/{run_id}/artifacts/{role}`
Stream a specific run artifact image safely.
- **Allowed Roles:** `source`, `display`, `working_display`, `reading_value`.

---

## 5. Running the Service & Testing

```bash
# Start Uvicorn single worker
uvicorn meter_reading_inference_service.main:app --host 0.0.0.0 --port 8000
```

### Quality Gates
```bash
# Run pytest test suite
python -m pytest -q

# Run Ruff linter
python -m ruff check src tests

# Run Ruff formatter check
python -m ruff format --check src tests
```

---

## 6. Zero-Download Guarantee
- All OCR and detector weights must reside locally on disk.
- If YAML configuration requests `allow_model_download: true`, config parsing fails immediately with `MODEL_DOWNLOAD_POLICY_FORBIDDEN`.
