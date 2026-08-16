"""API tests for /api/v1/runs/{run_id}/artifacts/{role} endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient


def test_artifact_fetch_success(test_app: TestClient, temp_artifact_dir: Path) -> None:
    # Set up dummy artifact
    run_dir = temp_artifact_dir / "runs" / "run_art_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    img_file = run_dir / "source.jpg"
    img_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")

    registry = test_app.app.state.registry
    registry.register_direct("run_art_1", {"source": img_file}, {"source": "image/jpeg"})

    response = test_app.get("/api/v1/runs/run_art_1/artifacts/source")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert len(response.content) > 0


def test_artifact_fetch_unknown_run_returns_404(test_app: TestClient) -> None:
    response = test_app.get("/api/v1/runs/non_existent_run/artifacts/source")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ARTIFACT_NOT_FOUND"


def test_artifact_fetch_disallowed_role_returns_404(test_app: TestClient) -> None:
    response = test_app.get("/api/v1/runs/run_art_1/artifacts/manifest")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ARTIFACT_NOT_FOUND"


def test_artifact_traversal_rejection(test_app: TestClient, tmp_path: Path) -> None:
    outside_file = tmp_path / "outside.jpg"
    outside_file.write_bytes(b"DATA")

    registry = test_app.app.state.registry
    registry.register_direct("run_traversal_api", {"source": outside_file})

    response = test_app.get("/api/v1/runs/run_traversal_api/artifacts/source")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ARTIFACT_NOT_FOUND"
