"""Unit tests for in-memory artifact registry and traversal safety."""

import time
from pathlib import Path

import pytest
from meter_reading_engine.contracts import (
    ArtifactIndex,
    ArtifactRef,
    ArtifactRole,
)

from meter_reading_inference_service.artifacts import InMemoryArtifactRegistry
from meter_reading_inference_service.errors import ArtifactNotFoundError


def test_registry_register_and_resolve(temp_artifact_dir: Path) -> None:
    registry = InMemoryArtifactRegistry(artifact_root=temp_artifact_dir)

    # Create dummy artifact
    run_dir = temp_artifact_dir / "runs" / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_file = run_dir / "source.jpg"
    source_file.write_bytes(b"TEST_IMAGE")

    index = ArtifactIndex(
        store_id="test",
        run_relative_root="runs/run_1",
        artifacts=(
            ArtifactRef(
                role=ArtifactRole.SOURCE,
                relative_path="source.jpg",
                media_type="image/jpeg",
                sha256="abc",
                byte_length=10,
            ),
        ),
    )

    registry.register_index("run_1", index)

    path, media_type = registry.resolve_artifact_file("run_1", "source")
    assert path == source_file.resolve()
    assert media_type == "image/jpeg"


def test_registry_unknown_or_invalid_role(temp_artifact_dir: Path) -> None:
    registry = InMemoryArtifactRegistry(artifact_root=temp_artifact_dir)

    with pytest.raises(ArtifactNotFoundError):
        registry.resolve_artifact_file("run_1", "manifest")

    with pytest.raises(ArtifactNotFoundError):
        registry.resolve_artifact_file("run_1", "invalid_role")


def test_registry_ttl_expiration(temp_artifact_dir: Path) -> None:
    registry = InMemoryArtifactRegistry(artifact_root=temp_artifact_dir, ttl_seconds=1)

    dummy_file = temp_artifact_dir / "test.jpg"
    dummy_file.write_bytes(b"DATA")

    registry.register_direct("run_exp", {"source": dummy_file})

    # Should resolve immediately
    p, _ = registry.resolve_artifact_file("run_exp", "source")
    assert p == dummy_file.resolve()

    # Sleep past TTL
    time.sleep(1.1)

    with pytest.raises(ArtifactNotFoundError):
        registry.resolve_artifact_file("run_exp", "source")


def test_registry_path_traversal_rejection(temp_artifact_dir: Path, tmp_path: Path) -> None:
    registry = InMemoryArtifactRegistry(artifact_root=temp_artifact_dir)

    # Create file outside artifact root
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_file = outside_dir / "secret.txt"
    outside_file.write_bytes(b"SECRET")

    # Try registering outside path
    registry.register_direct("run_traversal", {"source": outside_file})

    with pytest.raises(ArtifactNotFoundError):
        registry.resolve_artifact_file("run_traversal", "source")
