"""Unit tests for A1-007 cancellation-safe concurrency."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from meter_reading_engine.contracts import DecisionOutcome

from meter_reading_inference_service.artifacts import InMemoryArtifactRegistry
from meter_reading_inference_service.inference import InferenceCoordinator
from meter_reading_inference_service.settings import Settings
from tests.helpers import create_sample_pipeline_result


def test_cancellation_retains_semaphore_until_worker_finishes(
    synthetic_valid_image_bytes: bytes,
    tmp_path: Path,
) -> None:
    """A1-007: Cancelling an asyncio request task does not prematurely release the semaphore slot."""

    async def _run_test() -> None:
        settings = Settings(
            max_inference_concurrency=1,
            artifact_root=tmp_path,
        )
        registry = InMemoryArtifactRegistry(artifact_root=tmp_path)

        worker_started = asyncio.Event()
        worker_finished = asyncio.Event()
        active_count = 0
        max_active_observed = 0
        loop = asyncio.get_running_loop()

        def slow_pipeline_run(input_data: object) -> object:
            nonlocal active_count, max_active_observed
            active_count += 1
            max_active_observed = max(max_active_observed, active_count)
            loop.call_soon_threadsafe(worker_started.set)
            try:
                # Simulate CPU/blocking work in worker thread
                time.sleep(0.20)
                return create_sample_pipeline_result(
                    outcome=DecisionOutcome.ACCEPT,
                    run_id="run_cancel_test",
                    request_id="req_slow",
                    artifact_root=tmp_path,
                )
            finally:
                active_count -= 1
                loop.call_soon_threadsafe(worker_finished.set)

        mock_pipe = MagicMock()
        mock_pipe.run.side_effect = slow_pipeline_run
        mock_pipe.config = MagicMock()
        mock_pipe.config.config_id = "test_cfg"
        mock_pipe.config.revision = "1.0"
        mock_pipe.config.config_sha256 = "sha256"

        coordinator = InferenceCoordinator(
            pipeline=mock_pipe,
            registry=registry,
            settings=settings,
            core_revision_verified=True,
            core_current_revision=settings.expected_core_revision,
            import_origin_verified=True,
            is_calibrated=True,
            calibration_status="CALIBRATED",
            ocr_runtime_status="AVAILABLE",
            ocr_asset_status="AVAILABLE",
            detector_runtime_status="AVAILABLE",
            detector_asset_status="AVAILABLE",
            artifact_storage_status="AVAILABLE",
        )

        # 1. Launch Request A
        task_a = asyncio.create_task(
            coordinator.infer(
                image_bytes=synthetic_valid_image_bytes,
                media_type="image/jpeg",
                meter_type_str="VSEE_VSE3T",
                request_id="req_A",
            )
        )

        # 2. Wait until Request A has actually entered pipeline.run in thread pool
        await worker_started.wait()
        assert active_count == 1

        # 3. Cancel Request A while its underlying worker thread is still running
        task_a.cancel()

        # 4. Immediately launch Request B
        b_entered_before_a_finished = False

        async def run_b() -> object:
            nonlocal b_entered_before_a_finished
            res = await coordinator.infer(
                image_bytes=synthetic_valid_image_bytes,
                media_type="image/jpeg",
                meter_type_str="VSEE_VSE3T",
                request_id="req_B",
            )
            # If B finished before worker_finished was set, that's a failure
            if not worker_finished.is_set():
                b_entered_before_a_finished = True
            return res

        task_b = asyncio.create_task(run_b())

        # Task A should raise CancelledError when awaited
        with pytest.raises(asyncio.CancelledError):
            await task_a

        # Wait for Task B to complete
        resp_b = await task_b
        assert resp_b.summary.decision.outcome == "ACCEPT"

        # Concurrency verification:
        # 1. B could NOT have entered pipeline.run before A finished
        assert b_entered_before_a_finished is False
        # 2. At no point were there 2 active workers in pipeline.run
        assert max_active_observed == 1
        assert mock_pipe.run.call_count == 2

    asyncio.run(_run_test())
