import asyncio
import unittest
from unittest.mock import MagicMock, patch

from api.schemas.evaluations import EvalRunRequest
from api.services.eval_service import EvalRun, EvalRunError, EvalService


class EvalServiceContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_run_returns_cancelled_status_and_notifies_subscribers(self):
        service = EvalService()
        run = EvalRun(EvalRunRequest(models=["demo"], suite="smoke"))
        run.status = "running"
        run.message = "Evaluating demo"
        run._task = MagicMock()
        service._runs[run.run_id] = run

        queue: asyncio.Queue = asyncio.Queue()
        service.subscribe_progress(run.run_id, queue)

        cancelled = service.cancel_run(run.run_id)

        self.assertIsNotNone(cancelled)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(cancelled.error_code, None)
        self.assertEqual(service.get_run(run.run_id).message, "Cancelled by user")
        run._task.cancel.assert_called_once_with()

        update = queue.get_nowait()
        self.assertEqual(update.status, "cancelled")
        self.assertEqual(update.message, "Cancelled by user")
        self.assertEqual(update.error_code, None)
        self.assertEqual(update.error_stage, None)

    async def test_execute_run_surfaces_structured_failure_fields(self):
        service = EvalService()
        run = EvalRun(EvalRunRequest(models=["demo"], suite="smoke"))

        queue: asyncio.Queue = asyncio.Queue()
        service.subscribe_progress(run.run_id, queue)

        with patch.object(
            service,
            "_run_pipeline_sync",
            side_effect=EvalRunError("dataset_not_found", "dataset_resolution", "Dataset missing"),
        ):
            await service._execute_run(run)

        self.assertEqual(run.status, "failed")
        self.assertEqual(run.error, "Dataset missing")
        self.assertEqual(run.error_code, "dataset_not_found")
        self.assertEqual(run.error_stage, "dataset_resolution")
        self.assertEqual(run.message, "Failed during dataset_resolution: Dataset missing")

        first_update = await queue.get()
        final_update = await queue.get()

        self.assertEqual(first_update.status, "running")
        self.assertEqual(final_update.status, "failed")
        self.assertEqual(final_update.error_code, "dataset_not_found")
        self.assertEqual(final_update.error_stage, "dataset_resolution")


if __name__ == "__main__":
    unittest.main()