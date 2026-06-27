import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from api.schemas.evaluations import EvalRunRequest
from api.services.eval_service import EvalRun, EvalService


def _load_evaluations_router_module():
    module_path = Path(__file__).resolve().parent / "api" / "routers" / "evaluations.py"
    spec = importlib.util.spec_from_file_location("isolated_evaluations_router", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvaluationsRouterContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_evaluation_unknown_model_returns_structured_error(self):
        module = _load_evaluations_router_module()
        service = EvalService()
        request = EvalRunRequest(models=["missing-model"], suite="smoke")

        with patch("api.services.config_service.ConfigService.get_models", return_value={}):
            with patch(
                "api.services.config_service.ConfigService.get_test_suites",
                return_value={"smoke": {"tests": ["case_a"]}},
            ):
                with self.assertRaises(HTTPException) as context:
                    await module.start_evaluation(request, service)

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.detail["error_code"], "unknown_models")
        self.assertEqual(context.exception.detail["error_stage"], "request_validation")
        self.assertIn("Unknown models", context.exception.detail["message"])

    async def test_cancel_run_returns_cancelled_status_payload(self):
        module = _load_evaluations_router_module()
        service = EvalService()
        run = EvalRun(EvalRunRequest(models=["demo"], suite="smoke"))
        run.status = "running"
        service._runs[run.run_id] = run

        status = module.cancel_run(run.run_id, service)

        self.assertEqual(status.run_id, run.run_id)
        self.assertEqual(status.status, "cancelled")
        self.assertIsNotNone(status.finished_at)
        self.assertEqual(status.error_code, None)
        self.assertEqual(service.get_run(run.run_id).message, "Cancelled by user")


if __name__ == "__main__":
    unittest.main()