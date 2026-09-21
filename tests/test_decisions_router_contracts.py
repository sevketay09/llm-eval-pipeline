"""Contract tests for decision-model guards wired into api/routers/models.py
and api/routers/evaluations.py. No real Jev/LLM network calls: uses the
offline typesafe-mock and mock providers.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.main import app
from api.routers.models import get_config_service
from api.schemas.evaluations import EvalRunRequest
from api.services.config_service import ConfigService
from api.services.eval_service import EvalService


def _load_evaluations_router_module():
    module_path = Path(__file__).resolve().parent.parent / "api" / "routers" / "evaluations.py"
    spec = importlib.util.spec_from_file_location("isolated_evaluations_router_decisions", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_CONFIG_YAML = (
    "models:\n"
    "  demo-model:\n"
    "    provider: mock\n"
    "    model_name: demo-model\n"
    "    api_key: none\n"
    "  demo-jev:\n"
    "    provider: typesafe-mock\n"
    "    model_name: jev-mock\n"
    "    api_key: none\n"
)


def _make_client(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(_CONFIG_YAML, encoding="utf-8")

    def _override():
        return ConfigService(config_path=str(config_path))

    app.dependency_overrides[get_config_service] = _override
    client = TestClient(app)
    return client, str(config_path)


class ModelsRouterCapabilitiesContractTests(unittest.TestCase):
    def tearDown(self):
        app.dependency_overrides.pop(get_config_service, None)

    def test_capabilities_splits_decision_and_llm_models(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client, _ = _make_client(Path(tmp))
            resp = client.get("/api/models/capabilities")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertTrue(body["decision_available"])
            self.assertEqual(body["decision_models"], ["demo-jev"])
            self.assertEqual(body["llm_models"], ["demo-model"])

    def test_test_endpoint_decision_model_ok(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client, config_path = _make_client(Path(tmp))
            with patch("api.config.get_settings") as mock_settings:
                mock_settings.return_value.models_config_path = config_path
                resp = client.post("/api/models/demo-jev/test")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertTrue(body["ok"])
            self.assertIsNone(body["error"])

    def test_test_endpoint_llm_model_ok(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client, config_path = _make_client(Path(tmp))
            with patch("api.config.get_settings") as mock_settings:
                mock_settings.return_value.models_config_path = config_path
                resp = client.post("/api/models/demo-model/test")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertTrue(body["ok"])

    def test_test_endpoint_unknown_model_404(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client, _ = _make_client(Path(tmp))
            resp = client.post("/api/models/nonexistent/test")
            self.assertEqual(resp.status_code, 404)


class EvaluationsRouterDecisionGuardContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_evaluation_rejects_decision_model_as_target(self):
        module = _load_evaluations_router_module()
        service = EvalService()
        request = EvalRunRequest(models=["demo-jev"], suite="smoke")

        available = {
            "demo-jev": {"provider": "typesafe-mock", "model_name": "jev-mock"},
            "demo-model": {"provider": "mock", "model_name": "demo-model"},
        }
        with patch("api.services.config_service.ConfigService.get_models", return_value=available):
            with self.assertRaises(HTTPException) as ctx:
                await module.start_evaluation(request, service)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail["error_code"], "decision_model_not_evaluable")
        self.assertIn("demo-jev", ctx.exception.detail["message"])


class AdapterAndPipelineGuardContractTests(unittest.TestCase):
    def test_unified_adapter_rejects_decision_provider(self):
        from adapters.unified_adapter import UnifiedLLMAdapter

        with self.assertRaises(ValueError):
            UnifiedLLMAdapter({"provider": "typesafe", "model_name": "jev-latest"}, model_key="jev")

    def test_unified_adapter_rejects_mock_decision_provider(self):
        from adapters.unified_adapter import UnifiedLLMAdapter

        with self.assertRaises(ValueError):
            UnifiedLLMAdapter({"provider": "typesafe-mock", "model_name": "jev-mock"}, model_key="demo-jev")

    def test_pipeline_runner_initialize_model_rejects_decision_model(self):
        import pipeline_runner

        pipeline = pipeline_runner.EvaluationPipeline.__new__(pipeline_runner.EvaluationPipeline)
        pipeline.config = {
            "models": {
                "demo-jev": {"provider": "typesafe-mock", "model_name": "jev-mock"},
            },
            "embedding_models": {},
        }
        pipeline.runtime_overrides = {}
        pipeline.adapters = {}

        with self.assertRaises(ValueError) as ctx:
            pipeline_runner.EvaluationPipeline.initialize_model(pipeline, "demo-jev")
        message = str(ctx.exception).lower()
        self.assertIn("decision", message)
        self.assertIn("jev", message)


if __name__ == "__main__":
    unittest.main()
