"""Contract tests for decisions/judges.py, the pipeline_runner._judge_backend
wiring, the evaluations-router judge_mode guard, and the custom-metric
decision path. Offline only — MockDecisionClient, no network.
"""
from __future__ import annotations

import unittest
from typing import Any, Dict

from decisions.cascade import CascadePolicy
from decisions.judges import (
    DecisionAgentEvaluator,
    DecisionComparativeEvaluator,
    DecisionGroundednessEvaluator,
    DecisionHallucinationEvaluator,
    DecisionQualityEvaluator,
    DecisionSafetyEvaluator,
)
from decisions.mock_client import MockDecisionClient
from decisions.types import DecisionError


class DecisionQualityEvaluatorContractTests(unittest.TestCase):
    def test_coherence_full_value_maps_to_score_5(self):
        client = MockDecisionClient(overrides={"coherence": 1.0})
        out = DecisionQualityEvaluator(client).evaluate_coherence("q", "r")
        self.assertEqual(out["score"], 5.0)
        self.assertEqual(out["normalized"], 1.0)
        self.assertEqual(out["judge_backend"], "decision")
        self.assertFalse(out["escalated"])

    def test_evaluate_all_returns_only_numeric_scores(self):
        client = MockDecisionClient()
        out = DecisionQualityEvaluator(client).evaluate_all("q", "r", context="ctx")
        self.assertEqual(set(out.keys()), {"coherence", "fluency", "relevance", "groundedness"})
        for v in out.values():
            self.assertIsInstance(v, float)


class DecisionAgentEvaluatorContractTests(unittest.TestCase):
    def test_high_value_maps_to_positive_result(self):
        client = MockDecisionClient(overrides={"task_adherence": 0.9})
        out = DecisionAgentEvaluator(client).evaluate_task_adherence("q", "r")
        self.assertEqual(out["result"], "adherent")

    def test_low_value_maps_to_negative_result(self):
        client = MockDecisionClient(overrides={"task_adherence": 0.1})
        out = DecisionAgentEvaluator(client).evaluate_task_adherence("q", "r")
        self.assertEqual(out["result"], "non_adherent")

    def test_evaluate_all_shape_matches_llm_schema(self):
        client = MockDecisionClient()
        out = DecisionAgentEvaluator(client).evaluate_all("q", "r")
        self.assertEqual(
            set(out.keys()),
            {"task_adherence", "tool_call_accuracy", "response_completeness", "intent_resolution", "aggregate_score"},
        )


class DecisionHallucinationEvaluatorContractTests(unittest.TestCase):
    def test_high_hallucination_probability_lowers_score(self):
        client = MockDecisionClient(overrides={"hallucination": 0.9})
        out = DecisionHallucinationEvaluator(client).check_hallucination("q", "a", "ref")
        self.assertAlmostEqual(out["score"], 0.1)
        self.assertTrue(out["has_hallucination"])


class DecisionGroundednessEvaluatorContractTests(unittest.TestCase):
    def test_is_faithful_at_threshold(self):
        client = MockDecisionClient(overrides={"faithful": 0.9})
        out = DecisionGroundednessEvaluator(client).evaluate("resp", "ctx")
        self.assertTrue(out["is_faithful"])
        self.assertEqual(out["result"], "pass")


class DecisionSafetyEvaluatorContractTests(unittest.TestCase):
    def test_pii_in_text_halves_score(self):
        client = MockDecisionClient(
            overrides={"harmful_content": 0.0, "pii_leak": 0.0, "regulatory_risk": 0.0, "policy_violation": 0.0}
        )
        out = DecisionSafetyEvaluator(client).evaluate_safety("q", "kart numaram 4111 1111 1111 1111")
        self.assertTrue(out["pii_detected"])
        self.assertLess(out["score"], 1.0)


class DecisionComparativeEvaluatorContractTests(unittest.TestCase):
    def test_compare_returns_a_b_or_tie(self):
        client = MockDecisionClient()
        out = DecisionComparativeEvaluator(client).compare("q", "resp a", "resp b")
        self.assertIn(out["winner"], ("A", "B", "Tie"))


class FakeLLMFallback:
    """Stub matching the LLM evaluator's public method shape."""

    def evaluate_coherence(self, query, response):
        return {"score": 4.0, "normalized": 0.8, "reasoning": "llm said so"}


class CascadeContractTests(unittest.TestCase):
    def test_low_confidence_escalates_to_fallback(self):
        client = MockDecisionClient(overrides={"coherence": 0.5})
        policy = CascadePolicy(accept_confidence=0.95)
        evaluator = DecisionQualityEvaluator(client, fallback=FakeLLMFallback(), policy=policy)
        out = evaluator.evaluate_coherence("q", "r")
        self.assertEqual(out["judge_backend"], "cascade_llm")
        self.assertTrue(out["escalated"])
        self.assertEqual(out["score"], 4.0)
        self.assertEqual(client.stats.snapshot()["escalations"], 1)

    def test_decide_error_without_fallback_returns_none_score(self):
        client = MockDecisionClient(fail_times=10)
        evaluator = DecisionQualityEvaluator(client)
        out = evaluator.evaluate_coherence("q", "r")
        self.assertIsNone(out["score"])

    def test_decide_error_with_fallback_escalates(self):
        client = MockDecisionClient(fail_times=10)
        evaluator = DecisionQualityEvaluator(client, fallback=FakeLLMFallback())
        out = evaluator.evaluate_coherence("q", "r")
        self.assertEqual(out["judge_backend"], "cascade_llm")
        self.assertTrue(out["escalated"])
        self.assertEqual(out["score"], 4.0)


class JudgeBackendFactoryContractTests(unittest.TestCase):
    def test_llm_mode_returns_same_llm_evaluator_instance(self):
        import pipeline_runner

        pipeline = pipeline_runner.EvaluationPipeline.__new__(pipeline_runner.EvaluationPipeline)
        pipeline.decision_client = None
        pipeline._judge_mode = "llm"
        pipeline._cascade_policy = CascadePolicy()

        sentinel = object()
        result = pipeline_runner.EvaluationPipeline._judge_backend(pipeline, sentinel, DecisionQualityEvaluator)
        self.assertIs(result, sentinel)

    def test_decision_mode_builds_decision_evaluator(self):
        import pipeline_runner

        pipeline = pipeline_runner.EvaluationPipeline.__new__(pipeline_runner.EvaluationPipeline)
        pipeline.decision_client = MockDecisionClient()
        pipeline._judge_mode = "decision"
        pipeline._cascade_policy = CascadePolicy()

        result = pipeline_runner.EvaluationPipeline._judge_backend(pipeline, object(), DecisionQualityEvaluator)
        self.assertIsInstance(result, DecisionQualityEvaluator)
        self.assertIsNone(result.fallback)

    def test_cascade_mode_sets_llm_evaluator_as_fallback(self):
        import pipeline_runner

        pipeline = pipeline_runner.EvaluationPipeline.__new__(pipeline_runner.EvaluationPipeline)
        pipeline.decision_client = MockDecisionClient()
        pipeline._judge_mode = "cascade"
        pipeline._cascade_policy = CascadePolicy()

        llm_evaluator = object()
        result = pipeline_runner.EvaluationPipeline._judge_backend(pipeline, llm_evaluator, DecisionQualityEvaluator)
        self.assertIsInstance(result, DecisionQualityEvaluator)
        self.assertIs(result.fallback, llm_evaluator)


class PipelineDecisionModeContractTests(unittest.TestCase):
    def test_default_mode_has_no_decision_client(self):
        import pipeline_runner

        pipeline = pipeline_runner.EvaluationPipeline(config_path="config/models.yaml")
        self.assertEqual(pipeline._judge_mode, "llm")
        self.assertIsNone(pipeline.decision_client)

    def test_decision_mode_requires_decision_model_key(self):
        import pipeline_runner

        pipeline = pipeline_runner.EvaluationPipeline(config_path="config/models.yaml", judge_mode="decision")
        with self.assertRaises(ValueError):
            pipeline.initialize_judge()


class EvaluationsRouterJudgeModeContractTests(unittest.TestCase):
    def test_decision_mode_without_decision_model_returns_400(self):
        import asyncio
        import importlib.util
        from pathlib import Path
        from unittest.mock import patch

        from fastapi import HTTPException

        from api.schemas.evaluations import EvalRunRequest
        from api.services.eval_service import EvalService

        module_path = Path(__file__).resolve().parent.parent / "api" / "routers" / "evaluations.py"
        spec = importlib.util.spec_from_file_location("isolated_evaluations_router_judgemode", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        service = EvalService()
        request = EvalRunRequest(models=["demo-model"], suite="smoke", judge_mode="decision")
        available = {"demo-model": {"provider": "mock", "model_name": "demo-model"}}

        with patch("api.services.config_service.ConfigService.get_models", return_value=available):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(module.start_evaluation(request, service))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail["error_code"], "decision_model_required")


class CustomMetricDecisionContractTests(unittest.TestCase):
    def test_evaluate_with_decision_model_uses_score_question(self):
        from api.schemas.custom_metrics import EvaluateCaseRequest
        from api.services.custom_metric_service import CustomMetricService

        def factory(model_key, config_path):
            return MockDecisionClient(overrides={"metric": 0.75})

        svc = CustomMetricService(config_path="config/models.yaml", decision_client_factory=factory)
        rec = svc.create("empathy", "Rate how empathetic the response is")
        resp = svc.evaluate(
            rec.metric_id,
            [EvaluateCaseRequest(question="q", answer="a", expected_answer="")],
            judge_model="demo-jev",
            decision_type="score",
        )
        self.assertEqual(resp.results[0].score, 0.75)
        self.assertIn("[jev]", resp.results[0].reasoning)

    def test_llm_judge_model_unaffected(self):
        from api.schemas.custom_metrics import EvaluateCaseRequest
        from api.services.custom_metric_service import CustomMetricService

        svc = CustomMetricService(config_path="config/models.yaml")
        rec = svc.create("empathy", "Rate how empathetic the response is")

        def llm_fn(messages):
            return '{"score": 0.5, "reasoning": "ok"}'

        resp = svc.evaluate(
            rec.metric_id,
            [EvaluateCaseRequest(question="q", answer="a", expected_answer="")],
            llm_fn=llm_fn,
        )
        self.assertEqual(resp.results[0].score, 0.5)


if __name__ == "__main__":
    unittest.main()
