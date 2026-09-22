"""Contract tests for decisions/rag.py and its wiring into
analysis/rag_eval.py and the RAG eval API. Offline only — MockDecisionClient,
no network.
"""
from __future__ import annotations

import unittest

from analysis.rag_eval import evaluate_rag_case
from decisions.mock_client import MockDecisionClient
from decisions.rag import make_rag_decision_fn


def _case(n_contexts=3, expected=False):
    return {
        "question": "What is the return policy?",
        "contexts": [f"context chunk {i}" for i in range(n_contexts)],
        "answer": "You can return items within 30 days.",
        "expected_answer": "30-day return window." if expected else None,
    }


class MakeRagDecisionFnContractTests(unittest.TestCase):
    def test_schema_matches_lexical_case(self):
        client = MockDecisionClient()
        decision_fn = make_rag_decision_fn(client)
        lexical = evaluate_rag_case(_case())
        decisioned = evaluate_rag_case(_case(), decision_fn=decision_fn)

        for key in ("context_precision", "context_recall", "faithfulness", "answer_relevance"):
            self.assertEqual(set(decisioned[key].keys()), set(lexical[key].keys()))

    def test_scoring_mode_is_decision(self):
        client = MockDecisionClient()
        decision_fn = make_rag_decision_fn(client)
        result = evaluate_rag_case(_case(), decision_fn=decision_fn)
        self.assertIn("decision_fault", result["fault_isolation"])
        self.assertIn(result["fault_isolation"]["decision_fault"]["fault"], {"retriever", "generator", "both", "none"})

    def test_context_recall_none_without_expected_answer(self):
        client = MockDecisionClient()
        decision_fn = make_rag_decision_fn(client)
        result = evaluate_rag_case(_case(expected=False), decision_fn=decision_fn)
        self.assertIsNone(result["context_recall"]["recall"])

    def test_context_recall_present_with_expected_answer(self):
        client = MockDecisionClient(overrides={"recall": 0.8})
        decision_fn = make_rag_decision_fn(client)
        result = evaluate_rag_case(_case(expected=True), decision_fn=decision_fn)
        self.assertEqual(result["context_recall"]["recall"], 0.8)

    def test_chunk_count_capped_at_max_chunks(self):
        class CountingClient(MockDecisionClient):
            def _decide_raw(self, state, questions):
                self.last_questions = dict(questions)
                return super()._decide_raw(state, questions)

        client = CountingClient()
        decision_fn = make_rag_decision_fn(client, max_chunks=20)
        decision_fn(_case(n_contexts=25))
        ctx_questions = [k for k in client.last_questions if k.startswith("ctx_")]
        self.assertEqual(len(ctx_questions), 20)

    def test_decision_fn_none_leaves_output_unchanged(self):
        result = evaluate_rag_case(_case())
        self.assertNotIn("decision_fault", result["fault_isolation"])


class RagEvalServiceDecisionContractTests(unittest.TestCase):
    def test_decision_priority_over_embedding(self):
        from api.schemas.rag_eval import RagContext
        from api.services.rag_eval_service import RagEvalService

        calls = {"embed": 0}

        def embed_adapter_factory(model_key, config_path):
            calls["embed"] += 1
            raise AssertionError("embedding adapter should not be built when decision_model is set")

        def decision_client_factory(model_key, config_path):
            return MockDecisionClient()

        svc = RagEvalService(
            embedding_adapter_factory=embed_adapter_factory,
            decision_client_factory=decision_client_factory,
        )
        response = svc.evaluate(
            question="q",
            contexts=[RagContext(text="ctx")],
            answer="a",
            embedding_model="some-embedding-model",
            decision_model="demo-jev",
        )
        self.assertEqual(response.scoring_mode, "decision")
        self.assertEqual(response.decision_model, "demo-jev")
        self.assertIsNone(response.embedding_model)
        self.assertEqual(calls["embed"], 0)


if __name__ == "__main__":
    unittest.main()
