"""Contract tests for api/routers/rag_eval.py."""
from __future__ import annotations
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routers.rag_eval as rag_eval_module
from api.routers.rag_eval import router as rag_eval_router
from api.services.rag_eval_service import RagEvalService


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(rag_eval_router, prefix="/api")
    with TestClient(app) as c:
        yield c


def _payload(**overrides):
    base = {
        "question": "What is the return policy?",
        "contexts": [
            {"text": "Returns are accepted within 30 days of purchase."},
            {"text": "Contact support for refunds."},
        ],
        "answer": "You can return items within 30 days.",
        "expected_answer": "Returns accepted within 30 days.",
    }
    base.update(overrides)
    return base


class TestRagEvalEndpoint:
    def test_evaluate_200(self, client):
        r = client.post("/api/rag-eval", json=_payload())
        assert r.status_code == 200

    def test_requires_question(self, client):
        r = client.post("/api/rag-eval", json={**_payload(), "question": ""})
        assert r.status_code == 422

    def test_requires_contexts(self, client):
        r = client.post("/api/rag-eval", json={**_payload(), "contexts": []})
        assert r.status_code == 422

    def test_requires_answer(self, client):
        r = client.post("/api/rag-eval", json={**_payload(), "answer": ""})
        assert r.status_code == 422

    def test_returns_all_component_scores(self, client):
        body = client.post("/api/rag-eval", json=_payload()).json()
        for field in ("context_precision", "context_recall", "faithfulness", "answer_relevance", "overall_score"):
            assert field in body
            assert isinstance(body[field], float)

    def test_scores_in_range(self, client):
        body = client.post("/api/rag-eval", json=_payload()).json()
        for field in ("context_precision", "context_recall", "faithfulness", "answer_relevance", "overall_score"):
            assert 0.0 <= body[field] <= 1.0

    def test_fault_component_present(self, client):
        body = client.post("/api/rag-eval", json=_payload()).json()
        assert "fault_component" in body
        assert body["fault_component"] in ("retriever", "generator", "both", "none")

    def test_overall_score_is_the_underlying_weighted_score(self, client):
        # Not a flat average of the 4 components — analysis.rag_eval.evaluate_rag_case
        # weights faithfulness highest (hallucination is the worst failure mode) and
        # renormalizes when context_recall is absent (no expected_answer given).
        # A prior version of this test asserted a flat-average relationship that only
        # held because a service-layer bug zeroed every component (see
        # api/services/rag_eval_service.py + tests/test_rag_eval_service_contracts.py).
        body = client.post("/api/rag-eval", json=_payload()).json()
        assert abs(body["overall_score"] - body["details"]["overall_rag_score"]) < 0.001
        # This fixture's answer is faithful to and relevant for its context, so the
        # pipeline must not be universally zeroed out — the regression this bug caused.
        assert body["faithfulness"] > 0.0
        assert body["answer_relevance"] > 0.0
        assert body["overall_score"] > 0.0

    def test_without_expected_answer(self, client):
        payload = {k: v for k, v in _payload().items() if k != "expected_answer"}
        r = client.post("/api/rag-eval", json=payload)
        assert r.status_code == 200

    def test_returns_question_in_response(self, client):
        body = client.post("/api/rag-eval", json=_payload()).json()
        assert body["question"] == _payload()["question"]


class TestRagEvalEmbeddingMode:
    """embedding_model wiring at the router level — swaps the module-level
    singleton's factory, same pattern as test_skill_eval_router_contracts.py."""

    def test_unknown_embedding_model_returns_404(self, client):
        def failing_factory(model_key, config_path):
            raise ValueError(f"Embedding model '{model_key}' not found in config")

        original = rag_eval_module._service
        rag_eval_module._service = RagEvalService(embedding_adapter_factory=failing_factory)
        try:
            r = client.post("/api/rag-eval", json=_payload(embedding_model="does-not-exist"))
            assert r.status_code == 404
        finally:
            rag_eval_module._service = original

    def test_embedding_model_wires_through_to_response(self, client):
        class _FakeAdapter:
            def encode(self, texts, normalize=True):
                return {"embeddings": [[1.0, 0.0] for _ in texts]}

        original = rag_eval_module._service
        rag_eval_module._service = RagEvalService(embedding_adapter_factory=lambda k, p: _FakeAdapter())
        try:
            body = client.post("/api/rag-eval", json=_payload(embedding_model="fake-embed")).json()
            assert body["scoring_mode"] == "embedding"
            assert body["embedding_model"] == "fake-embed"
        finally:
            rag_eval_module._service = original
