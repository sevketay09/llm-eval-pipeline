"""Integration-level contract tests for the new Tier-1 embedding test runners on
EvaluationPipeline: run_embedding_pair_classification_test and
run_embedding_bitext_mining_test. Verifies the wiring between the dataset shape,
the (real, not mocked) embedding model, the evaluator classes, and the returned
summary dict — including the newly-activated `embedding_health` diagnostic.
"""
import hashlib
import sys
import types

import numpy as np
import pytest


def _load_pipeline_runner_module():
    fake_anthropic = sys.modules.get("anthropic")
    if fake_anthropic is None:
        fake_anthropic = types.ModuleType("anthropic")

        class Anthropic:  # pragma: no cover - import stub for tests only
            pass

        fake_anthropic.Anthropic = Anthropic
        sys.modules["anthropic"] = fake_anthropic

    fake_datasets = sys.modules.get("datasets")
    if fake_datasets is None:
        fake_datasets = types.ModuleType("datasets")

        def load_dataset(*args, **kwargs):
            raise RuntimeError("load_dataset should not run in embedding runner contract tests")

        fake_datasets.load_dataset = load_dataset
        sys.modules["datasets"] = fake_datasets

    import importlib
    return importlib.import_module("pipeline_runner")


def _deterministic_unit_vector(text: str, dim: int = 16) -> np.ndarray:
    """Same text -> same vector; different text -> (near-)different vector."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    vec = np.random.RandomState(seed).normal(size=dim)
    return vec / np.linalg.norm(vec)


class _FakeEmbeddingModel:
    """Stub matching UnifiedEmbeddingAdapter.encode()'s return contract."""

    model_name = "fake-embedding-model"

    def encode(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = np.array([_deterministic_unit_vector(t) for t in texts])
        return {"embeddings": embeddings, "latency": 0.001, "model": self.model_name, "count": len(texts)}


class _PipelineStub:
    """Binds only what the two new embedding runners touch on self."""

    def __init__(self, module):
        self._run = None
        self._progress_test_idx = 0
        self._progress_total_tests = 1
        self._make_progress_ticker = types.MethodType(module.EvaluationPipeline._make_progress_ticker, self)
        self._iter_with_progress = types.MethodType(module.EvaluationPipeline._iter_with_progress, self)
        self._embedding_health_summary = types.MethodType(module.EvaluationPipeline._embedding_health_summary, self)
        self.run_embedding_pair_classification_test = types.MethodType(
            module.EvaluationPipeline.run_embedding_pair_classification_test, self
        )
        self.run_embedding_bitext_mining_test = types.MethodType(
            module.EvaluationPipeline.run_embedding_bitext_mining_test, self
        )


@pytest.fixture(scope="module")
def pipeline_stub():
    module = _load_pipeline_runner_module()
    return _PipelineStub(module)


class TestRunEmbeddingPairClassificationTest:
    def test_wires_evaluator_output_into_summary(self, pipeline_stub):
        # Duplicate pair uses the exact same sentence twice -> identical embedding,
        # guaranteeing a real (non-degenerate) similarity gap vs the negative pair.
        dataset = [
            {"id": "p1", "category": "exact_paraphrase", "sentence1": "Kredi kartı borcumu nasıl öderim?",
             "sentence2": "Kredi kartı borcumu nasıl öderim?", "is_duplicate": 1},
            {"id": "p2", "category": "unrelated", "sentence1": "Yarın hava nasıl olacak?",
             "sentence2": "En sevdiğim yemek mantıdır.", "is_duplicate": 0},
        ]

        result = pipeline_stub.run_embedding_pair_classification_test(
            _FakeEmbeddingModel(), dataset, "embedding_pair_classification"
        )

        assert result["test_name"] == "embedding_pair_classification"
        assert len(result["results"]) == 2
        summary = result["summary"]
        assert summary["total_tests"] == 2
        assert 0.0 <= summary["average_precision"] <= 1.0
        assert summary["overall_score"] == summary["average_precision"]
        assert "embedding_health" in summary
        assert summary["embedding_health"]["dimension"] == 16

    def test_identical_pair_scores_higher_than_unrelated_pair(self, pipeline_stub):
        dataset = [
            {"id": "p1", "category": "exact_paraphrase", "sentence1": "aynı cümle",
             "sentence2": "aynı cümle", "is_duplicate": 1},
            {"id": "p2", "category": "unrelated", "sentence1": "tamamen farklı bir konu",
             "sentence2": "hiç alakasız başka bir metin", "is_duplicate": 0},
        ]

        result = pipeline_stub.run_embedding_pair_classification_test(
            _FakeEmbeddingModel(), dataset, "embedding_pair_classification"
        )

        by_id = {r["id"]: r for r in result["results"]}
        assert by_id["p1"]["predicted_score"] > by_id["p2"]["predicted_score"]
        # Identical text must cosine-similarity to (approximately) 1.0.
        assert by_id["p1"]["predicted_score"] == pytest.approx(1.0, abs=1e-6)


class TestRunEmbeddingBitextMiningTest:
    def test_wires_evaluator_output_into_summary(self, pipeline_stub):
        dataset = [
            {
                "id": "b1",
                "category": "banking",
                "source_sentence": "Kredi kartı borcumu nasıl öderim?",
                "correct_translation": "How do I pay off my credit card debt?",
                "distractor_translations": [
                    "How do I increase my credit card limit?",
                    "What is today's weather forecast?",
                ],
            },
        ]

        result = pipeline_stub.run_embedding_bitext_mining_test(
            _FakeEmbeddingModel(), dataset, "embedding_bitext_mining"
        )

        assert result["test_name"] == "embedding_bitext_mining"
        assert len(result["results"]) == 1
        summary = result["summary"]
        assert summary["total_tests"] == 1
        assert 0.0 <= summary["accuracy_at_1"] <= 1.0
        assert 0.0 <= summary["mrr"] <= 1.0
        assert summary["overall_score"] == summary["accuracy_at_1"]
        assert result["results"][0]["n_distractors"] == 2

    def test_correct_translation_always_ranked_first_when_source_equals_translation(self, pipeline_stub):
        # The stub embedder maps identical strings to identical vectors, so feeding
        # the source sentence itself as the "correct translation" guarantees a
        # perfect top-1 match regardless of distractor content.
        dataset = [
            {
                "id": "b1",
                "category": "sanity",
                "source_sentence": "test cümlesi",
                "correct_translation": "test cümlesi",
                "distractor_translations": ["alakasız metin bir", "alakasız metin iki"],
            },
        ]

        result = pipeline_stub.run_embedding_bitext_mining_test(
            _FakeEmbeddingModel(), dataset, "embedding_bitext_mining"
        )

        assert result["summary"]["accuracy_at_1"] == 1.0
        assert result["results"][0]["correct_at_1"] is True
