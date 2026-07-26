"""Integration-level contract tests for the new Tier-1 embedding test runners on
EvaluationPipeline: run_embedding_pair_classification_test and
run_embedding_bitext_mining_test. Verifies the wiring between the dataset shape,
the (real, not mocked) embedding model, the evaluator classes, and the returned
summary dict — including the newly-activated `embedding_health` diagnostic.
"""
import hashlib
import sys
import threading
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
        self.test_config = {"concurrent_items": 3}
        self._llm_call_semaphore = threading.Semaphore(8)
        self._PREFIX_SENSITIVITY_QUERY_PREFIX = module.EvaluationPipeline._PREFIX_SENSITIVITY_QUERY_PREFIX
        self._PREFIX_SENSITIVITY_PASSAGE_PREFIX = module.EvaluationPipeline._PREFIX_SENSITIVITY_PASSAGE_PREFIX
        self._make_progress_ticker = types.MethodType(module.EvaluationPipeline._make_progress_ticker, self)
        self._iter_with_progress = types.MethodType(module.EvaluationPipeline._iter_with_progress, self)
        self._run_items_concurrently = types.MethodType(module.EvaluationPipeline._run_items_concurrently, self)
        self._embedding_health_summary = types.MethodType(module.EvaluationPipeline._embedding_health_summary, self)
        self.run_embedding_pair_classification_test = types.MethodType(
            module.EvaluationPipeline.run_embedding_pair_classification_test, self
        )
        self.run_embedding_bitext_mining_test = types.MethodType(
            module.EvaluationPipeline.run_embedding_bitext_mining_test, self
        )
        self.run_embedding_prefix_sensitivity_test = types.MethodType(
            module.EvaluationPipeline.run_embedding_prefix_sensitivity_test, self
        )
        self.run_embedding_consistency_test = types.MethodType(
            module.EvaluationPipeline.run_embedding_consistency_test, self
        )
        self.run_embedding_long_context_test = types.MethodType(
            module.EvaluationPipeline.run_embedding_long_context_test, self
        )
        self.run_embedding_reranking_test = types.MethodType(
            module.EvaluationPipeline.run_embedding_reranking_test, self
        )
        self.run_embedding_perturbation_stability_test = types.MethodType(
            module.EvaluationPipeline.run_embedding_perturbation_stability_test, self
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


class TestRunEmbeddingPrefixSensitivityTest:
    def test_wires_evaluator_output_into_summary(self, pipeline_stub):
        dataset = [
            {
                "id": "r1",
                "category": "banking",
                "query": "Kredi kartı borcumu nasıl öderim?",
                "positive_docs": ["Kredi kartı borcunuzu mobil uygulamadan ödeyebilirsiniz."],
                "hard_negatives": ["Kredi kartı limitinizi artırmak için başvuru yapabilirsiniz."],
                "random_negatives": ["Yarın hava güneşli olacak."],
            },
        ]

        result = pipeline_stub.run_embedding_prefix_sensitivity_test(
            _FakeEmbeddingModel(), dataset, "embedding_prefix_sensitivity"
        )

        assert result["test_name"] == "embedding_prefix_sensitivity"
        assert len(result["results"]) == 1
        summary = result["summary"]
        assert summary["total_tests"] == 1
        assert 0.0 <= summary["ndcg_at_10_raw"] <= 1.0
        assert 0.0 <= summary["ndcg_at_10_prefixed"] <= 1.0
        assert summary["overall_score"] == summary["ndcg_at_10_raw"]
        assert "prefix_sensitivity_delta_ndcg" in summary
        assert "prefix_sensitivity_delta_mrr" in summary
        assert "raw" in result["detailed_metrics"] and "prefixed" in result["detailed_metrics"]

    def test_progress_ticks_across_both_conditions(self, pipeline_stub):
        # Regression guard: this test's two full passes over the dataset must not
        # leave run.progress frozen mid-way through the second pass (the exact bug
        # class fixed earlier for other multi-pass test runners).
        dataset = [
            {"id": f"r{i}", "category": "c", "query": f"soru {i}",
             "positive_docs": [f"doğru cevap {i}"], "hard_negatives": [], "random_negatives": []}
            for i in range(3)
        ]

        class _FakeRun:
            def __init__(self):
                self.progress = 0.0

        pipeline_stub._run = _FakeRun()
        pipeline_stub._progress_test_idx = 0
        pipeline_stub._progress_total_tests = 1

        pipeline_stub.run_embedding_prefix_sensitivity_test(
            _FakeEmbeddingModel(), dataset, "embedding_prefix_sensitivity"
        )

        assert pipeline_stub._run.progress == pytest.approx(1.0)
        pipeline_stub._run = None


class TestRunEmbeddingConsistencyTest:
    def test_wires_evaluator_output_into_summary(self, pipeline_stub):
        dataset = [
            {"id": "c1", "category": "short", "text": "Merhaba dünya"},
            {"id": "c2", "category": "short", "text": "Test cümlesi bir"},
            {"id": "c3", "category": "short", "text": "Test cümlesi iki"},
        ]

        result = pipeline_stub.run_embedding_consistency_test(
            _FakeEmbeddingModel(), dataset, "embedding_consistency"
        )

        assert result["test_name"] == "embedding_consistency"
        assert len(result["results"]) == 3
        summary = result["summary"]
        assert summary["total_tests"] == 3
        assert 0.0 <= summary["avg_batch_consistency"] <= 1.0
        assert 0.0 <= summary["avg_order_consistency"] <= 1.0
        assert summary["overall_score"] == min(summary["avg_batch_consistency"], summary["avg_order_consistency"])
        assert "embedding_health" in summary

    def test_deterministic_model_scores_near_perfect_consistency(self, pipeline_stub):
        # The stub embedder is a pure function of text content (no real batching/padding
        # effects), so a well-behaved embedder should score ~1.0 on both consistency axes.
        dataset = [
            {"id": f"c{i}", "category": "short", "text": f"cümle numara {i}"}
            for i in range(5)
        ]

        result = pipeline_stub.run_embedding_consistency_test(
            _FakeEmbeddingModel(), dataset, "embedding_consistency"
        )

        assert result["summary"]["avg_batch_consistency"] == pytest.approx(1.0, abs=1e-6)
        assert result["summary"]["avg_order_consistency"] == pytest.approx(1.0, abs=1e-6)


class TestRunEmbeddingLongContextTest:
    def test_wires_evaluator_output_into_summary(self, pipeline_stub):
        dataset = [
            {
                "id": "lc1",
                "category": "banking",
                "query": "Hesap açılışı için hangi belgeler gerekiyor?",
                "signal_sentence": "Hesap açılışı için kimlik belgesi ve ikametgah belgesi gerekmektedir.",
                "filler_text": "Bankacılık hizmetleri sürekli gelişmektedir. " * 20,
            },
        ]

        result = pipeline_stub.run_embedding_long_context_test(
            _FakeEmbeddingModel(), dataset, "embedding_long_context"
        )

        assert result["test_name"] == "embedding_long_context"
        assert len(result["results"]) == 1
        summary = result["summary"]
        assert summary["total_tests"] == 1
        assert -1.0 <= summary["avg_similarity_signal_first"] <= 1.0
        assert -1.0 <= summary["avg_similarity_signal_last"] <= 1.0
        assert summary["overall_score"] == summary["avg_similarity_signal_last"]
        assert "embedding_health" in summary


class TestRunEmbeddingRerankingTest:
    def test_wires_evaluator_output_into_summary(self, pipeline_stub):
        dataset = [
            {
                "id": "rr1",
                "category": "banking",
                "query": "Kredi kartı borcumu nasıl öderim?",
                "candidates": [
                    {"text": "Kredi kartı borcumu nasıl öderim?", "relevance": 2},
                    {"text": "Kredi kartı limitimi nasıl artırırım?", "relevance": 1},
                    {"text": "Yarın hava nasıl olacak?", "relevance": 0},
                ],
            },
        ]

        result = pipeline_stub.run_embedding_reranking_test(
            _FakeEmbeddingModel(), dataset, "embedding_reranking"
        )

        assert result["test_name"] == "embedding_reranking"
        assert len(result["results"]) == 1
        summary = result["summary"]
        assert summary["total_tests"] == 1
        assert 0.0 <= summary["avg_ndcg"] <= 1.0
        assert summary["overall_score"] == summary["avg_ndcg"]
        assert result["results"][0]["n_candidates"] == 3
        assert "embedding_health" in summary

    def test_identical_text_ranks_as_most_relevant(self, pipeline_stub):
        # The stub embedder maps identical strings to identical vectors, so the
        # candidate matching the query verbatim should rank #1.
        dataset = [
            {
                "id": "rr1",
                "category": "sanity",
                "query": "test sorgusu",
                "candidates": [
                    {"text": "alakasız metin", "relevance": 0},
                    {"text": "test sorgusu", "relevance": 2},
                    {"text": "başka alakasız metin", "relevance": 0},
                ],
            },
        ]

        result = pipeline_stub.run_embedding_reranking_test(
            _FakeEmbeddingModel(), dataset, "embedding_reranking"
        )

        assert result["results"][0]["top1_is_most_relevant"] is True


class TestRunEmbeddingPerturbationStabilityTest:
    def test_wires_evaluator_output_into_summary(self, pipeline_stub):
        dataset = [
            {
                "id": "ps1",
                "category": "banking",
                "query_original": "Kredi kartı borcumu nasıl öderim?",
                "query_perturbed": "Kredi katı borcumu nasl öderim?",
                "perturbation_type": "typo",
                "positive_docs": ["Kredi kartı borcunuzu mobil uygulamadan ödeyebilirsiniz."],
                "hard_negatives": ["Kredi kartı limitinizi artırabilirsiniz."],
                "random_negatives": ["Yarın hava güneşli olacak."],
            },
        ]

        result = pipeline_stub.run_embedding_perturbation_stability_test(
            _FakeEmbeddingModel(), dataset, "embedding_perturbation_stability"
        )

        assert result["test_name"] == "embedding_perturbation_stability"
        assert len(result["results"]) == 1
        summary = result["summary"]
        assert summary["total_tests"] == 1
        assert 0.0 <= summary["avg_top1_stable"] <= 1.0
        assert 0.0 <= summary["avg_top_k_overlap"] <= 1.0
        assert summary["overall_score"] == summary["avg_top_k_overlap"]
        assert result["results"][0]["perturbation_type"] == "typo"
        assert "embedding_health" in summary

    def test_identical_query_and_perturbed_query_are_fully_stable(self, pipeline_stub):
        # Using the exact same string for both "original" and "perturbed" is a
        # sanity floor: stability must be perfect since nothing actually changed.
        dataset = [
            {
                "id": "ps1",
                "category": "sanity",
                "query_original": "aynı sorgu",
                "query_perturbed": "aynı sorgu",
                "positive_docs": ["doğru cevap"],
                "hard_negatives": ["yanlış ama benzer cevap"],
                "random_negatives": ["alakasız metin"],
            },
        ]

        result = pipeline_stub.run_embedding_perturbation_stability_test(
            _FakeEmbeddingModel(), dataset, "embedding_perturbation_stability"
        )

        assert result["summary"]["avg_top1_stable"] == 1.0
        assert result["summary"]["avg_top_k_overlap"] == pytest.approx(1.0)
