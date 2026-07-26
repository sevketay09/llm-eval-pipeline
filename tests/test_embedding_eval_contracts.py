"""Contract tests for evaluators/embedding_eval.py — Tier-1 (PairClassificationEvaluator,
BitextMiningEvaluator) and Tier-2 (BatchConsistencyEvaluator) embedding test types, plus
the previously-unused EmbeddingQualityMetrics now wired into every embedding test's summary.
"""
import numpy as np
import pytest

from evaluators.embedding_eval import (
    PairClassificationEvaluator,
    BitextMiningEvaluator,
    BatchConsistencyEvaluator,
    EmbeddingQualityMetrics,
)


def _unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.normal(size=dim)
    return vec / np.linalg.norm(vec)


class TestPairClassificationEvaluator:
    def test_perfect_separation_yields_high_average_precision(self):
        # Duplicates get near-identical embeddings, non-duplicates get orthogonal-ish ones.
        base = _unit_vector(0)
        embeddings1 = np.array([base, base, _unit_vector(10), _unit_vector(20)])
        embeddings2 = np.array([base, base, _unit_vector(11), _unit_vector(21)])
        labels = [1, 1, 0, 0]

        result = PairClassificationEvaluator.evaluate(embeddings1, embeddings2, labels)

        assert result["average_precision"] == pytest.approx(1.0, abs=1e-6)
        assert result["accuracy_at_best_threshold"] >= 0.75
        assert result["mean_positive_score"] > result["mean_negative_score"]

    def test_single_class_returns_zero_average_precision(self):
        embeddings1 = np.array([_unit_vector(0), _unit_vector(1)])
        embeddings2 = np.array([_unit_vector(2), _unit_vector(3)])
        labels = [0, 0]

        result = PairClassificationEvaluator.evaluate(embeddings1, embeddings2, labels)

        assert result["average_precision"] == 0.0
        assert result["mean_positive_score"] is None
        assert result["mean_negative_score"] is not None

    def test_best_threshold_is_within_valid_similarity_range(self):
        embeddings1 = np.array([_unit_vector(i) for i in range(6)])
        embeddings2 = np.array([_unit_vector(i + 1) for i in range(6)])
        labels = [1, 0, 1, 0, 1, 0]

        result = PairClassificationEvaluator.evaluate(embeddings1, embeddings2, labels)

        assert -1.0 <= result["best_threshold"] <= 1.0
        assert 0.0 <= result["accuracy_at_best_threshold"] <= 1.0


class TestBitextMiningEvaluator:
    def test_correct_translation_ranked_first(self):
        source = _unit_vector(0)
        correct = _unit_vector(0)  # near-identical -> should rank #1
        distractors = np.array([_unit_vector(5), _unit_vector(6), _unit_vector(7)])
        candidates = np.vstack([correct, distractors])

        result = BitextMiningEvaluator.evaluate_single(source, candidates, correct_index=0)

        assert result["rank"] == 1
        assert result["correct_at_1"] is True
        assert result["reciprocal_rank"] == 1.0
        assert result["correct_similarity"] > result["top_distractor_similarity"]

    def test_correct_translation_ranked_lower_when_distractor_is_closer(self):
        source = _unit_vector(0)
        correct = _unit_vector(50)  # unrelated direction
        closer_distractor = _unit_vector(0)  # near-identical to source -> outranks correct
        other_distractors = np.array([_unit_vector(60), _unit_vector(61)])
        candidates = np.vstack([correct, closer_distractor, other_distractors])

        result = BitextMiningEvaluator.evaluate_single(source, candidates, correct_index=0)

        assert result["rank"] > 1
        assert result["correct_at_1"] is False
        assert result["reciprocal_rank"] < 1.0

    def test_aggregate_computes_accuracy_and_mrr(self):
        results = [
            {"rank": 1, "correct_at_1": True, "reciprocal_rank": 1.0, "correct_similarity": 0.9, "top_distractor_similarity": 0.4},
            {"rank": 2, "correct_at_1": False, "reciprocal_rank": 0.5, "correct_similarity": 0.5, "top_distractor_similarity": 0.6},
        ]

        aggregated = BitextMiningEvaluator.aggregate(results)

        assert aggregated["accuracy_at_1"] == pytest.approx(0.5)
        assert aggregated["mrr"] == pytest.approx(0.75)
        assert aggregated["avg_margin"] == pytest.approx(((0.9 - 0.4) + (0.5 - 0.6)) / 2)


class TestBatchConsistencyEvaluator:
    def test_identical_embeddings_score_perfect_similarity(self):
        embeddings = np.array([_unit_vector(i) for i in range(5)])

        result = BatchConsistencyEvaluator.compare(embeddings, embeddings)

        assert result["mean_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert result["min_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert result["pass_rate"] == 1.0

    def test_divergent_embeddings_fail_tolerance(self):
        embeddings_a = np.array([_unit_vector(i) for i in range(3)])
        embeddings_b = np.array([_unit_vector(i + 100) for i in range(3)])

        result = BatchConsistencyEvaluator.compare(embeddings_a, embeddings_b, tolerance=0.999)

        assert result["mean_similarity"] < 0.999
        assert result["pass_rate"] < 1.0

    def test_small_perturbation_still_passes_loose_tolerance(self):
        base = _unit_vector(0)
        embeddings_a = np.array([base, base, base])
        rng = np.random.RandomState(1)
        embeddings_b = np.array([base + rng.normal(scale=1e-5, size=8) for _ in range(3)])

        result = BatchConsistencyEvaluator.compare(embeddings_a, embeddings_b, tolerance=0.99)

        assert result["pass_rate"] == 1.0

    def test_aggregate_takes_worst_case_as_overall_score(self):
        batch_vs_individual = {"mean_similarity": 0.9995, "min_similarity": 0.998, "pass_rate": 1.0}
        order_vs_reordered = {"mean_similarity": 0.85, "min_similarity": 0.7, "pass_rate": 0.4}

        aggregated = BatchConsistencyEvaluator.aggregate(batch_vs_individual, order_vs_reordered)

        assert aggregated["overall_score"] == pytest.approx(0.85)
        assert aggregated["avg_batch_consistency"] == pytest.approx(0.9995)
        assert aggregated["avg_order_consistency"] == pytest.approx(0.85)


class TestEmbeddingQualityMetricsActivation:
    def test_embedding_statistics_shape(self):
        embeddings = np.array([_unit_vector(i) for i in range(5)])
        stats = EmbeddingQualityMetrics.compute_embedding_statistics(embeddings)

        assert stats["dimension"] == 8
        assert stats["mean_norm"] == pytest.approx(1.0, abs=1e-6)  # unit vectors

    def test_intra_list_similarity_low_for_diverse_embeddings(self):
        embeddings = np.array([_unit_vector(i) for i in range(20)])
        ils = EmbeddingQualityMetrics.compute_intra_list_similarity(embeddings)

        assert -1.0 <= ils <= 1.0

    def test_intra_list_similarity_high_for_near_identical_embeddings(self):
        base = _unit_vector(0)
        embeddings = np.array([base + np.random.RandomState(i).normal(scale=0.001, size=8) for i in range(5)])
        ils = EmbeddingQualityMetrics.compute_intra_list_similarity(embeddings)

        assert ils > 0.9
