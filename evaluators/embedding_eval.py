"""
Embedding Model Evaluators
Metrics for evaluating Turkish embedding models without LLM judges
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import get_logger

logger = get_logger(__name__)


class SemanticSimilarityEvaluator:
    """Evaluate embedding model on Semantic Textual Similarity (STS) task"""
    
    @staticmethod
    def evaluate(
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        expected_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Evaluate STS performance using correlation metrics
        
        Args:
            embeddings1: Embeddings for first sentences (n, dim)
            embeddings2: Embeddings for second sentences (n, dim)
            expected_scores: Ground truth similarity scores (n,)
            
        Returns:
            {
                "spearman_correlation": float,
                "pearson_correlation": float,
                "mae": float (Mean Absolute Error),
                "accuracy_at_threshold": Dict[float, float]
            }
        """
        # Compute cosine similarities
        predicted_scores = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            predicted_scores.append(float(sim))
        
        predicted_scores = np.array(predicted_scores)
        expected_scores = np.array(expected_scores)
        
        # Spearman correlation (rank-based)
        spearman, spearman_p = spearmanr(expected_scores, predicted_scores)
        
        # Pearson correlation (linear)
        pearson, pearson_p = pearsonr(expected_scores, predicted_scores)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(expected_scores - predicted_scores))
        
        # Accuracy at different thresholds
        accuracy_thresholds = {}
        for threshold in [0.5, 0.7, 0.9]:
            # Binary classification: similar (>threshold) or not
            expected_binary = (expected_scores > threshold).astype(int)
            predicted_binary = (predicted_scores > threshold).astype(int)
            accuracy = np.mean(expected_binary == predicted_binary)
            accuracy_thresholds[threshold] = float(accuracy)
        
        return {
            "spearman_correlation": float(spearman),
            "spearman_p_value": float(spearman_p),
            "pearson_correlation": float(pearson),
            "pearson_p_value": float(pearson_p),
            "mae": float(mae),
            "rmse": float(np.sqrt(np.mean((expected_scores - predicted_scores) ** 2))),
            "accuracy_at_threshold": accuracy_thresholds,
            "mean_predicted_score": float(np.mean(predicted_scores)),
            "std_predicted_score": float(np.std(predicted_scores))
        }


class RetrievalEvaluator:
    """Evaluate embedding model on information retrieval task"""
    
    @staticmethod
    def compute_ndcg_at_k(
        relevance_scores: List[float],
        k: int
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k
        
        Args:
            relevance_scores: Relevance scores in ranked order
            k: Cutoff position
            
        Returns:
            NDCG@k score
        """
        relevance_scores = np.array(relevance_scores[:k])
        
        # DCG@k
        dcg = relevance_scores[0] + np.sum(
            relevance_scores[1:] / np.log2(np.arange(2, len(relevance_scores) + 1))
        )
        
        # IDCG@k (ideal DCG)
        ideal_scores = np.sort(relevance_scores)[::-1]
        idcg = ideal_scores[0] + np.sum(
            ideal_scores[1:] / np.log2(np.arange(2, len(ideal_scores) + 1))
        )
        
        if idcg == 0:
            return 0.0
        
        return float(dcg / idcg)
    
    @staticmethod
    def compute_mrr(ranks: List[int]) -> float:
        """
        Compute Mean Reciprocal Rank
        
        Args:
            ranks: List of ranks of first relevant document for each query
            
        Returns:
            MRR score
        """
        reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in ranks]
        return float(np.mean(reciprocal_ranks))
    
    @staticmethod
    def evaluate(
        query_embeddings: np.ndarray,
        doc_embeddings_list: List[np.ndarray],
        relevance_labels: List[List[int]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance
        
        Args:
            query_embeddings: Query embeddings (n_queries, dim)
            doc_embeddings_list: List of document embeddings for each query
            relevance_labels: List of relevance labels (1=relevant, 0=not) for each query
            k_values: List of k values for metrics
            
        Returns:
            {
                "ndcg@k": Dict[int, float],
                "recall@k": Dict[int, float],
                "precision@k": Dict[int, float],
                "mrr": float,
                "map": float (Mean Average Precision)
            }
        """
        ndcg_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        precision_scores = {k: [] for k in k_values}
        mrr_ranks = []
        average_precisions = []
        
        for query_emb, doc_embs, labels in zip(query_embeddings, doc_embeddings_list, relevance_labels):
            # Compute similarities
            similarities = cosine_similarity([query_emb], doc_embs)[0]
            
            # Rank documents by similarity (descending)
            ranked_indices = np.argsort(similarities)[::-1]
            ranked_labels = [labels[i] for i in ranked_indices]
            
            # Find rank of first relevant document (for MRR)
            try:
                first_relevant_rank = ranked_labels.index(1) + 1
                mrr_ranks.append(first_relevant_rank)
            except ValueError:
                mrr_ranks.append(0)  # No relevant document found
            
            # Compute metrics at different k values
            for k in k_values:
                top_k_labels = ranked_labels[:k]
                
                # NDCG@k
                ndcg = RetrievalEvaluator.compute_ndcg_at_k(top_k_labels, k)
                ndcg_scores[k].append(ndcg)
                
                # Recall@k
                n_relevant_total = sum(labels)
                n_relevant_retrieved = sum(top_k_labels)
                recall = n_relevant_retrieved / n_relevant_total if n_relevant_total > 0 else 0
                recall_scores[k].append(recall)
                
                # Precision@k
                precision = n_relevant_retrieved / k if k > 0 else 0
                precision_scores[k].append(precision)
            
            # Average Precision
            precisions_at_relevant = []
            n_relevant_seen = 0
            for i, label in enumerate(ranked_labels, 1):
                if label == 1:
                    n_relevant_seen += 1
                    precision_at_i = n_relevant_seen / i
                    precisions_at_relevant.append(precision_at_i)
            
            avg_precision = np.mean(precisions_at_relevant) if precisions_at_relevant else 0
            average_precisions.append(avg_precision)
        
        # Aggregate results
        return {
            "ndcg": {k: float(np.mean(scores)) for k, scores in ndcg_scores.items()},
            "recall": {k: float(np.mean(scores)) for k, scores in recall_scores.items()},
            "precision": {k: float(np.mean(scores)) for k, scores in precision_scores.items()},
            "mrr": RetrievalEvaluator.compute_mrr(mrr_ranks),
            "map": float(np.mean(average_precisions))
        }


class ClusteringEvaluator:
    """Evaluate embedding quality through clustering metrics"""
    
    @staticmethod
    def evaluate_term_clustering(
        term_embedding: np.ndarray,
        similar_embeddings: np.ndarray,
        dissimilar_embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate if model correctly clusters similar/dissimilar terms
        
        Args:
            term_embedding: Target term embedding (dim,)
            similar_embeddings: Embeddings of similar terms (n_similar, dim)
            dissimilar_embeddings: Embeddings of dissimilar terms (n_dissim, dim)
            
        Returns:
            {
                "avg_similar_score": float,
                "avg_dissimilar_score": float,
                "separation_margin": float,
                "accuracy": float
            }
        """
        # Compute similarities
        similar_scores = cosine_similarity([term_embedding], similar_embeddings)[0]
        dissimilar_scores = cosine_similarity([term_embedding], dissimilar_embeddings)[0]
        
        avg_similar = float(np.mean(similar_scores))
        avg_dissimilar = float(np.mean(dissimilar_scores))
        
        # Separation margin (higher is better)
        margin = avg_similar - avg_dissimilar
        
        # Accuracy: % of similar terms ranked higher than dissimilar terms
        correct_rankings = 0
        total_comparisons = 0
        for sim_score in similar_scores:
            for dissim_score in dissimilar_scores:
                if sim_score > dissim_score:
                    correct_rankings += 1
                total_comparisons += 1
        
        accuracy = correct_rankings / total_comparisons if total_comparisons > 0 else 0
        
        return {
            "avg_similar_score": avg_similar,
            "avg_dissimilar_score": avg_dissimilar,
            "separation_margin": float(margin),
            "accuracy": float(accuracy),
            "min_similar_score": float(np.min(similar_scores)),
            "max_dissimilar_score": float(np.max(dissimilar_scores))
        }
    
    @staticmethod
    def aggregate_clustering_results(
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate clustering results across multiple terms"""
        return {
            "avg_similar_score": float(np.mean([r["avg_similar_score"] for r in results])),
            "avg_dissimilar_score": float(np.mean([r["avg_dissimilar_score"] for r in results])),
            "avg_separation_margin": float(np.mean([r["separation_margin"] for r in results])),
            "avg_accuracy": float(np.mean([r["accuracy"] for r in results])),
            "pass_rate": float(np.mean([r["separation_margin"] > 0.1 for r in results]))
        }


class PairClassificationEvaluator:
    """Evaluate embedding model on binary pair classification (duplicate/paraphrase detection).

    Unlike STS (graded similarity), this scores a binary decision: is this pair a
    duplicate/paraphrase or not? Mirrors MTEB's PairClassification task — cosine
    similarity is the score, and the best-threshold Average Precision is the metric,
    so the evaluator doesn't need to guess a similarity cutoff up front.
    """

    @staticmethod
    def evaluate(
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        labels: List[int],
    ) -> Dict[str, Any]:
        """
        Args:
            embeddings1: Embeddings for the first item in each pair (n, dim)
            embeddings2: Embeddings for the second item in each pair (n, dim)
            labels: 1 if the pair is a duplicate/paraphrase, 0 otherwise (n,)

        Returns:
            {
                "average_precision": float,
                "best_threshold": float,
                "accuracy_at_best_threshold": float,
                "predicted_scores": List[float],
            }
        """
        predicted_scores = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            predicted_scores.append(float(sim))

        predicted_scores = np.array(predicted_scores)
        labels_arr = np.array(labels)

        if len(set(labels)) < 2:
            # Average precision is undefined with only one class present.
            average_precision = 0.0
        else:
            average_precision = float(average_precision_score(labels_arr, predicted_scores))

        # Grid search the cosine-similarity threshold that maximizes accuracy —
        # simple, deterministic, and matches how a real dedup/FAQ-matching
        # pipeline would pick an operating point.
        candidate_thresholds = np.unique(np.round(predicted_scores, 3))
        best_threshold = 0.5
        best_accuracy = 0.0
        for threshold in candidate_thresholds:
            predicted_labels = (predicted_scores >= threshold).astype(int)
            accuracy = float(np.mean(predicted_labels == labels_arr))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = float(threshold)

        return {
            "average_precision": average_precision,
            "best_threshold": best_threshold,
            "accuracy_at_best_threshold": best_accuracy,
            "mean_positive_score": float(np.mean(predicted_scores[labels_arr == 1])) if np.any(labels_arr == 1) else None,
            "mean_negative_score": float(np.mean(predicted_scores[labels_arr == 0])) if np.any(labels_arr == 0) else None,
        }


class BitextMiningEvaluator:
    """Evaluate embedding model on cross-lingual bitext mining (translation retrieval).

    For each source-language sentence, the model must pick its true translation out of
    a small candidate pool (the correct translation plus several distractor sentences
    on a similar topic). This is the task LaBSE-style models are explicitly trained
    for, and it's a different skill from STS: it tests whether the *correct* match
    ranks first among close competitors, not just whether similarity scores correlate.
    """

    @staticmethod
    def evaluate_single(
        source_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        correct_index: int,
    ) -> Dict[str, Any]:
        similarities = cosine_similarity([source_embedding], candidate_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1]
        rank = int(np.where(ranked_indices == correct_index)[0][0]) + 1  # 1-based rank
        return {
            "rank": rank,
            "correct_at_1": rank == 1,
            "reciprocal_rank": 1.0 / rank,
            "correct_similarity": float(similarities[correct_index]),
            "top_distractor_similarity": float(
                max((s for i, s in enumerate(similarities) if i != correct_index), default=0.0)
            ),
        }

    @staticmethod
    def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "accuracy_at_1": float(np.mean([r["correct_at_1"] for r in results])),
            "mrr": float(np.mean([r["reciprocal_rank"] for r in results])),
            "avg_correct_similarity": float(np.mean([r["correct_similarity"] for r in results])),
            "avg_top_distractor_similarity": float(np.mean([r["top_distractor_similarity"] for r in results])),
            "avg_margin": float(
                np.mean([r["correct_similarity"] - r["top_distractor_similarity"] for r in results])
            ),
        }


class BatchConsistencyEvaluator:
    """Evaluate whether encode() is invariant to batch composition and item order.

    A production pipeline encodes documents in large batches for throughput, but this
    must not silently change the vector a document gets versus encoding it alone or in
    a different position — otherwise retrieval/index quality depends on incidental
    batch composition, a real failure mode for some padding-sensitive or quantized
    runtimes. This isn't about embedding *quality*, only about *determinism*.
    """

    @staticmethod
    def compare(embeddings_a: np.ndarray, embeddings_b: np.ndarray, tolerance: float = 0.999) -> Dict[str, Any]:
        """Pairwise cosine similarity between two equal-length, index-aligned embedding sets."""
        similarities = []
        for a, b in zip(embeddings_a, embeddings_b):
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            similarities.append(sim)

        return {
            "mean_similarity": float(np.mean(similarities)),
            "min_similarity": float(np.min(similarities)),
            "pass_rate": float(np.mean([s >= tolerance for s in similarities])),
            "similarities": similarities,
        }

    @staticmethod
    def aggregate(
        batch_vs_individual: Dict[str, Any],
        order_vs_reordered: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "avg_batch_consistency": batch_vs_individual["mean_similarity"],
            "min_batch_consistency": batch_vs_individual["min_similarity"],
            "batch_consistency_pass_rate": batch_vs_individual["pass_rate"],
            "avg_order_consistency": order_vs_reordered["mean_similarity"],
            "min_order_consistency": order_vs_reordered["min_similarity"],
            "order_consistency_pass_rate": order_vs_reordered["pass_rate"],
            "overall_score": min(
                batch_vs_individual["mean_similarity"], order_vs_reordered["mean_similarity"]
            ),
        }


class LongContextRobustnessEvaluator:
    """Evaluate whether a fact buried near the end of a long document is still
    findable, versus the same fact placed at the very start.

    Many embedding models silently truncate input beyond their configured
    max_sequence_length (or simply down-weight later tokens even without hard
    truncation). A model that handles long context well should embed a document
    such that a query about its content matches similarly well regardless of where
    in the document that content sits; a model that doesn't will show a large gap.
    """

    @staticmethod
    def evaluate_single(
        query_embedding: np.ndarray,
        doc_signal_first_embedding: np.ndarray,
        doc_signal_last_embedding: np.ndarray,
    ) -> Dict[str, Any]:
        def _cosine(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        similarity_signal_first = _cosine(query_embedding, doc_signal_first_embedding)
        similarity_signal_last = _cosine(query_embedding, doc_signal_last_embedding)

        return {
            "similarity_signal_first": similarity_signal_first,
            "similarity_signal_last": similarity_signal_last,
            "position_gap": similarity_signal_first - similarity_signal_last,
        }

    @staticmethod
    def aggregate(results: List[Dict[str, Any]], robust_tolerance: float = 0.15) -> Dict[str, Any]:
        gaps = [r["position_gap"] for r in results]
        return {
            "avg_similarity_signal_first": float(np.mean([r["similarity_signal_first"] for r in results])),
            "avg_similarity_signal_last": float(np.mean([r["similarity_signal_last"] for r in results])),
            "avg_position_gap": float(np.mean(gaps)),
            "max_position_gap": float(np.max(gaps)),
            "robust_rate": float(np.mean([g <= robust_tolerance for g in gaps])),
        }


class RerankingEvaluator:
    """Evaluate reranking of a small, already-retrieved candidate list with graded
    relevance (0/1/2 …), rather than the binary relevant/not-relevant labels used by
    the Retrieval task. This is the MTEB "Reranking" distinction: given a short list a
    first-stage retriever already produced, can the embedding model order it well by
    *degree* of relevance, not just separate relevant from irrelevant.
    """

    @staticmethod
    def evaluate_single(
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        relevance_scores: List[float],
    ) -> Dict[str, Any]:
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_relevance = [relevance_scores[i] for i in ranked_indices]

        # NDCG naturally supports graded (non-binary) relevance — see RetrievalEvaluator.
        ndcg = RetrievalEvaluator.compute_ndcg_at_k(ranked_relevance, k=len(ranked_relevance))

        rank_correlation = 0.0
        if len(set(relevance_scores)) > 1:
            predicted_rank = np.argsort(np.argsort(-similarities))
            rho, _ = spearmanr(relevance_scores, -predicted_rank)
            rank_correlation = float(rho) if not np.isnan(rho) else 0.0

        top1_is_most_relevant = ranked_relevance[0] == max(relevance_scores)

        return {
            "ndcg": ndcg,
            "rank_correlation": rank_correlation,
            "top1_is_most_relevant": bool(top1_is_most_relevant),
        }

    @staticmethod
    def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "avg_ndcg": float(np.mean([r["ndcg"] for r in results])),
            "avg_rank_correlation": float(np.mean([r["rank_correlation"] for r in results])),
            "top1_accuracy": float(np.mean([r["top1_is_most_relevant"] for r in results])),
        }


class PerturbationStabilityEvaluator:
    """Evaluate whether retrieval ranking stays stable when a query is lightly
    perturbed (typo, word reorder, synonym swap) but its meaning is unchanged.

    A brittle embedding model can rank documents very differently for "kredi kartı
    borcum" vs "kredi kartım borcu" even though a human reads them identically —
    this directly measures that kind of instability, distinct from raw retrieval
    quality (which only checks whether *a* correctly-worded query finds the answer).
    """

    @staticmethod
    def evaluate_single(
        original_ranked_indices: np.ndarray,
        perturbed_ranked_indices: np.ndarray,
        positive_indices: set,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        original_top_k = list(original_ranked_indices[:top_k])
        perturbed_top_k = list(perturbed_ranked_indices[:top_k])
        overlap = len(set(original_top_k) & set(perturbed_top_k)) / top_k

        return {
            "top1_stable": bool(original_ranked_indices[0] == perturbed_ranked_indices[0]),
            "top_k_overlap": overlap,
            "original_top1_positive": bool(original_ranked_indices[0] in positive_indices),
            "perturbed_top1_positive": bool(perturbed_ranked_indices[0] in positive_indices),
        }

    @staticmethod
    def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Cases where the original query already failed to find a positive doc aren't
        # informative about *stability* — only count degradation among cases that
        # started from a correct result.
        degraded = [
            r for r in results
            if r["original_top1_positive"] and not r["perturbed_top1_positive"]
        ]
        return {
            "avg_top1_stable": float(np.mean([r["top1_stable"] for r in results])),
            "avg_top_k_overlap": float(np.mean([r["top_k_overlap"] for r in results])),
            "degradation_rate": float(len(degraded) / max(len(results), 1)),
        }


class EmbeddingQualityMetrics:
    """General embedding quality metrics"""
    
    @staticmethod
    def compute_intra_list_similarity(embeddings: np.ndarray) -> float:
        """
        Compute average similarity within a list of embeddings
        Low ILS indicates diverse embeddings (good for retrieval)
        """
        n = len(embeddings)
        if n <= 1:
            return 0.0
        
        similarities = cosine_similarity(embeddings)
        # Exclude diagonal (self-similarity)
        mask = ~np.eye(n, dtype=bool)
        avg_similarity = np.mean(similarities[mask])
        
        return float(avg_similarity)
    
    @staticmethod
    def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute statistical properties of embeddings"""
        norms = np.linalg.norm(embeddings, axis=1)
        
        return {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "dimension": int(embeddings.shape[1]),
            "sparsity": float(np.mean(embeddings == 0)),
            "avg_absolute_value": float(np.mean(np.abs(embeddings)))
        }
    
    @staticmethod
    def compute_discriminative_power(
        embeddings: np.ndarray,
        labels: List[int]
    ) -> float:
        """
        Compute how well embeddings discriminate between different classes
        Uses silhouette-like metric
        """
        unique_labels = list(set(labels))
        if len(unique_labels) <= 1:
            return 0.0
        
        from sklearn.metrics import silhouette_score
        score = silhouette_score(embeddings, labels, metric='cosine')
        
        return float(score)
