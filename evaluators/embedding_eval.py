"""
Embedding Model Evaluators
Metrics for evaluating Turkish embedding models without LLM judges
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.stats import spearmanr, pearsonr
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
