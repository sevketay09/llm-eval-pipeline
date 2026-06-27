"""
Advanced Metrics Module
Throughput, confidence intervals, statistical analysis
Enhanced with: Effect Size, Bootstrap CI, Multiple Comparison Correction,
Bayesian Testing, Distribution Shift Detection, Trend Analysis
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats
import warnings
import requests
import os


class ThroughputMetrics:
    """Calculate throughput metrics"""
    
    @staticmethod
    def calculate(
        latencies: List[float],
        input_tokens: List[int],
        output_tokens: List[int]
    ) -> Dict[str, float]:
        """
        Calculate throughput metrics
        
        Returns:
            {
                "tokens_per_second": float,
                "requests_per_minute": float,
                "avg_tokens_per_request": float
            }
        """
        if not latencies:
            return {
                "tokens_per_second": 0.0,
                "requests_per_minute": 0.0,
                "avg_tokens_per_request": 0.0
            }
        
        total_tokens = sum(input_tokens) + sum(output_tokens)
        total_time = sum(latencies)
        avg_latency = sum(latencies) / len(latencies)
        
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        requests_per_minute = 60 / avg_latency if avg_latency > 0 else 0
        avg_tokens_per_request = total_tokens / len(latencies) if latencies else 0
        
        return {
            "tokens_per_second": tokens_per_second,
            "requests_per_minute": requests_per_minute,
            "avg_tokens_per_request": avg_tokens_per_request
        }


class StatisticalMetrics:
    """Statistical analysis metrics"""
    
    @staticmethod
    def cohens_d(
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate Cohen's d effect size
        
        Effect size interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
        
        Returns:
            {
                "cohens_d": float,
                "effect_size": str,
                "interpretation": str
            }
        """
        if len(scores_a) < 2 or len(scores_b) < 2:
            return {
                "cohens_d": 0.0,
                "effect_size": "negligible",
                "interpretation": "Insufficient data"
            }
        
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        var_a = np.var(scores_a, ddof=1)
        var_b = np.var(scores_b, ddof=1)
        
        # Pooled standard deviation
        n_a = len(scores_a)
        n_b = len(scores_b)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            return {
                "cohens_d": 0.0,
                "effect_size": "negligible",
                "interpretation": "No variance in data"
            }
        
        d = (mean_a - mean_b) / pooled_std
        abs_d = abs(d)
        
        if abs_d < 0.2:
            effect_size = "negligible"
            interpretation = "Pratik olarak anlamli fark yok"
        elif abs_d < 0.5:
            effect_size = "small"
            interpretation = "Kucuk etki boyutu"
        elif abs_d < 0.8:
            effect_size = "medium"
            interpretation = "Orta etki boyutu"
        else:
            effect_size = "large"
            interpretation = "Buyuk etki boyutu"
        
        return {
            "cohens_d": float(d),
            "effect_size": effect_size,
            "interpretation": interpretation,
            "mean_difference": float(mean_a - mean_b)
        }
    
    @staticmethod
    def bootstrap_confidence_interval(
        scores: List[float],
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence interval (non-parametric)
        
        Returns:
            {
                "mean": float,
                "ci_lower": float,
                "ci_upper": float,
                "bootstrap_std": float
            }
        """
        if len(scores) < 2:
            mean_val = scores[0] if scores else 0
            return {
                "mean": float(mean_val),
                "ci_lower": float(mean_val),
                "ci_upper": float(mean_val),
                "bootstrap_std": 0.0
            }
        
        np.random.seed(random_seed)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return {
            "mean": float(np.mean(scores)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "bootstrap_std": float(np.std(bootstrap_means)),
            "confidence_level": confidence
        }
    
    @staticmethod
    def calculate_confidence_interval(
        scores: List[float],
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence interval for scores
        
        Returns:
            {
                "mean": float,
                "ci_lower": float,
                "ci_upper": float,
                "std": float
            }
        """
        if len(scores) < 2:
            return {
                "mean": scores[0] if scores else 0,
                "ci_lower": 0,
                "ci_upper": 0,
                "std": 0
            }
        
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        sem = stats.sem(scores)
        
        # Handle zero variance case (all scores are identical)
        if std == 0 or sem == 0:
            return {
                "mean": float(mean),
                "ci_lower": float(mean),
                "ci_upper": float(mean),
                "std": 0.0,
                "confidence_level": confidence
            }
        
        ci = stats.t.interval(
            confidence,
            len(scores) - 1,
            loc=mean,
            scale=sem
        )
        
        # Handle potential NaN values
        ci_lower = float(ci[0]) if not np.isnan(ci[0]) else float(mean)
        ci_upper = float(ci[1]) if not np.isnan(ci[1]) else float(mean)
        
        return {
            "mean": float(mean),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std": float(std),
            "confidence_level": confidence
        }
    
    @staticmethod
    def t_test(
        scores_a: List[float],
        scores_b: List[float],
        include_effect_size: bool = True
    ) -> Dict[str, Any]:
        """
        Perform t-test to compare two models with effect size
        
        Returns:
            {
                "t_statistic": float,
                "p_value": float,
                "is_significant": bool (p < 0.05),
                "better_model": str ("A", "B", or "No difference"),
                "effect_size": dict (Cohen's d)
            }
        """
        # Ensure scores are numeric floats
        _LABEL_MAP = {"TAM_DOGRU": 1.0, "KISMEN_DOGRU": 0.5, "YANLIS": 0.0}
        def _to_float(v):
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                if v in _LABEL_MAP:
                    return _LABEL_MAP[v]
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
            return None

        scores_a = [x for x in (_to_float(v) for v in scores_a) if x is not None]
        scores_b = [x for x in (_to_float(v) for v in scores_b) if x is not None]

        if len(scores_a) < 2 or len(scores_b) < 2:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "better_model": "No difference",
                "mean_a": 0.0,
                "mean_b": 0.0
            }
        
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        var_a = np.var(scores_a)
        var_b = np.var(scores_b)
        
        # If both groups have zero variance (all same values), no meaningful comparison
        if var_a == 0 and var_b == 0:
            result = {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "better_model": "No difference",
                "mean_a": float(mean_a),
                "mean_b": float(mean_b)
            }
            if include_effect_size:
                result["effect_size"] = StatisticalMetrics.cohens_d(scores_a, scores_b)
            return result
        
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
        
        # Handle NaN values from scipy (can occur with edge cases)
        if np.isnan(t_stat) or np.isnan(p_value):
            result = {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "better_model": "No difference",
                "mean_a": float(mean_a),
                "mean_b": float(mean_b)
            }
            if include_effect_size:
                result["effect_size"] = StatisticalMetrics.cohens_d(scores_a, scores_b)
            return result
        
        is_significant = p_value < 0.05
        
        if is_significant:
            better_model = "A" if mean_a > mean_b else "B"
        else:
            better_model = "No difference"
        
        result = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "better_model": better_model,
            "mean_a": float(mean_a),
            "mean_b": float(mean_b)
        }
        
        if include_effect_size:
            result["effect_size"] = StatisticalMetrics.cohens_d(scores_a, scores_b)
        
        return result
    
    @staticmethod
    def mann_whitney_u_test(
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test)
        
        Returns:
            {
                "u_statistic": float,
                "p_value": float,
                "is_significant": bool,
                "better_model": str
            }
        """
        if len(scores_a) < 2 or len(scores_b) < 2:
            return {
                "u_statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "better_model": "No difference",
                "median_a": 0.0,
                "median_b": 0.0
            }
        
        median_a = np.median(scores_a)
        median_b = np.median(scores_b)
        
        # If all values are identical in both groups, no meaningful comparison
        if len(set(scores_a)) == 1 and len(set(scores_b)) == 1 and scores_a[0] == scores_b[0]:
            return {
                "u_statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "better_model": "No difference",
                "median_a": float(median_a),
                "median_b": float(median_b)
            }
        
        u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
        
        # Handle NaN values from scipy
        if np.isnan(u_stat) or np.isnan(p_value):
            return {
                "u_statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "better_model": "No difference",
                "median_a": float(median_a),
                "median_b": float(median_b)
            }
        
        is_significant = p_value < 0.05
        
        if is_significant:
            better_model = "A" if median_a > median_b else "B"
        else:
            better_model = "No difference"
        
        return {
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "better_model": better_model,
            "median_a": float(median_a),
            "median_b": float(median_b)
        }


class CategoryMetrics:
    """Per-category performance analysis"""
    
    @staticmethod
    def calculate_per_category(
        results: List[Dict[str, Any]],
        category_field: str = "category"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate categorical label distribution per category.

        Returns:
            {
                "category_name": {
                    "count": int,
                    "TAM_DOGRU": int,
                    "KISMEN_DOGRU": int,
                    "YANLIS": int,
                    "tam_dogru_rate": float,
                }
            }
        """
        categories: Dict[str, List[str]] = {}

        for result in results:
            category = result.get(category_field, "unknown")
            label = result.get("scores", {}).get("judge_label", "YANLIS")
            if not isinstance(label, str):
                label = "YANLIS"
            if category not in categories:
                categories[category] = []
            categories[category].append(label)

        category_stats: Dict[str, Any] = {}
        for category, labels in categories.items():
            n = len(labels)
            tam = labels.count("TAM_DOGRU")
            kismen = labels.count("KISMEN_DOGRU")
            yanlis = labels.count("YANLIS")
            category_stats[category] = {
                "count": n,
                "TAM_DOGRU": tam,
                "KISMEN_DOGRU": kismen,
                "YANLIS": yanlis,
                "tam_dogru_rate": round(tam / n, 3) if n > 0 else 0.0,
            }

        return category_stats


class AdvancedStatisticalMetrics:
    """Advanced statistical tests and comparisons"""
    
    @staticmethod
    def multiple_comparison_correction(
        p_values: List[float],
        method: str = "fdr_bh",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Apply multiple comparison correction to p-values
        
        Methods:
        - bonferroni: Conservative, controls family-wise error rate
        - fdr_bh: Benjamini-Hochberg, controls false discovery rate (recommended)
        - holm: Step-down method
        
        Returns:
            {
                "reject": List[bool],  # Which hypotheses to reject
                "p_corrected": List[float],  # Corrected p-values
                "alpha_corrected": float  # Adjusted significance level
            }
        """
        if not p_values:
            return {
                "reject": [],
                "p_corrected": [],
                "alpha_corrected": alpha
            }
        
        n = len(p_values)
        
        if method == "bonferroni":
            alpha_corrected = alpha / n
            p_corrected = [min(p * n, 1.0) for p in p_values]
            reject = [p < alpha_corrected for p in p_values]
        
        elif method == "fdr_bh":
            # Benjamini-Hochberg procedure
            sorted_indices = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_indices]
            
            reject_temp = [False] * n
            p_corrected_temp = [0.0] * n
            
            for i in range(n - 1, -1, -1):
                p_corrected_temp[i] = min(sorted_p[i] * n / (i + 1), 1.0)
                if i < n - 1:
                    p_corrected_temp[i] = min(p_corrected_temp[i], p_corrected_temp[i + 1])
                reject_temp[i] = p_corrected_temp[i] < alpha
            
            # Reorder to original order
            p_corrected = [0.0] * n
            reject = [False] * n
            for i, idx in enumerate(sorted_indices):
                p_corrected[idx] = p_corrected_temp[i]
                reject[idx] = reject_temp[i]
            
            alpha_corrected = alpha
        
        elif method == "holm":
            # Holm-Bonferroni step-down
            sorted_indices = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_indices]
            
            reject_temp = [False] * n
            p_corrected_temp = [0.0] * n
            
            for i in range(n):
                p_corrected_temp[i] = min(sorted_p[i] * (n - i), 1.0)
                if i > 0:
                    p_corrected_temp[i] = max(p_corrected_temp[i], p_corrected_temp[i - 1])
                reject_temp[i] = p_corrected_temp[i] < alpha
            
            # Reorder
            p_corrected = [0.0] * n
            reject = [False] * n
            for i, idx in enumerate(sorted_indices):
                p_corrected[idx] = p_corrected_temp[i]
                reject[idx] = reject_temp[i]
            
            alpha_corrected = alpha
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            "reject": reject,
            "p_corrected": p_corrected,
            "alpha_corrected": alpha_corrected,
            "method": method
        }
    
    @staticmethod
    def bayesian_ab_test(
        scores_a: List[float],
        scores_b: List[float],
        prior_mean: float = 0.5,
        prior_std: float = 0.25,
        n_samples: int = 10000,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Bayesian A/B test using Monte Carlo sampling
        
        Returns:
            {
                "prob_a_better": float,  # P(A > B)
                "prob_b_better": float,  # P(B > A)
                "expected_difference": float,  # E[A - B]
                "credible_interval": tuple,  # 95% CI of difference
            }
        """
        if len(scores_a) < 2 or len(scores_b) < 2:
            return {
                "prob_a_better": 0.5,
                "prob_b_better": 0.5,
                "expected_difference": 0.0,
                "credible_interval": (0.0, 0.0)
            }
        
        np.random.seed(random_seed)
        
        # Posterior sampling for A
        mean_a = np.mean(scores_a)
        std_a = np.std(scores_a, ddof=1) if len(scores_a) > 1 else 0.01
        n_a = len(scores_a)
        
        # Posterior sampling for B
        mean_b = np.mean(scores_b)
        std_b = np.std(scores_b, ddof=1) if len(scores_b) > 1 else 0.01
        n_b = len(scores_b)
        
        # Sample from posterior distributions (assuming normal likelihood)
        samples_a = np.random.normal(mean_a, std_a / np.sqrt(n_a), n_samples)
        samples_b = np.random.normal(mean_b, std_b / np.sqrt(n_b), n_samples)
        
        differences = samples_a - samples_b
        
        prob_a_better = float(np.mean(differences > 0))
        prob_b_better = float(np.mean(differences < 0))
        expected_diff = float(np.mean(differences))
        
        # 95% credible interval
        ci_lower = float(np.percentile(differences, 2.5))
        ci_upper = float(np.percentile(differences, 97.5))
        
        return {
            "prob_a_better": prob_a_better,
            "prob_b_better": prob_b_better,
            "expected_difference": expected_diff,
            "credible_interval": (ci_lower, ci_upper),
            "interpretation": f"Model A is better with {prob_a_better*100:.1f}% probability"
        }
    
    @staticmethod
    def distribution_shift_test(
        baseline_scores: List[float],
        current_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Detect distribution shift using Kolmogorov-Smirnov test
        
        Returns:
            {
                "shift_detected": bool,
                "ks_statistic": float,
                "p_value": float
            }
        """
        if len(baseline_scores) < 2 or len(current_scores) < 2:
            return {
                "shift_detected": False,
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "interpretation": "Insufficient data"
            }
        
        ks_stat, p_value = stats.ks_2samp(baseline_scores, current_scores)
        
        shift_detected = p_value < 0.05
        
        if shift_detected:
            interpretation = "UYARI: Skor dagiliminda anlamli kayma tespit edildi"
        else:
            interpretation = "Skor dagilimi stabil"
        
        return {
            "shift_detected": shift_detected,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "interpretation": interpretation
        }


class TrendAnalysisMetrics:
    """Time series and trend analysis"""
    
    @staticmethod
    def linear_trend_test(
        scores_over_time: List[float],
        timestamps: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Test for linear trend in scores over time
        
        Returns:
            {
                "trend": str ("improving", "declining", "stable"),
                "slope": float,
                "p_value": float,
                "is_significant": bool,
                "r_squared": float
            }
        """
        if len(scores_over_time) < 3:
            return {
                "trend": "stable",
                "slope": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "r_squared": 0.0,
                "interpretation": "Insufficient data for trend analysis"
            }
        
        if timestamps is None:
            x = np.arange(len(scores_over_time))
        else:
            x = np.array(timestamps)
        
        y = np.array(scores_over_time)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        is_significant = p_value < 0.05
        
        if is_significant:
            if slope > 0:
                trend = "improving"
                interpretation = f"Performans artis trendinde (slope={slope:.4f})"
            else:
                trend = "declining"
                interpretation = f"UYARI: Performans dusus trendinde (slope={slope:.4f})"
        else:
            trend = "stable"
            interpretation = "Performans stabil"
        
        return {
            "trend": trend,
            "slope": float(slope),
            "intercept": float(intercept),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "r_squared": float(r_value ** 2),
            "interpretation": interpretation
        }
    
    @staticmethod
    def stratified_performance(
        results: List[Dict[str, Any]],
        strata_key: str = "difficulty"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance across different strata (e.g., difficulty levels)
        
        Returns:
            {
                "easy": {"mean": ..., "ci": ..., "count": ...},
                "medium": {...},
                "hard": {...}
            }
        """
        strata_groups = {}
        
        for result in results:
            stratum = result.get(strata_key, "unknown")
            score = result.get("scores", {})
            
            if isinstance(score, dict):
                score_value = list(score.values())[0] if score else 0
            else:
                score_value = score
            
            if stratum not in strata_groups:
                strata_groups[stratum] = []
            
            strata_groups[stratum].append(float(score_value))
        
        performance_by_strata = {}
        
        for stratum, scores in strata_groups.items():
            if scores:
                mean_score = np.mean(scores)
                
                # Bootstrap CI
                if len(scores) >= 2:
                    bootstrap_ci = StatisticalMetrics.bootstrap_confidence_interval(scores)
                    ci = (bootstrap_ci["ci_lower"], bootstrap_ci["ci_upper"])
                else:
                    ci = (mean_score, mean_score)
                
                performance_by_strata[stratum] = {
                    "mean": float(mean_score),
                    "ci_lower": float(ci[0]),
                    "ci_upper": float(ci[1]),
                    "count": len(scores),
                    "std": float(np.std(scores)) if len(scores) > 1 else 0.0
                }
        
        return performance_by_strata


class SemanticSimilarityMetrics:
    """Semantic similarity calculation using embeddings"""
    
    # Default embedding service URL (can be overridden via environment variable)
    EMBEDDING_SERVICE_URL = os.getenv(
        "EMBEDDING_SERVICE_URL",
        os.getenv("EMBEDDING_SERVICE_BASE_URL", "") + "/v1/embeddings"
    )
    
    @staticmethod
    def get_embeddings(
        texts: List[str],
        service_url: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Get embeddings from the embedding service
        
        Returns:
            np.ndarray of shape (n_texts, embedding_dim) or None if failed
        """
        if not texts:
            return None
        
        url = service_url or SemanticSimilarityMetrics.EMBEDDING_SERVICE_URL
        
        try:
            payload = {
                "input": texts,
                "model": "multi-qa-mpnet-base-dot-v1"
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract embeddings from response
            if "data" in data:
                embeddings = [item["embedding"] for item in data["data"]]
                return np.array(embeddings)
            else:
                warnings.warn("Unexpected response format from embedding service")
                return None
                
        except Exception as e:
            warnings.warn(f"Failed to get embeddings: {str(e)}")
            return None
    
    @staticmethod
    def cosine_similarity(
        vec_a: np.ndarray,
        vec_b: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    @staticmethod
    def semantic_variance(
        responses: List[str],
        service_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate semantic variance of responses
        Low variance = high consistency (semantically similar)
        
        Returns:
            {
                "semantic_variance": float,
                "avg_similarity": float,
                "min_similarity": float,
                "max_similarity": float,
                "is_consistent": bool
            }
        """
        if len(responses) < 2:
            return {
                "semantic_variance": 0.0,
                "avg_similarity": 1.0,
                "min_similarity": 1.0,
                "max_similarity": 1.0,
                "is_consistent": True,
                "error": "Insufficient responses"
            }
        
        embeddings = SemanticSimilarityMetrics.get_embeddings(responses, service_url)
        
        if embeddings is None:
            # Fallback to text similarity
            from difflib import SequenceMatcher
            similarities = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    sim = SequenceMatcher(None, responses[i], responses[j]).ratio()
                    similarities.append(sim)
            
            if similarities:
                avg_sim = float(np.mean(similarities))
                variance = float(1 - avg_sim)
                
                return {
                    "semantic_variance": variance,
                    "avg_similarity": avg_sim,
                    "min_similarity": float(np.min(similarities)),
                    "max_similarity": float(np.max(similarities)),
                    "is_consistent": variance < 0.3,
                    "method": "text_similarity_fallback"
                }
        
        # Calculate pairwise cosine similarities
        similarities = []
        n = len(embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = SemanticSimilarityMetrics.cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        if not similarities:
            return {
                "semantic_variance": 0.0,
                "avg_similarity": 1.0,
                "min_similarity": 1.0,
                "max_similarity": 1.0,
                "is_consistent": True
            }
        
        avg_similarity = float(np.mean(similarities))
        variance = float(1 - avg_similarity)  # Convert similarity to variance
        
        # Consistent if average similarity > 0.7 (variance < 0.3)
        is_consistent = variance < 0.3
        
        return {
            "semantic_variance": variance,
            "avg_similarity": avg_similarity,
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "is_consistent": is_consistent,
            "num_comparisons": len(similarities),
            "method": "embedding_cosine_similarity"
        }
    
    @staticmethod
    def semantic_consistency_score(
        responses: List[str],
        threshold: float = 0.7,
        service_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced semantic consistency with detailed analysis
        
        Returns:
            {
                "consistency_score": float (0-1),
                "semantic_variance": float,
                "interpretation": str
            }
        """
        variance_result = SemanticSimilarityMetrics.semantic_variance(responses, service_url)
        
        # Consistency score is inverse of variance
        consistency_score = 1 - variance_result["semantic_variance"]
        
        if consistency_score >= 0.9:
            interpretation = "Cok yuksek tutarlilik - neredeyse identik anlamlar"
        elif consistency_score >= 0.7:
            interpretation = "Yuksek tutarlilik - anlamsal olarak uyumlu"
        elif consistency_score >= 0.5:
            interpretation = "Orta tutarlilik - bazi farkliliklar var"
        elif consistency_score >= 0.3:
            interpretation = "Dusuk tutarlilik - onemli anlamsal farklar"
        else:
            interpretation = "UYARI: Cok dusuk tutarlilik - celistkili yanıtlar"
        
        return {
            "consistency_score": float(consistency_score),
            "semantic_variance": variance_result["semantic_variance"],
            "avg_similarity": variance_result["avg_similarity"],
            "is_consistent": variance_result["is_consistent"],
            "interpretation": interpretation,
            "method": variance_result.get("method", "unknown")
        }
