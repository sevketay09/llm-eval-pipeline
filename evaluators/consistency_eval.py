"""
Consistency Evaluator - Enhanced Version
Aynı soruya farklı formülasyonlarla tutarlı yanıt verip vermediğini test eder
ENHANCED: Self-consistency with variance metrics, temperature testing, semantic similarity
"""
import json
import statistics
from typing import Dict, Any, List, Tuple, Optional
from adapters.unified_adapter import UnifiedLLMAdapter
from difflib import SequenceMatcher
from metrics import SemanticSimilarityMetrics


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using SequenceMatcher"""
    return SequenceMatcher(None, text1, text2).ratio()


class ConsistencyEvaluator:
    """Test model consistency across multiple runs"""
    
    CONSISTENCY_PROMPT = """
Aşağıda aynı sorunun farklı versiyonlarına verilen yanıtları değerlendirin.

Orijinal Soru: {original_question}

Yanıtlar:
{responses}

Bu yanıtların birbirleriyle tutarlılığını değerlendirin:
- 1-3: Çelişkili, farklı cevaplar
- 4-6: Kısmen tutarlı, bazı farklılıklar var
- 7-8: Çoğunlukla tutarlı, küçük farklılıklar
- 9-10: Tamamen tutarlı, aynı bilgileri veriyor

JSON formatında yanıt verin:
{{"score": <1-10>, "reasoning": "<açıklama>", "inconsistencies": ["<tespit edilen tutarsızlıklar>"]}}
"""

    def __init__(self, judge_adapter: UnifiedLLMAdapter):
        self.judge = judge_adapter
    
    def test_consistency(
        self,
        model: UnifiedLLMAdapter,
        question: str,
        num_runs: int = 3,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Test model consistency by asking same question multiple times
        
        Args:
            model: Model to test
            question: Question to ask
            num_runs: Number of times to ask (default 3)
            temperature: Sampling temperature
        
        Returns:
            {
                "score": float (0-1),
                "responses": List[str],
                "variance": float,
                "is_consistent": bool
            }
        """
        responses = []

        # Generate multiple responses
        messages = [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": question}
        ]

        for _ in range(num_runs):
            result = model.generate(messages, temperature=temperature)
            if result.get("error"):
                # Drop the failed run rather than counting an infrastructure
                # error as one more (identical-looking "None") response —
                # that would corrupt the consistency judge/variance below.
                continue
            responses.append(result['content'])

        if not responses:
            return {"generation_error": "all consistency runs failed"}

        # Evaluate consistency with judge
        responses_text = "\n\n".join([f"Yanıt {i+1}: {r}" for i, r in enumerate(responses)])
        
        prompt = self.CONSISTENCY_PROMPT.format(
            original_question=question,
            responses=responses_text
        )
        
        judge_messages = [
            {"role": "system", "content": "Sen bir tutarlılık analisti olarak çalışırsın."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.judge.generate(judge_messages, temperature=0.0, max_tokens=500)
        
        try:
            content = result['content'].strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(content)
            score = evaluation.get("score", 0) / 10.0
            
            # Calculate response length variance as additional metric
            lengths = [len(r) for r in responses]
            avg_length = sum(lengths) / len(lengths)
            variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            
            return {
                "score": score,
                "reasoning": evaluation.get("reasoning", ""),
                "inconsistencies": evaluation.get("inconsistencies", []),
                "responses": responses,
                "variance": variance,
                "is_consistent": score >= 0.7,
                "raw_response": result['content']
            }
        except Exception as e:
            return {
                "score": 0.5,
                "reasoning": f"Parse error: {str(e)}",
                "inconsistencies": [],
                "responses": responses,
                "variance": 0,
                "is_consistent": False,
                "raw_response": result['content']
            }
    
    def test_paraphrase_consistency(
        self,
        model: UnifiedLLMAdapter,
        question_variations: List[str]
    ) -> Dict[str, Any]:
        """
        Test consistency across paraphrased questions
        
        Args:
            model: Model to test
            question_variations: List of paraphrased versions of same question
        
        Returns:
            Similar to test_consistency
        """
        responses = []
        
        for question in question_variations:
            messages = [
                {"role": "system", "content": "Sen yardımcı bir asistansın."},
                {"role": "user", "content": question}
            ]
            result = model.generate(messages, temperature=0.0)
            responses.append(result['content'])
        
        # Evaluate with judge
        responses_text = "\n\n".join([
            f"Soru {i+1}: {question_variations[i]}\nYanıt {i+1}: {responses[i]}"
            for i in range(len(responses))
        ])
        
        prompt = self.CONSISTENCY_PROMPT.format(
            original_question=question_variations[0],
            responses=responses_text
        )
        
        judge_messages = [
            {"role": "system", "content": "Sen bir tutarlılık analisti olarak çalışırsın."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.judge.generate(judge_messages, temperature=0.0, max_tokens=500)
        
        try:
            content = result['content'].strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(content)
            score = evaluation.get("score", 0) / 10.0
            
            return {
                "score": score,
                "reasoning": evaluation.get("reasoning", ""),
                "inconsistencies": evaluation.get("inconsistencies", []),
                "question_variations": question_variations,
                "responses": responses,
                "is_consistent": score >= 0.7,
                "raw_response": result['content']
            }
        except Exception as e:
            return {
                "score": 0.5,
                "reasoning": f"Parse error: {str(e)}",
                "inconsistencies": [],
                "question_variations": question_variations,
                "responses": responses,
                "is_consistent": False,
                "raw_response": result['content']
            }


class SelfConsistencyEvaluator:
    """
    Enhanced Self-Consistency Evaluator
    
    Tests model consistency by:
    - Running same question multiple times (3-5 runs)
    - Testing with different temperatures (0.0, 0.3, 0.7)
    - Measuring response variance (length, word count, semantic similarity)
    - Detecting answer changes and instability
    - Calculating confidence scores
    """
    
    def __init__(self, judge_adapter: UnifiedLLMAdapter = None):
        self.judge = judge_adapter
    
    def evaluate_self_consistency(
        self,
        model: UnifiedLLMAdapter,
        question: str,
        num_runs: int = 5,
        temperatures: List[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive self-consistency evaluation.
        
        Args:
            model: Model to test
            question: Question to ask
            num_runs: Number of runs per temperature (default: 5)
            temperatures: List of temperatures to test (default: [0.0, 0.3, 0.7])
        
        Returns:
            Detailed consistency metrics
        """
        if temperatures is None:
            temperatures = [0.0, 0.3, 0.7]
        
        all_results = {}
        
        for temp in temperatures:
            results = self._run_multiple_times(model, question, num_runs, temp)
            all_results[f"temp_{temp}"] = results
        
        # Calculate comprehensive metrics
        metrics = self._calculate_consistency_metrics(all_results, question)
        
        return metrics
    
    def _run_multiple_times(
        self,
        model: UnifiedLLMAdapter,
        question: str,
        num_runs: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Run same question multiple times with given temperature"""
        responses = []
        latencies = []
        
        messages = [
            {"role": "system", "content": "Sen yardımcı bir asistansın. Soruları doğru ve tutarlı şekilde cevapla."},
            {"role": "user", "content": question}
        ]
        
        for _ in range(num_runs):
            result = model.generate(messages, temperature=temperature)
            responses.append(result['content'] or "")
            latencies.append(result['latency'])
        
        return {
            "responses": responses,
            "latencies": latencies,
            "temperature": temperature
        }
    
    def _calculate_consistency_metrics(
        self,
        all_results: Dict[str, Dict],
        question: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive consistency metrics"""
        
        # Aggregate all responses
        all_responses = []
        for temp_results in all_results.values():
            all_responses.extend(temp_results["responses"])
        
        # Calculate metrics
        metrics = {
            "question": question,
            "total_runs": len(all_responses),
            "temperatures_tested": list(all_results.keys()),
            "by_temperature": {}
        }
        
        # Per-temperature analysis
        for temp_key, temp_results in all_results.items():
            responses = temp_results["responses"]
            temp_metrics = self._analyze_responses(responses, temp_results["temperature"])
            metrics["by_temperature"][temp_key] = temp_metrics
        
        # Overall consistency across all temperatures
        metrics["overall"] = self._calculate_overall_consistency(all_results)
        
        # Final score (0-1 scale)
        metrics["consistency_score"] = self._calculate_final_score(metrics)
        
        return metrics
    
    def _analyze_responses(
        self,
        responses: List[str],
        temperature: float
    ) -> Dict[str, Any]:
        """Analyze responses for a specific temperature with semantic similarity"""
        
        if not responses or all(r == "" for r in responses):
            return {
                "temperature": temperature,
                "num_responses": len(responses),
                "error": "No valid responses"
            }
        
        # Filter empty responses
        valid_responses = [r for r in responses if r]
        
        if not valid_responses:
            return {
                "temperature": temperature,
                "num_responses": 0,
                "error": "All responses empty"
            }
        
        # Length variance
        lengths = [len(r) for r in valid_responses]
        length_mean = statistics.mean(lengths)
        length_variance = statistics.variance(lengths) if len(lengths) > 1 else 0
        length_std = statistics.stdev(lengths) if len(lengths) > 1 else 0
        
        # Word count variance
        word_counts = [len(r.split()) for r in valid_responses]
        word_mean = statistics.mean(word_counts)
        word_variance = statistics.variance(word_counts) if len(word_counts) > 1 else 0
        word_std = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        
        # Semantic similarity (using embeddings)
        semantic_metrics = SemanticSimilarityMetrics.semantic_variance(valid_responses)
        
        # Fallback: Pairwise text similarity (if embedding service fails)
        text_similarities = []
        for i in range(len(valid_responses)):
            for j in range(i + 1, len(valid_responses)):
                sim = calculate_text_similarity(valid_responses[i], valid_responses[j])
                text_similarities.append(sim)
        
        avg_text_similarity = statistics.mean(text_similarities) if text_similarities else 0.0
        
        # Detect unique responses (using similarity threshold)
        unique_responses = self._count_unique_responses(valid_responses, threshold=0.8)
        
        # Consistency score for this temperature
        # Use semantic similarity if available, otherwise text similarity
        primary_similarity = semantic_metrics.get("avg_similarity", avg_text_similarity)
        
        # Higher similarity = higher consistency
        # Lower variance = higher consistency
        consistency_score = primary_similarity * (1 - min(length_std / (length_mean + 1), 1.0))
        
        return {
            "temperature": temperature,
            "num_responses": len(valid_responses),
            "length_stats": {
                "mean": length_mean,
                "variance": length_variance,
                "std": length_std,
                "min": min(lengths),
                "max": max(lengths),
                "coefficient_of_variation": length_std / length_mean if length_mean > 0 else 0
            },
            "word_count_stats": {
                "mean": word_mean,
                "variance": word_variance,
                "std": word_std,
                "min": min(word_counts),
                "max": max(word_counts),
                "coefficient_of_variation": word_std / word_mean if word_mean > 0 else 0
            },
            "semantic_similarity": semantic_metrics,
            "text_similarity_stats": {
                "mean_pairwise_similarity": avg_text_similarity,
                "min_similarity": min(text_similarities) if text_similarities else 0,
                "max_similarity": max(text_similarities) if text_similarities else 0
            },
            "unique_responses_count": unique_responses,
            "consistency_score": consistency_score,
            "is_consistent": semantic_metrics.get("is_consistent", consistency_score >= 0.7),
            "sample_responses": valid_responses[:3]  # First 3 for inspection
        }
    
    def _count_unique_responses(
        self,
        responses: List[str],
        threshold: float = 0.8
    ) -> int:
        """Count unique responses using similarity threshold"""
        if not responses:
            return 0
        
        unique = [responses[0]]
        
        for response in responses[1:]:
            is_unique = True
            for unique_resp in unique:
                if calculate_text_similarity(response, unique_resp) >= threshold:
                    is_unique = False
                    break
            if is_unique:
                unique.append(response)
        
        return len(unique)
    
    def _calculate_overall_consistency(
        self,
        all_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Calculate consistency across all temperatures"""
        
        # Collect all responses
        all_responses = []
        for temp_results in all_results.values():
            all_responses.extend([r for r in temp_results["responses"] if r])
        
        if not all_responses:
            return {"error": "No valid responses"}
        
        # Overall similarity
        all_similarities = []
        for i in range(len(all_responses)):
            for j in range(i + 1, len(all_responses)):
                sim = calculate_text_similarity(all_responses[i], all_responses[j])
                all_similarities.append(sim)
        
        avg_similarity = statistics.mean(all_similarities) if all_similarities else 0.0
        
        # Temperature stability: Do responses change significantly with temperature?
        temp_stability_score = self._calculate_temperature_stability(all_results)
        
        return {
            "overall_similarity": avg_similarity,
            "temperature_stability": temp_stability_score,
            "total_unique_responses": self._count_unique_responses(all_responses, threshold=0.8),
            "is_stable_across_temps": temp_stability_score >= 0.7
        }
    
    def _calculate_temperature_stability(
        self,
        all_results: Dict[str, Dict]
    ) -> float:
        """Calculate how stable responses are across different temperatures"""
        
        temperature_groups = []
        for temp_results in all_results.values():
            valid_responses = [r for r in temp_results["responses"] if r]
            if valid_responses:
                # Representative response (first one) from this temperature
                temperature_groups.append(valid_responses[0])
        
        if len(temperature_groups) < 2:
            return 1.0  # Only one temperature, perfectly stable
        
        # Calculate similarity between temperature groups
        similarities = []
        for i in range(len(temperature_groups)):
            for j in range(i + 1, len(temperature_groups)):
                sim = calculate_text_similarity(temperature_groups[i], temperature_groups[j])
                similarities.append(sim)
        
        return statistics.mean(similarities) if similarities else 0.0
    
    def _calculate_final_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate final consistency score (0-1)"""
        
        if "overall" not in metrics or "error" in metrics["overall"]:
            return 0.0
        
        # Weight different aspects
        overall_similarity = metrics["overall"].get("overall_similarity", 0.0)
        temp_stability = metrics["overall"].get("temperature_stability", 0.0)
        
        # Average consistency across temperatures
        temp_scores = []
        for temp_metrics in metrics["by_temperature"].values():
            if "consistency_score" in temp_metrics:
                temp_scores.append(temp_metrics["consistency_score"])
        
        avg_temp_score = statistics.mean(temp_scores) if temp_scores else 0.0
        
        # Weighted combination
        final_score = (
            0.4 * overall_similarity +
            0.3 * temp_stability +
            0.3 * avg_temp_score
        )
        
        return min(1.0, max(0.0, final_score))
    
    def evaluate_with_judge(
        self,
        model: UnifiedLLMAdapter,
        question: str,
        num_runs: int = 3,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluate self-consistency using LLM-as-Judge.
        Simpler version for when judge is available.
        """
        if not self.judge:
            raise ValueError("Judge adapter not provided")
        
        # Run multiple times
        result_data = self._run_multiple_times(model, question, num_runs, temperature)
        responses = result_data["responses"]
        
        # Ask judge to evaluate consistency
        responses_text = "\n\n".join([f"Yanıt {i+1}: {r}" for i, r in enumerate(responses)])
        
        judge_prompt = f"""
Aşağıda aynı soruya {num_runs} kez verilen yanıtları görüyorsunuz.

Soru: {question}

Yanıtlar:
{responses_text}

Bu yanıtların tutarlılığını değerlendirin:
- Tüm yanıtlar aynı bilgiyi mi veriyor?
- Çelişkiler var mı?
- Cevaplar birbirinden çok farklı mı?

JSON formatında değerlendirin:
{{"score": <0-10>, "reasoning": "<açıklama>", "is_consistent": <true/false>, "main_differences": ["<farklılıklar>"]}}
"""
        
        judge_result = self.judge.generate([
            {"role": "system", "content": "Sen bir tutarlılık değerlendirme uzmanısın."},
            {"role": "user", "content": judge_prompt}
        ], temperature=0.0)
        
        try:
            content = judge_result['content'].strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(content)
            
            return {
                "score": evaluation.get("score", 0) / 10.0,
                "is_consistent": evaluation.get("is_consistent", False),
                "reasoning": evaluation.get("reasoning", ""),
                "main_differences": evaluation.get("main_differences", []),
                "responses": responses,
                "num_runs": num_runs,
                "temperature": temperature
            }
        except Exception as e:
            # Fallback to metric-based evaluation
            metrics = self._analyze_responses(responses, temperature)
            return {
                "score": metrics.get("consistency_score", 0.5),
                "is_consistent": metrics.get("is_consistent", False),
                "reasoning": f"Judge parse failed: {str(e)}. Using metric-based evaluation.",
                "main_differences": [],
                "responses": responses,
                "metrics": metrics
            }
