"""
Prompt Compression Evaluator
Tests how different prompt lengths affect model performance
"""
from typing import Dict, List, Any, Optional, Tuple
from adapters.unified_adapter import UnifiedLLMAdapter
import re


class PromptCompressionEvaluator:
    """
    Evaluates model performance under different prompt compression strategies.
    
    Tests:
    - Original (full context)
    - Compressed (key points only)
    - Minimal (ultra-short)
    - Progressive compression levels (100%, 75%, 50%, 25%)
    
    Metrics:
    - Quality score (does answer change?)
    - Token savings
    - Latency improvements
    - Information retention rate
    """
    
    def __init__(self, judge_adapter: UnifiedLLMAdapter = None):
        self.judge = judge_adapter
    
    def evaluate_prompt_compression(
        self,
        model: UnifiedLLMAdapter,
        original_prompt: str,
        compressed_prompts: Dict[str, str],
        expected_answer: str = None,
        question_type: str = "qa"
    ) -> Dict[str, Any]:
        """
        Evaluate model performance with different prompt compressions.
        
        Args:
            model: Model to test
            original_prompt: Full original prompt
            compressed_prompts: Dict of {compression_level: compressed_prompt}
                e.g., {"75%": "...", "50%": "...", "25%": "..."}
            expected_answer: Expected/reference answer
            question_type: Type of question (qa, reasoning, summarization, etc.)
        
        Returns:
            Detailed comparison metrics
        """
        results = {}

        # Baseline: Original prompt. If this fails there's nothing to compare
        # compressed levels against, so exclude the whole item rather than
        # comparing compressed responses to an empty baseline.
        baseline_result = self._run_with_prompt(
            model,
            original_prompt,
            label="original"
        )
        if baseline_result.get("error"):
            return {"generation_error": baseline_result["error"]}
        results["original"] = baseline_result

        # Compressed versions. A single failed level is excluded on its own —
        # the other levels can still be compared against the (successful) baseline.
        for compression_level, compressed_prompt in compressed_prompts.items():
            compressed_result = self._run_with_prompt(
                model,
                compressed_prompt,
                label=compression_level
            )
            if compressed_result.get("error"):
                continue
            results[compression_level] = compressed_result
        
        # Compare results
        comparison = self._compare_results(
            results,
            original_prompt,
            expected_answer,
            question_type
        )
        
        return comparison
    
    def _run_with_prompt(
        self,
        model: UnifiedLLMAdapter,
        prompt: str,
        label: str
    ) -> Dict[str, Any]:
        """Run model with given prompt"""
        messages = [
            {"role": "system", "content": "Sen yardımcı bir asistansın. Soruları doğru ve öz bir şekilde cevapla."},
            {"role": "user", "content": prompt}
        ]
        
        result = model.generate(messages, temperature=0.0)
        if result.get("error"):
            return {"label": label, "error": result["error"]}

        response = result.get('content', '')

        # Calculate token counts
        input_tokens = len(prompt.split())  # Rough estimate
        output_tokens = len(response.split()) if response else 0

        return {
            "label": label,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "prompt_tokens_estimate": input_tokens,
            "response": response,
            "response_length": len(response) if response else 0,
            "response_tokens_estimate": output_tokens,
            "latency": result.get('latency', 0),
        }
    
    def _compare_results(
        self,
        results: Dict[str, Dict],
        original_prompt: str,
        expected_answer: str,
        question_type: str
    ) -> Dict[str, Any]:
        """Compare compression results"""
        
        baseline = results["original"]
        
        comparison_data = {
            "baseline": {
                "prompt_length": baseline["prompt_length"],
                "prompt_tokens": baseline["prompt_tokens_estimate"],
                "response": baseline["response"],
                "latency": baseline["latency"],
            },
            "compressions": {},
            "metrics": {},
            "recommendation": None
        }
        
        best_compression = None
        best_score = -1
        
        for level, result in results.items():
            if level == "original":
                continue
            
            # Calculate savings
            prompt_reduction = (1 - result["prompt_length"] / baseline["prompt_length"]) * 100
            token_savings = baseline["prompt_tokens_estimate"] - result["prompt_tokens_estimate"]
            latency_improvement = ((baseline["latency"] - result["latency"]) / baseline["latency"] * 100) if baseline["latency"] > 0 else 0
            
            # Calculate semantic similarity between baseline and compressed response
            similarity = self._calculate_response_similarity(
                baseline["response"],
                result["response"]
            )
            
            # Information retention (how much of baseline answer is in compressed answer)
            retention = self._calculate_information_retention(
                baseline["response"],
                result["response"]
            )
            
            # Quality score: balance between compression and retention
            # High compression + High retention = Best
            quality_score = (similarity * 0.5 + retention * 0.5) * (prompt_reduction / 100)
            
            compression_data = {
                "prompt_length": result["prompt_length"],
                "prompt_tokens": result["prompt_tokens_estimate"],
                "response": result["response"],
                "latency": result["latency"],
                "metrics": {
                    "prompt_reduction_pct": prompt_reduction,
                    "token_savings": token_savings,
                    "latency_improvement_pct": latency_improvement,
                    "response_similarity": similarity,
                    "information_retention": retention,
                    "quality_score": quality_score
                }
            }
            
            comparison_data["compressions"][level] = compression_data
            
            # Track best compression (highest quality score)
            if quality_score > best_score:
                best_score = quality_score
                best_compression = level
        
        # Overall metrics
        if comparison_data["compressions"]:
            avg_prompt_reduction = sum(c["metrics"]["prompt_reduction_pct"] for c in comparison_data["compressions"].values()) / len(comparison_data["compressions"])
            avg_retention = sum(c["metrics"]["information_retention"] for c in comparison_data["compressions"].values()) / len(comparison_data["compressions"])
            
            comparison_data["metrics"] = {
                "average_prompt_reduction": avg_prompt_reduction,
                "average_information_retention": avg_retention,
                "best_compression_level": best_compression,
                "best_quality_score": best_score
            }
            
            # Recommendation
            if best_score > 0.7:
                comparison_data["recommendation"] = f"Use {best_compression} compression - good quality retention with significant savings"
            elif best_score > 0.5:
                comparison_data["recommendation"] = f"Use {best_compression} compression with caution - moderate quality retention"
            else:
                comparison_data["recommendation"] = "Compression not recommended - significant quality loss"
        
        return comparison_data
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate semantic similarity between two responses"""
        from difflib import SequenceMatcher
        
        if not response1 or not response2:
            return 0.0
        
        # Normalize
        r1 = response1.lower().strip()
        r2 = response2.lower().strip()
        
        # Use SequenceMatcher
        similarity = SequenceMatcher(None, r1, r2).ratio()
        
        return similarity
    
    def _calculate_information_retention(self, baseline: str, compressed: str) -> float:
        """
        Calculate what percentage of baseline information is retained in compressed.
        Uses key terms extraction.
        """
        if not baseline or not compressed:
            return 0.0
        
        # Extract key terms (words longer than 3 chars, excluding common words)
        stop_words = {'için', 'olan', 'veya', 'ancak', 'fakat', 'çünkü', 'daha', 'sonra', 
                      'önce', 'kadar', 'gibi', 'bile', 'henüz', 'artık', 'hala', 'sadece'}
        
        def extract_key_terms(text: str) -> set:
            words = re.findall(r'\w+', text.lower())
            return {w for w in words if len(w) > 3 and w not in stop_words}
        
        baseline_terms = extract_key_terms(baseline)
        compressed_terms = extract_key_terms(compressed)
        
        if not baseline_terms:
            return 1.0
        
        # Calculate overlap
        retained_terms = baseline_terms.intersection(compressed_terms)
        retention = len(retained_terms) / len(baseline_terms)
        
        return retention
    
    def compress_prompt_auto(
        self,
        original_prompt: str,
        target_reduction: float = 0.5
    ) -> str:
        """
        Automatically compress prompt to target reduction.
        Simple heuristic-based compression.
        
        Args:
            original_prompt: Original prompt text
            target_reduction: Target reduction (0.5 = reduce to 50% of original)
        
        Returns:
            Compressed prompt
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', original_prompt)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return original_prompt
        
        # Calculate target number of sentences
        target_count = max(1, int(len(sentences) * target_reduction))
        
        # Score sentences by importance (length and position)
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            # Prefer earlier sentences and longer sentences (more info)
            position_score = 1.0 - (idx / len(sentences))
            length_score = min(1.0, len(sentence) / 100)
            
            score = position_score * 0.6 + length_score * 0.4
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top N
        scored_sentences.sort(reverse=True)
        selected = scored_sentences[:target_count]
        
        # Re-order by original position
        selected_sentences = []
        for sentence in sentences:
            if any(sentence == s[1] for s in selected):
                selected_sentences.append(sentence)
        
        compressed = '. '.join(selected_sentences)
        
        # Ensure it ends with punctuation
        if compressed and compressed[-1] not in '.!?':
            compressed += '.'
        
        return compressed
    
    def generate_compression_variants(
        self,
        original_prompt: str,
        levels: List[float] = None
    ) -> Dict[str, str]:
        """
        Generate multiple compression variants at different levels.
        
        Args:
            original_prompt: Original prompt
            levels: List of reduction levels (default: [0.75, 0.50, 0.25])
        
        Returns:
            Dict of {level_label: compressed_prompt}
        """
        if levels is None:
            levels = [0.75, 0.50, 0.25]
        
        variants = {}
        
        for level in levels:
            compressed = self.compress_prompt_auto(original_prompt, level)
            label = f"{int(level * 100)}%"
            variants[label] = compressed
        
        return variants


def evaluate_prompt_compression(
    model: UnifiedLLMAdapter,
    original_prompt: str,
    compressed_prompts: Dict[str, str],
    expected_answer: str = None,
    question_type: str = "qa"
) -> Dict[str, Any]:
    """
    Convenience function for prompt compression evaluation.
    
    Args:
        model: Model to test
        original_prompt: Full original prompt
        compressed_prompts: Dict of compression variants
        expected_answer: Optional expected answer
        question_type: Question type
    
    Returns:
        Comparison results
    """
    evaluator = PromptCompressionEvaluator()
    return evaluator.evaluate_prompt_compression(
        model,
        original_prompt,
        compressed_prompts,
        expected_answer,
        question_type
    )
