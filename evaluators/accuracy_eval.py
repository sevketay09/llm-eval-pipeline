"""
Accuracy Evaluator
Exact match, fuzzy match, ve numerical accuracy için evaluator
"""
import re
from typing import Dict, Any, Optional
from difflib import SequenceMatcher


class AccuracyEvaluator:
    """Traditional accuracy metrics"""
    
    @staticmethod
    def exact_match(predicted: str, expected: str, case_sensitive: bool = False) -> float:
        """Exact match score"""
        if not case_sensitive:
            predicted = predicted.lower().strip()
            expected = expected.lower().strip()
        else:
            predicted = predicted.strip()
            expected = expected.strip()
        
        return 1.0 if predicted == expected else 0.0
    
    @staticmethod
    def fuzzy_match(predicted: str, expected: str, threshold: float = 0.8) -> float:
        """
        Fuzzy string matching using SequenceMatcher
        Returns similarity ratio (0-1)
        """
        predicted = predicted.lower().strip()
        expected = expected.lower().strip()
        
        ratio = SequenceMatcher(None, predicted, expected).ratio()
        return ratio if ratio >= threshold else 0.0
    
    @staticmethod
    def contains_answer(predicted: str, expected: str) -> float:
        """Check if expected answer is contained in predicted"""
        predicted = predicted.lower().strip()
        expected = expected.lower().strip()
        
        # Split expected into key phrases
        expected_phrases = [phrase.strip() for phrase in expected.split(',')]
        
        # Check if any key phrase is in predicted
        for phrase in expected_phrases:
            if phrase in predicted:
                return 1.0
        
        return 0.0
    
    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """Extract first number from text"""
        # Remove common Turkish/English number separators
        text = text.replace('.', '').replace(',', '.')
        
        # Try to find number
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except:
                return None
        return None
    
    @staticmethod
    def numerical_accuracy(predicted: str, expected: str, tolerance: float = 0.01) -> float:
        """
        Compare numerical answers with tolerance
        tolerance: relative error threshold (0.01 = 1%)
        """
        pred_num = AccuracyEvaluator.extract_number(predicted)
        exp_num = AccuracyEvaluator.extract_number(expected)
        
        if pred_num is None or exp_num is None:
            return 0.0
        
        if exp_num == 0:
            return 1.0 if abs(pred_num) < 1e-6 else 0.0
        
        relative_error = abs(pred_num - exp_num) / abs(exp_num)
        return 1.0 if relative_error <= tolerance else 0.0
    
    @staticmethod
    def evaluate(
        predicted: str,
        expected: str,
        eval_type: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Auto-select appropriate evaluation method
        
        Args:
            predicted: Model's answer
            expected: Ground truth
            eval_type: "exact", "fuzzy", "contains", "numerical", or "auto"
        
        Returns:
            {"score": float, "method": str, "details": dict}
        """
        if eval_type == "auto":
            # Try to detect if it's numerical
            pred_num = AccuracyEvaluator.extract_number(predicted)
            exp_num = AccuracyEvaluator.extract_number(expected)
            
            if pred_num is not None and exp_num is not None:
                eval_type = "numerical"
            elif len(expected.split()) <= 5:  # Short answers
                eval_type = "fuzzy"
            else:  # Long answers
                eval_type = "contains"
        
        # Evaluate based on type
        if eval_type == "exact":
            score = AccuracyEvaluator.exact_match(predicted, expected)
            method = "exact_match"
        elif eval_type == "fuzzy":
            threshold = kwargs.get("threshold", 0.8)
            score = AccuracyEvaluator.fuzzy_match(predicted, expected, threshold)
            method = "fuzzy_match"
        elif eval_type == "contains":
            score = AccuracyEvaluator.contains_answer(predicted, expected)
            method = "contains"
        elif eval_type == "numerical":
            tolerance = kwargs.get("tolerance", 0.01)
            score = AccuracyEvaluator.numerical_accuracy(predicted, expected, tolerance)
            method = "numerical"
        else:
            raise ValueError(f"Unknown eval_type: {eval_type}")
        
        return {
            "score": score,
            "method": method,
            "predicted": predicted,
            "expected": expected
        }


class FunctionCallingEvaluator:
    """Evaluate function calling accuracy"""
    
    @staticmethod
    def evaluate_tool_selection(
        selected_tool: Optional[str],
        expected_tool: str
    ) -> float:
        """Check if correct tool was selected"""
        if selected_tool is None:
            return 0.0
        return 1.0 if selected_tool == expected_tool else 0.0
    
    @staticmethod
    def evaluate_parameters(
        selected_params: Dict[str, Any],
        expected_params: Dict[str, Any],
        strict: bool = False
    ) -> float:
        """
        Compare tool parameters for correctness.

        Args:
            selected_params: Parameters extracted by model
            expected_params: Reference parameters
            strict: Penalize extra parameters when True

        Returns:
            Score between 0 and 1
        """
        if not expected_params:
            return 1.0

        correct = 0
        total = len(expected_params)

        for key, expected_value in expected_params.items():
            if key not in selected_params:
                continue

            selected_value = selected_params[key]

            # Type matching
            if type(selected_value) != type(expected_value):
                # Try type conversion
                try:
                    if isinstance(expected_value, (int, float)):
                        selected_value = float(selected_value)
                        expected_value = float(expected_value)
                    elif isinstance(expected_value, str):
                        selected_value = str(selected_value)
                except Exception:
                    continue

            # Value matching
            if isinstance(expected_value, (int, float)):
                # Numerical comparison with tolerance
                if abs(selected_value - expected_value) / max(abs(expected_value), 1) < 0.01:
                    correct += 1
            elif isinstance(expected_value, str):
                # String comparison (case-insensitive)
                if selected_value.lower().strip() == expected_value.lower().strip():
                    correct += 1
            else:
                # Direct equality
                if selected_value == expected_value:
                    correct += 1

        score = correct / total

        if strict:
            # Check for extra parameters (penalty)
            extra_params = set(selected_params.keys()) - set(expected_params.keys())
            if extra_params:
                score *= 0.8  # 20% penalty for extra params

        return score
    
    @staticmethod
    def evaluate(
        tool_calls: Optional[list],
        expected_tool: str,
        expected_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete function calling evaluation
        
        Returns:
            {
                "tool_selection_score": float,
                "parameter_score": float,
                "overall_score": float,
                "details": dict
            }
        """
        if not tool_calls or len(tool_calls) == 0:
            return {
                "tool_selection_score": 0.0,
                "parameter_score_lenient": 0.0,
                "parameter_score_strict": 0.0,
                "overall_score_lenient": 0.0,
                "overall_score_strict": 0.0,
                "parameter_score": 0.0,
                "overall_score": 0.0,
                "details": {"error": "No tool calls made"}
            }
        
        # Take first tool call
        tool_call = tool_calls[0]
        selected_tool = tool_call.get("name")
        selected_params = tool_call.get("arguments", {})
        
        # Evaluate tool selection
        tool_score = FunctionCallingEvaluator.evaluate_tool_selection(
            selected_tool, expected_tool
        )
        
        # Evaluate parameters
        param_score_lenient = FunctionCallingEvaluator.evaluate_parameters(
            selected_params, expected_params, strict=False
        )
        param_score_strict = FunctionCallingEvaluator.evaluate_parameters(
            selected_params, expected_params, strict=True
        )

        # Overall scores (weighted)
        overall_score_lenient = 0.4 * tool_score + 0.6 * param_score_lenient
        overall_score_strict = 0.4 * tool_score + 0.6 * param_score_strict

        return {
            "tool_selection_score": tool_score,
            "parameter_score_lenient": param_score_lenient,
            "parameter_score_strict": param_score_strict,
            "overall_score_lenient": overall_score_lenient,
            "overall_score_strict": overall_score_strict,
            "parameter_score": param_score_lenient,
            "overall_score": overall_score_lenient,
            "details": {
                "selected_tool": selected_tool,
                "expected_tool": expected_tool,
                "selected_params": selected_params,
                "expected_params": expected_params
            }
        }
