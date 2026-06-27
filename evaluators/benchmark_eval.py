"""
Benchmark evaluators for standard datasets.
"""
import re
from typing import Dict, Any, List, Optional
from .accuracy_eval import AccuracyEvaluator


def _normalize_choice(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def evaluate_multiple_choice(
    predicted: str,
    choices: List[str],
    correct_index: Optional[int] = None,
    correct_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate multiple choice answer.
    """
    normalized_pred = _normalize_choice(predicted)

    matched_index = None
    for idx, choice in enumerate(choices):
        if _normalize_choice(choice) in normalized_pred:
            matched_index = idx
            break

    if matched_index is None:
        letter_match = re.search(r"\b([abcd])\b", normalized_pred)
        if letter_match:
            matched_index = ord(letter_match.group(1)) - ord("a")

    is_correct = False
    if correct_index is not None and matched_index is not None:
        is_correct = matched_index == correct_index
    elif correct_text is not None and matched_index is not None:
        is_correct = _normalize_choice(choices[matched_index]) == _normalize_choice(correct_text)

    return {
        "score": 1.0 if is_correct else 0.0,
        "matched_index": matched_index,
        "correct_index": correct_index,
        "correct_text": correct_text
    }


def extract_gsm8k_answer(text: str) -> str:
    """Extract final numeric answer from GSM8K reference format."""
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def evaluate_gsm8k(predicted: str, expected: str) -> Dict[str, Any]:
    """Evaluate GSM8K numeric answers."""
    expected_final = extract_gsm8k_answer(expected)
    score = AccuracyEvaluator.numerical_accuracy(predicted, expected_final, tolerance=0.01)
    return {
        "score": score,
        "expected_final": expected_final
    }
