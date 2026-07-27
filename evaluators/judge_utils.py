"""
Shared helpers for LLM-as-judge evaluators.

Parse failures must never produce a fake 0.0 score — they bias aggregate
metrics downward. Instead: retry once, then return None so downstream
metric builders (which drop non-numeric values) exclude the item.
"""
import json
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Passed to UnifiedLLMAdapter.generate; applied only when the provider
# supports response_format (adapter checks supports_response_format).
JSON_RESPONSE_FORMAT = {"type": "json_object"}


def strip_code_fences(content: str) -> str:
    """Extract payload from ```json ...``` or ``` ...``` fenced blocks."""
    if "```json" in content:
        return content.split("```json")[1].split("```")[0].strip()
    if "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    return content


def request_judge_json(
    judge: Any,
    messages: List[Dict[str, str]],
    tag: str,
    max_attempts: int = 2,
) -> Optional[Dict[str, Any]]:
    """Call the judge model and parse its JSON verdict, retrying once.

    Returns the parsed dict, or None if all attempts fail to produce
    a JSON object. Never fabricates a score.
    """
    for attempt in range(1, max_attempts + 1):
        result = judge.generate(messages, response_format=JSON_RESPONSE_FORMAT)
        content = (result.get("content") or "").strip()
        try:
            parsed = json.loads(strip_code_fences(content))
            if isinstance(parsed, dict):
                return parsed
            raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
        except Exception as e:
            logger.warning(
                f"[{tag}] parse failed (attempt {attempt}/{max_attempts}): {e} | raw: {content[:200]!r}"
            )
    return None


def extract_score(parsed: Optional[Dict[str, Any]], tag: str) -> Optional[float]:
    """Pull a numeric 'score' out of a judge verdict; None if absent/invalid."""
    if not isinstance(parsed, dict):
        return None
    try:
        return float(parsed.get("score"))
    except (TypeError, ValueError):
        logger.warning(f"[{tag}] non-numeric score: {parsed.get('score')!r}")
        return None
