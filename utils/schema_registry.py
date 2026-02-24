"""
JSON schema registry for structured outputs.
"""
from typing import Dict, Any


DEFAULT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {
        "answer": {"type": "string", "minLength": 1}
    }
}

REASONING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["final_answer", "reasoning"],
    "properties": {
        "final_answer": {"type": "string", "minLength": 1},
        "reasoning": {"type": "string", "minLength": 1}
    }
}

RAG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "citations": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

AGENTIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["plan", "answer"],
    "properties": {
        "plan": {"type": "string", "minLength": 1},
        "answer": {"type": "string", "minLength": 1}
    }
}


def get_schema_for_test(test_name: str) -> Dict[str, Any]:
    """Return JSON schema for a test type."""
    if test_name in {"turkish_reasoning", "fintech_calculations", "gsm8k"}:
        return REASONING_SCHEMA
    if test_name in {"rag_test", "truthfulqa"}:
        return RAG_SCHEMA
    if test_name in {"agentic_workflows"}:
        return AGENTIC_SCHEMA
    return DEFAULT_SCHEMA
