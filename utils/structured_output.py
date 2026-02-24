"""
Helpers for enforcing and validating structured outputs.
"""
import json
from typing import Any, Dict, Optional, Tuple
from jsonschema import validate, ValidationError


def extract_json(content: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extract a JSON object from model output.

    Returns:
        (parsed_json, error_message)
    """
    if not content:
        return None, "empty_content"

    raw = content.strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return None, "json_not_object"
        return parsed, None
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"


def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    """
    Validate data against JSON schema. Returns error message or None.
    """
    try:
        validate(instance=data, schema=schema)
        return None
    except ValidationError as exc:
        return f"schema_validation_error: {exc.message}"


def build_response_format(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build OpenAI-style response_format for JSON schema.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": schema,
            "strict": True
        }
    }
