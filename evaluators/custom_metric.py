"""Custom metric builder — natural language description → LLM-as-judge prompt."""

import json
import re
import math
import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def _render_prompt(prompt: str, case: dict) -> str:
    """Fills {question}/{answer}/{expected_answer} placeholders in prompt."""
    filled = prompt.format_map(defaultdict(str, {
        "question": case.get("question", ""),
        "answer": case.get("answer", ""),
        "expected_answer": case.get("expected_answer", ""),
    }))
    return filled


def _parse_judge_response(raw: str) -> dict:
    """
    Extracts {"score": float, "reasoning": str} from JSON.
    Strips markdown fences. Returns {"score": None, "reasoning": raw} if unparseable.
    """
    raw_stripped = raw.strip()

    # Strip markdown code fences
    if raw_stripped.startswith("```"):
        raw_stripped = re.sub(r"^```(?:json)?\n?", "", raw_stripped)
        raw_stripped = re.sub(r"\n?```$", "", raw_stripped).strip()

    try:
        data = json.loads(raw_stripped)
        score = data.get("score")
        reasoning = data.get("reasoning", "")

        # Clamp score to [0, 1] if present
        if score is not None:
            score = max(0.0, min(1.0, float(score)))

        return {"score": score, "reasoning": reasoning}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"score": None, "reasoning": raw}


def _pearson_correlation(xs: list, ys: list):
    """
    Calculates Pearson correlation coefficient.
    Returns None if < 3 pairs or zero variance.
    """
    if len(xs) < 3 or len(ys) < 3 or len(xs) != len(ys):
        return None

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    numerator = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(len(xs)))

    sum_sq_x = sum((x - mean_x) ** 2 for x in xs)
    sum_sq_y = sum((y - mean_y) ** 2 for y in ys)

    if sum_sq_x == 0 or sum_sq_y == 0:
        return None

    denominator = math.sqrt(sum_sq_x * sum_sq_y)

    return numerator / denominator


def generate_judge_prompt(description: str, *, llm_fn=None) -> str:
    """
    Natural language description → structured LLM-as-judge prompt.

    If llm_fn is None, use a template-based approach (no LLM needed).
    If llm_fn provided, use it to enhance/refine the prompt.

    Output prompt template must contain placeholders:
      {question}, {answer}, {expected_answer}
    And must instruct the judge to return JSON: {"score": <float 0-1>, "reasoning": "<str>"}

    Args:
        description: e.g. "Rate how empathetic the response is, 0 to 1"
        llm_fn: optional callable(messages: list[dict]) -> str
    Returns:
        A complete judge prompt string with {question}/{answer}/{expected_answer} placeholders
    """

    # Base template - double-braces for literal placeholders
    base_prompt = f"""Sen bir değerlendirme uzmanısın. Aşağıdaki kritere göre yanıtı değerlendir.

KRİTER: {description}

SORU: {{question}}
BEKLENEN CEVAP: {{expected_answer}}
VERİLEN CEVAP: {{answer}}

Yanıtı 0.0 ile 1.0 arasında bir puanla değerlendir.
0.0 = kritere hiç uymayan yanıt
0.5 = kısmen uyan yanıt
1.0 = kritere tam uyan yanıt

Sadece JSON formatında yanıt ver:
{{"score": <0.0-1.0 arası ondalıklı sayı>, "reasoning": "<Türkçe kısa açıklama>"}}"""

    if llm_fn is None:
        return base_prompt

    # Try to enhance with llm_fn
    messages = [
        {
            "role": "user",
            "content": f"You are helping create an evaluation prompt. Improve this judge prompt for clarity and effectiveness, keeping all placeholders intact:\n\n{base_prompt}\n\nReturn only the improved prompt."
        }
    ]

    try:
        refined = llm_fn(messages)
        # Check if refined prompt has the required placeholders
        if "{question}" in refined and "{answer}" in refined and "{expected_answer}" in refined:
            return refined
    except Exception:
        pass

    # Fall back to base prompt if enhancement fails
    return base_prompt


def evaluate_with_custom_metric(case: dict, prompt: str, *, llm_fn) -> dict:
    """
    Run custom metric on a single case.

    case: {"question": str, "answer": str, "expected_answer": str, ...}
    prompt: judge prompt template with {question}/{answer}/{expected_answer}

    Returns: {"score": float, "reasoning": str, "raw": str}
    Raises ValueError if response is unparseable.
    """
    rendered = _render_prompt(prompt, case)

    messages = [
        {"role": "user", "content": rendered}
    ]

    raw_response = llm_fn(messages)
    parsed = _parse_judge_response(raw_response)

    return {
        "score": parsed["score"],
        "reasoning": parsed["reasoning"],
        "raw": raw_response
    }


def calibrate_metric(prompt: str, examples: list, *, llm_fn) -> dict:
    """
    Run prompt on labeled examples, compute alignment with expected scores.

    examples: list of {"question": str, "answer": str, "expected_answer": str, "expected_score": float}

    For each example: call llm_fn with the rendered prompt, parse JSON response.

    Returns:
    {
        "total_examples": int,
        "successful_evaluations": int,
        "mean_absolute_error": float,  # avg |predicted - expected|
        "correlation": float,           # Pearson r (or None if < 3 examples)
        "alignment_level": str,         # "good" (MAE<0.15), "moderate" (MAE<0.30), "poor"
        "per_example": [{"expected": float, "predicted": float, "reasoning": str, "error": float}]
    }
    """
    per_example = []
    predicted_scores = []
    expected_scores = []
    successful = 0

    for example in examples:
        result = evaluate_with_custom_metric(example, prompt, llm_fn=llm_fn)
        predicted = result["score"] if result["score"] is not None else 0.0
        expected = example.get("expected_score", 0.5)
        error = abs(predicted - expected)

        if result["score"] is not None:
            successful += 1
            predicted_scores.append(result["score"])
            expected_scores.append(expected)

        per_example.append({
            "expected": expected,
            "predicted": predicted,
            "reasoning": result["reasoning"],
            "error": error
        })

    total_examples = len(examples)

    # Calculate MAE over all examples (using 0.0 for None scores)
    mean_absolute_error = (
        sum(ex["error"] for ex in per_example) / total_examples
        if total_examples > 0
        else 0.0
    )

    # Calculate correlation (only for successful evaluations with >= 3 examples)
    correlation = (
        _pearson_correlation(expected_scores, predicted_scores)
        if len(predicted_scores) >= 3
        else None
    )

    # Determine alignment level
    if mean_absolute_error < 0.15:
        alignment_level = "good"
    elif mean_absolute_error < 0.30:
        alignment_level = "moderate"
    else:
        alignment_level = "poor"

    return {
        "total_examples": total_examples,
        "successful_evaluations": successful,
        "mean_absolute_error": mean_absolute_error,
        "correlation": correlation,
        "alignment_level": alignment_level,
        "per_example": per_example
    }


def save_metric(
    name: str,
    prompt: str,
    *,
    description: str = "",
    calibration: dict = None,
    output_dir: str = None
) -> str:
    """
    Save custom metric to YAML file.

    File: <output_dir>/<name>.yaml  (default output_dir: "custom_metrics/")
    Creates directory if needed.

    YAML structure:
      name: str
      description: str
      prompt: str
      calibration: dict or null
      created_at: ISO datetime str
      schema_version: "1"

    Returns: absolute path string of saved file.
    """
    if output_dir is None:
        output_dir = "custom_metrics"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_data = {
        "name": name,
        "description": description,
        "prompt": prompt,
        "calibration": calibration,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1"
    }

    if _YAML_AVAILABLE:
        file_path = output_path / f"{name}.yaml"
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(metric_data, f, allow_unicode=True, default_flow_style=False)
    else:
        file_path = output_path / f"{name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metric_data, f, indent=2, ensure_ascii=False)

    return str(file_path.absolute())


def load_metric(path: str) -> dict:
    """Load custom metric YAML or JSON. Returns dict with same keys as save_metric produces."""
    file_path = Path(path)

    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise RuntimeError("yaml module not available for loading .yaml files")
            return yaml.safe_load(f)
        else:
            return json.load(f)


__all__ = [
    "generate_judge_prompt",
    "calibrate_metric",
    "evaluate_with_custom_metric",
    "save_metric",
    "load_metric",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and calibrate custom metrics for LLM evaluation"
    )
    parser.add_argument("--name", required=True, help="Metric name")
    parser.add_argument("--description", default="", help="Metric description")
    parser.add_argument("--calibrate", help="Path to JSON file with calibration examples")
    parser.add_argument("--output-dir", default="custom_metrics", help="Output directory")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--save", action="store_true", help="Save generated prompt")

    args = parser.parse_args()

    # Generate prompt
    prompt = generate_judge_prompt(args.description, llm_fn=None)

    if args.format == "json":
        print(json.dumps({"prompt": prompt}, indent=2, ensure_ascii=False))
    else:
        print("Generated prompt:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)

    # Calibrate if examples provided
    if args.calibrate:
        print("\nNote: Calibration requires LLM function (not available in CLI mode)")
    elif args.save:
        path = save_metric(
            args.name,
            prompt,
            description=args.description,
            output_dir=args.output_dir
        )
        print(f"\nMetric saved to: {path}")
