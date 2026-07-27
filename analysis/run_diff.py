"""Run/Prompt Diff — case-level comparison between two eval reports.

Compares two eval runs (report A vs report B) at the case level — which exact
test cases improved, regressed, or stayed the same across runs. Like git diff
but for eval results. Output: per-model, per-test diff with case status.

Report contract::

    report["models"][model_key]["tests"][test_name]["results"]
        → list of cases, each:
            {
              "case_id": str,
              "scores": {"overall_score": float, ...},   # key may also be flat dict at top level
              "latency": float,
              "category": str,   # optional
              "question": str,   # optional, may also be "input_text" or "prompt"
              "error": str,      # optional — if set, treat score as 0.0
            }
    report["models"][model_key]["tests"][test_name]["summary"]["overall_score"]
    report["models"][model_key]["overall_metrics"]["weighted_score"]

CLI::

    python -m analysis.run_diff REPORT_A.json REPORT_B.json
        [--label-a NAME] [--label-b NAME]
        [--regression-threshold 0.05] [--improvement-threshold 0.05]
        [--format text|json|markdown]
        [--output FILE]
        [--model MODEL_KEY]

Exit codes: 0 = no regressions, 1 = regressions found, 2 = error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_case_scores(report: dict) -> dict:
    """Extract case scores from report.

    Returns nested dict:
    {
      model_key: {
        test_name: {
          case_id: {"score": float, "category": str, "text": str}
        }
      }
    }

    score = overall_score from scores dict, or 0.0 if "error" key present.
    text = best of: case["question"] / case["input_text"] / case["prompt"] / case_id
    category = case["category"] if present, else "unknown"
    """
    out: dict = {}

    models = report.get("models", {}) or {}
    for model_key, model_data in models.items():
        model_out: dict = {}
        tests = model_data.get("tests", {}) or {}
        for test_name, test_data in tests.items():
            test_out: dict = {}
            results = test_data.get("results", []) or []
            for case in results:
                if not isinstance(case, dict):
                    continue

                case_id = case.get("case_id", "unknown")

                # Extract score
                score = 0.0
                if "error" in case and case["error"]:
                    score = 0.0
                else:
                    scores_dict = case.get("scores", {})
                    if isinstance(scores_dict, dict):
                        score = scores_dict.get("overall_score", 0.0)
                    else:
                        score = 0.0

                # Coerce to float
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0.0

                # Extract text
                text = (
                    case.get("question")
                    or case.get("input_text")
                    or case.get("prompt")
                    or case_id
                )
                if not isinstance(text, str):
                    text = case_id

                # Extract category
                category = case.get("category", "unknown")
                if not isinstance(category, str):
                    category = "unknown"

                test_out[case_id] = {
                    "score": score,
                    "category": category,
                    "text": text,
                }

            model_out[test_name] = test_out

        out[model_key] = model_out

    return out


# --------------------------------------------------------------------------- #
# Diffing
# --------------------------------------------------------------------------- #


def diff_reports(
    report_a: dict,
    report_b: dict,
    *,
    label_a: str = "run_a",
    label_b: str = "run_b",
    regression_threshold: float = 0.05,
    improvement_threshold: float = 0.05,
) -> dict:
    """Compare two reports case-by-case.

    Returns:
    {
      "label_a": str,
      "label_b": str,
      "regression_threshold": float,
      "improvement_threshold": float,
      "summary": {
        "total_cases_a": int,
        "total_cases_b": int,
        "improved": int,        # delta >= improvement_threshold
        "regressed": int,       # delta <= -regression_threshold
        "unchanged": int,       # |delta| < both thresholds
        "added": int,           # in B but not A
        "removed": int,         # in A but not B
      },
      "models": {
        model_key: {
          "score_a": float,     # weighted_score from report_a, None if missing
          "score_b": float,     # weighted_score from report_b, None if missing
          "score_delta": float | None,
          "tests": {
            test_name: {
              "score_a": float | None,
              "score_b": float | None,
              "score_delta": float | None,
              "cases": [
                {
                  "case_id": str,
                  "status": "improved" | "regressed" | "unchanged" | "added" | "removed",
                  "score_a": float | None,
                  "score_b": float | None,
                  "delta": float | None,    # score_b - score_a, None if added/removed
                  "category": str,
                  "text": str,
                },
                ...  # sorted: regressed first, then improved, then unchanged, then added/removed
              ]
            }
          }
        }
      }
    }

    Only models that appear in at least one report are included.
    Only tests that appear in at least one report are included.
    """
    cases_a = extract_case_scores(report_a)
    cases_b = extract_case_scores(report_b)

    # Collect all model keys
    all_models = set(cases_a.keys()) | set(cases_b.keys())

    # Initialize counters
    summary = {
        "total_cases_a": 0,
        "total_cases_b": 0,
        "improved": 0,
        "regressed": 0,
        "unchanged": 0,
        "added": 0,
        "removed": 0,
    }

    models_out: dict = {}

    for model_key in sorted(all_models):
        tests_a = cases_a.get(model_key, {})
        tests_b = cases_b.get(model_key, {})

        # Model-level scores
        score_a = _get_weighted_score(report_a, model_key)
        score_b = _get_weighted_score(report_b, model_key)
        score_delta = None
        if score_a is not None and score_b is not None:
            score_delta = score_b - score_a

        # Collect all test names
        all_tests = set(tests_a.keys()) | set(tests_b.keys())

        tests_out: dict = {}

        for test_name in sorted(all_tests):
            cases_a_in_test = tests_a.get(test_name, {})
            cases_b_in_test = tests_b.get(test_name, {})

            # Test-level scores
            test_score_a = _get_test_score(report_a, model_key, test_name)
            test_score_b = _get_test_score(report_b, model_key, test_name)
            test_delta = None
            if test_score_a is not None and test_score_b is not None:
                test_delta = test_score_b - test_score_a

            # Collect all case ids
            all_case_ids = set(cases_a_in_test.keys()) | set(cases_b_in_test.keys())

            cases_out: list = []

            for case_id in all_case_ids:
                case_a = cases_a_in_test.get(case_id)
                case_b = cases_b_in_test.get(case_id)

                if case_a is None and case_b is not None:
                    # Added case
                    status = "added"
                    summary["added"] += 1
                    cases_out.append({
                        "case_id": case_id,
                        "status": status,
                        "score_a": None,
                        "score_b": case_b["score"],
                        "delta": None,
                        "category": case_b["category"],
                        "text": case_b["text"],
                    })
                elif case_a is not None and case_b is None:
                    # Removed case
                    status = "removed"
                    summary["removed"] += 1
                    cases_out.append({
                        "case_id": case_id,
                        "status": status,
                        "score_a": case_a["score"],
                        "score_b": None,
                        "delta": None,
                        "category": case_a["category"],
                        "text": case_a["text"],
                    })
                else:
                    # Both present
                    score_a_val = case_a["score"]
                    score_b_val = case_b["score"]
                    delta = score_b_val - score_a_val

                    if delta >= improvement_threshold:
                        status = "improved"
                        summary["improved"] += 1
                    elif delta <= -regression_threshold:
                        status = "regressed"
                        summary["regressed"] += 1
                    else:
                        status = "unchanged"
                        summary["unchanged"] += 1

                    cases_out.append({
                        "case_id": case_id,
                        "status": status,
                        "score_a": score_a_val,
                        "score_b": score_b_val,
                        "delta": delta,
                        "category": case_a["category"],
                        "text": case_a["text"],
                    })

            # Count total cases
            summary["total_cases_a"] += len(cases_a_in_test)
            summary["total_cases_b"] += len(cases_b_in_test)

            # Sort cases: regressed first, improved second, unchanged third, added/removed last
            def sort_key(case_dict):
                status_order = {
                    "regressed": 0,
                    "improved": 1,
                    "unchanged": 2,
                    "added": 3,
                    "removed": 4,
                }
                return (status_order.get(case_dict["status"], 5), case_dict["case_id"])

            cases_out.sort(key=sort_key)

            tests_out[test_name] = {
                "score_a": test_score_a,
                "score_b": test_score_b,
                "score_delta": test_delta,
                "cases": cases_out,
            }

        models_out[model_key] = {
            "score_a": score_a,
            "score_b": score_b,
            "score_delta": score_delta,
            "tests": tests_out,
        }

    return {
        "label_a": label_a,
        "label_b": label_b,
        "regression_threshold": regression_threshold,
        "improvement_threshold": improvement_threshold,
        "summary": summary,
        "models": models_out,
    }


def _get_weighted_score(report: dict, model_key: str) -> Optional[float]:
    """Get weighted_score for a model, or None if missing."""
    model = (report.get("models", {}) or {}).get(model_key, {})
    metrics = model.get("overall_metrics", {}) or {}
    score = metrics.get("weighted_score")
    if score is not None:
        try:
            return float(score)
        except (ValueError, TypeError):
            return None
    return None


def _get_test_score(report: dict, model_key: str, test_name: str) -> Optional[float]:
    """Get overall_score for a test, or None if missing."""
    model = (report.get("models", {}) or {}).get(model_key, {})
    test = (model.get("tests", {}) or {}).get(test_name, {})
    summary = test.get("summary", {}) or {}
    score = summary.get("overall_score")
    if score is not None:
        try:
            return float(score)
        except (ValueError, TypeError):
            return None
    return None


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def _fmt_score(score: Optional[float]) -> str:
    """Format score as 2 decimal places, or '-' if None."""
    if score is None:
        return "-"
    return f"{score:.2f}"


def _truncate_text(text: str, max_len: int = 60) -> str:
    """Truncate text to max_len chars, adding ... if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _status_icon(status: str) -> str:
    """Return icon/symbol for status."""
    icons = {
        "improved": "↑",
        "regressed": "↓",
        "unchanged": "·",
        "added": "+",
        "removed": "-",
    }
    return icons.get(status, " ")


def format_diff_text(diff: dict) -> str:
    """Human-readable text output.

    Example:

    Run Diff: run_a → run_b
    ════════════════════════
    Summary: 3 improved ↑  5 regressed ↓  12 unchanged  1 added  2 removed

    Model: gpt-4o  (0.82 → 0.85, +0.03)
      Test: turkish_qa  (0.78 → 0.81, +0.03)
        ↓ REGRESSED  case_003  [reasoning]  0.90 → 0.72  (-0.18)  "Hangisi daha büyük?"
        ↑ IMPROVED   case_007  [support]    0.55 → 0.88  (+0.33)  "Fatura nasıl kesilir?"
        ·  UNCHANGED  case_001  [policy]     0.80 → 0.81  (+0.01)  "İade süresi nedir?"
        +  ADDED      case_011  [new]                     -        "Yeni soru?"
        -  REMOVED    case_005  [old]        0.75          -
    """
    lines = []

    label_a = diff["label_a"]
    label_b = diff["label_b"]
    summary = diff["summary"]

    lines.append(f"Run Diff: {label_a} → {label_b}")
    lines.append("═" * 50)

    # Summary line
    summary_parts = []
    if summary["improved"] > 0:
        summary_parts.append(f"{summary['improved']} improved ↑")
    if summary["regressed"] > 0:
        summary_parts.append(f"{summary['regressed']} regressed ↓")
    if summary["unchanged"] > 0:
        summary_parts.append(f"{summary['unchanged']} unchanged")
    if summary["added"] > 0:
        summary_parts.append(f"{summary['added']} added")
    if summary["removed"] > 0:
        summary_parts.append(f"{summary['removed']} removed")

    if summary_parts:
        lines.append("Summary: " + "  ".join(summary_parts))
    else:
        lines.append("Summary: No changes")

    lines.append("")

    # Per-model output
    for model_key in sorted(diff["models"].keys()):
        model_data = diff["models"][model_key]
        score_a = model_data["score_a"]
        score_b = model_data["score_b"]
        score_delta = model_data["score_delta"]

        score_str = f"{_fmt_score(score_a)} → {_fmt_score(score_b)}"
        if score_delta is not None:
            sign = "+" if score_delta >= 0 else ""
            score_str += f", {sign}{score_delta:.2f}"

        lines.append(f"Model: {model_key}  ({score_str})")

        for test_name in sorted(model_data["tests"].keys()):
            test_data = model_data["tests"][test_name]
            test_a = test_data["score_a"]
            test_b = test_data["score_b"]
            test_delta = test_data["score_delta"]

            test_str = f"{_fmt_score(test_a)} → {_fmt_score(test_b)}"
            if test_delta is not None:
                sign = "+" if test_delta >= 0 else ""
                test_str += f", {sign}{test_delta:.2f}"

            lines.append(f"  Test: {test_name}  ({test_str})")

            for case in test_data["cases"]:
                icon = _status_icon(case["status"])
                status_text = case["status"].upper()

                if case["status"] in ("added", "removed"):
                    # No delta display for added/removed
                    text_display = _truncate_text(case["text"])
                    lines.append(
                        f"    {icon}  {status_text:<10} {case['case_id']:<10} "
                        f"[{case['category']:<8}] {_fmt_score(case['score_a']):<6} → "
                        f"{_fmt_score(case['score_b']):<6}  -        \"{text_display}\""
                    )
                else:
                    # Both present
                    score_a_val = _fmt_score(case["score_a"])
                    score_b_val = _fmt_score(case["score_b"])
                    delta = case["delta"]
                    sign = "+" if delta >= 0 else ""
                    delta_str = f"{sign}{delta:.2f}"
                    text_display = _truncate_text(case["text"])

                    lines.append(
                        f"    {icon}  {status_text:<10} {case['case_id']:<10} "
                        f"[{case['category']:<8}] {score_a_val} → {score_b_val}  "
                        f"({delta_str:<6})  \"{text_display}\""
                    )

    return "\n".join(lines)


def format_diff_markdown(diff: dict) -> str:
    """Markdown output with ## headers, bold labels, and a summary table."""
    lines = []

    label_a = diff["label_a"]
    label_b = diff["label_b"]
    summary = diff["summary"]

    lines.append(f"# Run Diff: {label_a} → {label_b}")
    lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Improved**: {summary['improved']}")
    lines.append(f"- **Regressed**: {summary['regressed']}")
    lines.append(f"- **Unchanged**: {summary['unchanged']}")
    lines.append(f"- **Added**: {summary['added']}")
    lines.append(f"- **Removed**: {summary['removed']}")
    lines.append(f"- **Total cases (A)**: {summary['total_cases_a']}")
    lines.append(f"- **Total cases (B)**: {summary['total_cases_b']}")
    lines.append("")

    # Per-model details
    for model_key in sorted(diff["models"].keys()):
        model_data = diff["models"][model_key]
        score_a = model_data["score_a"]
        score_b = model_data["score_b"]

        lines.append(f"## Model: {model_key}")
        lines.append("")
        lines.append(
            f"**Score**: {_fmt_score(score_a)} → {_fmt_score(score_b)}"
        )
        lines.append("")

        for test_name in sorted(model_data["tests"].keys()):
            test_data = model_data["tests"][test_name]

            lines.append(f"### Test: {test_name}")
            lines.append("")
            lines.append(
                f"**Test Score**: {_fmt_score(test_data['score_a'])} → "
                f"{_fmt_score(test_data['score_b'])}"
            )
            lines.append("")

            # Table header
            lines.append(
                "| Status | Case ID | Category | Score A | Score B | Delta | Text |"
            )
            lines.append("|--------|---------|----------|---------|---------|-------|------|")

            for case in test_data["cases"]:
                status = case["status"]
                case_id = case["case_id"]
                category = case["category"]
                score_a_val = _fmt_score(case["score_a"])
                score_b_val = _fmt_score(case["score_b"])
                delta = case["delta"]
                text = _truncate_text(case["text"])

                if delta is not None:
                    sign = "+" if delta >= 0 else ""
                    delta_str = f"{sign}{delta:.2f}"
                else:
                    delta_str = "-"

                lines.append(
                    f"| {status} | {case_id} | {category} | {score_a_val} | "
                    f"{score_b_val} | {delta_str} | {text} |"
                )

            lines.append("")

    return "\n".join(lines)


def format_diff_json(diff: dict) -> str:
    """Returns json.dumps(diff, ensure_ascii=False, indent=2)"""
    return json.dumps(diff, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Compare two evaluation runs at the case level."
    )
    parser.add_argument("report_a", help="Path to first report (JSON)")
    parser.add_argument("report_b", help="Path to second report (JSON)")
    parser.add_argument(
        "--label-a", default="run_a", help="Label for report A (default: run_a)"
    )
    parser.add_argument(
        "--label-b", default="run_b", help="Label for report B (default: run_b)"
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.05,
        help="Score drop threshold for regression (default: 0.05)",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.05,
        help="Score gain threshold for improvement (default: 0.05)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--model", help="Filter to single model (optional)"
    )

    args = parser.parse_args()

    try:
        with open(args.report_a) as f:
            report_a = json.load(f)
    except Exception as e:
        print(f"Error loading {args.report_a}: {e}", file=sys.stderr)
        return 2

    try:
        with open(args.report_b) as f:
            report_b = json.load(f)
    except Exception as e:
        print(f"Error loading {args.report_b}: {e}", file=sys.stderr)
        return 2

    try:
        diff = diff_reports(
            report_a,
            report_b,
            label_a=args.label_a,
            label_b=args.label_b,
            regression_threshold=args.regression_threshold,
            improvement_threshold=args.improvement_threshold,
        )

        # Filter to single model if requested
        if args.model:
            diff["models"] = {
                k: v for k, v in diff["models"].items() if k == args.model
            }

        # Format output
        if args.format == "json":
            output = format_diff_json(diff)
        elif args.format == "markdown":
            output = format_diff_markdown(diff)
        else:
            output = format_diff_text(diff)

        # Write output
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
        else:
            print(output)

        # Exit code: 0 if no regressions, 1 if regressions found
        if diff["summary"]["regressed"] > 0:
            return 1
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
