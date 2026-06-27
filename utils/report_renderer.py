"""Shared report renderers for terminal and markdown summaries."""
from __future__ import annotations

from html import escape
from typing import Any, Mapping

from utils.policy_reporting import build_policy_audit_summary_from_results, build_policy_summary_from_results


CASE_PANEL_LIMIT = 4
CASE_TEXT_LIMIT = 160
CASE_REASON_LIMIT = 180
SNAPSHOT_VALUE_LIMIT = 220
TRACE_SPAN_DETAIL_LIMIT = 6


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _format_decimal(value: Any, digits: int = 3, suffix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}{suffix}"
    return "n/a"


def _format_latency(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}s"
    return "n/a"


def _format_percent(value: Any, digits: int = 1) -> str:
    if isinstance(value, (int, float)):
        candidate = float(value)
        if candidate <= 1.0:
            candidate *= 100.0
        return f"{candidate:.{digits}f}%"
    return "n/a"


def _format_integer(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    return "n/a"


def _format_currency(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${float(value):.4f}"
    return "n/a"


def _humanize_policy_decision(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "n/a"
    return value.replace("_", " ").strip().title()


def _format_count_breakdown(counts: Any, limit: int = 3) -> str:
    items = [
        (str(key), int(value))
        for key, value in _coerce_mapping(counts).items()
        if isinstance(value, (int, float))
    ]
    items.sort(key=lambda item: (-item[1], item[0]))
    if not items:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in items[:limit])


def _format_path_step(span: Mapping[str, Any]) -> str:
    span_type = str(span.get("span_type") or "unknown")
    name = span.get("name")
    if isinstance(name, str) and name.strip() and name not in {"unknown", span_type}:
        return f"{span_type}({name.strip()})"
    return span_type


def _build_trace_path_signature(spans: list[Any], limit_steps: int = 6) -> str:
    steps = [_format_path_step(span) for span in spans if isinstance(span, Mapping)]
    if not steps:
        return "unknown"
    if len(steps) <= limit_steps:
        return " > ".join(steps)
    head = steps[: max(1, limit_steps - 1)]
    return " > ".join(head + [f"... ({len(steps)} steps)"])


def _truncate_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_case_id(result: Mapping[str, Any], test_name: str, index: int) -> str:
    return _first_text(result.get("id"), result.get("test_id"), result.get("case_id")) or f"{test_name}-{index + 1}"


def _slugify(value: str) -> str:
    normalized = [character.lower() if character.isalnum() else "-" for character in value]
    slug = "".join(normalized)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "trace"


def _extract_case_score(result: Mapping[str, Any]) -> float | None:
    candidates = [
        result.get("overall_score"),
        result.get("score"),
        _coerce_mapping(result.get("scores")).get("overall_score"),
        _coerce_mapping(result.get("scores")).get("score"),
        result.get("llm_judge_score"),
    ]
    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            return float(candidate)

    for boolean_key in ("passed", "success"):
        candidate = result.get(boolean_key)
        if isinstance(candidate, bool):
            return 1.0 if candidate else 0.0
    return None


def _extract_case_reason(result: Mapping[str, Any]) -> str:
    details = _coerce_mapping(result.get("details"))
    metadata = _coerce_mapping(result.get("metadata"))
    return _first_text(
        result.get("error"),
        result.get("reason"),
        result.get("reasoning"),
        result.get("llm_judge_reasoning"),
        details.get("reason"),
        details.get("reasoning"),
        metadata.get("queue_reason"),
    )


def _build_case_panels(results: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    panels: dict[str, list[dict[str, Any]]] = {}
    for model_key, model_payload in _coerce_mapping(results.get("models")).items():
        tests = _coerce_mapping(_coerce_mapping(model_payload).get("tests"))
        model_cases: list[dict[str, Any]] = []
        for test_name, test_payload in tests.items():
            test_results = _coerce_mapping(test_payload).get("results")
            if not isinstance(test_results, list):
                continue
            for index, raw_result in enumerate(test_results):
                if not isinstance(raw_result, Mapping):
                    continue
                result = dict(raw_result)
                score = _extract_case_score(result)
                error_text = _first_text(result.get("error"))
                failed = bool(error_text)
                if isinstance(result.get("passed"), bool):
                    failed = failed or (result.get("passed") is False)
                if isinstance(result.get("success"), bool):
                    failed = failed or (result.get("success") is False)
                if isinstance(score, float):
                    failed = failed or score < 0.5

                prompt_text = _first_text(
                    result.get("question"),
                    result.get("input"),
                    result.get("prompt"),
                    result.get("task"),
                    result.get("input_text"),
                )
                answer_text = _first_text(
                    result.get("response"),
                    result.get("output"),
                    result.get("model_answer"),
                    result.get("answer"),
                )
                reason_text = _extract_case_reason(result)
                latency = result.get("latency")
                model_cases.append(
                    {
                        "model": model_key,
                        "test_name": test_name,
                        "case_id": _build_case_id(result, test_name, index),
                        "score": score,
                        "failed": failed,
                        "has_error": bool(error_text),
                        "prompt": _truncate_text(prompt_text, CASE_TEXT_LIMIT),
                        "answer": _truncate_text(answer_text, CASE_TEXT_LIMIT),
                        "reason": _truncate_text(reason_text, CASE_REASON_LIMIT),
                        "latency": float(latency) if isinstance(latency, (int, float)) else None,
                        "has_trace": bool(_coerce_mapping(result.get("trace"))),
                        "trace_ref": None,
                    }
                )

        model_cases.sort(
            key=lambda item: (
                0 if item.get("failed") else 1,
                0 if item.get("has_error") else 1,
                item.get("score") if isinstance(item.get("score"), float) else 999.0,
                -(item.get("latency") or 0.0),
                item.get("test_name") or "",
                item.get("case_id") or "",
            )
        )
        if model_cases:
            panels[model_key] = model_cases[:CASE_PANEL_LIMIT]
    return panels


def _build_trace_detail_sections(
    results: Mapping[str, Any],
    case_panels: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    selected_keys = {
        (str(model_key), str(case.get("test_name") or ""), str(case.get("case_id") or ""))
        for model_key, cases in case_panels.items()
        for case in cases
        if isinstance(case, Mapping) and case.get("has_trace")
    }
    details: dict[str, list[dict[str, Any]]] = {}
    if not selected_keys:
        return details

    for model_key, model_payload in _coerce_mapping(results.get("models")).items():
        tests = _coerce_mapping(_coerce_mapping(model_payload).get("tests"))
        model_details: list[dict[str, Any]] = []
        for test_name, test_payload in tests.items():
            test_results = _coerce_mapping(test_payload).get("results")
            if not isinstance(test_results, list):
                continue
            for index, raw_result in enumerate(test_results):
                if not isinstance(raw_result, Mapping):
                    continue
                result = dict(raw_result)
                case_id = _build_case_id(result, str(test_name), index)
                selection_key = (str(model_key), str(test_name), case_id)
                if selection_key not in selected_keys:
                    continue

                trace = _coerce_mapping(result.get("trace"))
                spans = trace.get("spans") if isinstance(trace.get("spans"), list) else []
                if not spans:
                    continue

                trace_ref = "trace-{}-{}-{}".format(
                    _slugify(str(model_key)),
                    _slugify(str(test_name)),
                    _slugify(case_id),
                )
                span_rows = []
                for span in spans[:TRACE_SPAN_DETAIL_LIMIT]:
                    if not isinstance(span, Mapping):
                        continue
                    metadata = _coerce_mapping(span.get("metadata"))
                    span_rows.append(
                        {
                            "span_type": str(span.get("span_type") or "unknown"),
                            "name": str(span.get("name") or "unknown"),
                            "status": str(span.get("status") or "unknown"),
                            "error_type": str(metadata.get("error_type") or span.get("error") or "none"),
                        }
                    )

                model_details.append(
                    {
                        "trace_ref": trace_ref,
                        "trace_id": str(trace.get("trace_id") or trace_ref),
                        "case_id": case_id,
                        "test_name": str(test_name),
                        "path_signature": _build_trace_path_signature(spans),
                        "span_count": len(spans),
                        "failed_spans": sum(
                            1
                            for span in spans
                            if isinstance(span, Mapping) and span.get("status") == "failed"
                        ),
                        "span_rows": span_rows,
                    }
                )

        if model_details:
            details[str(model_key)] = model_details

    for model_key, cases in case_panels.items():
        detail_map = {
            (detail.get("test_name"), detail.get("case_id")): detail.get("trace_ref")
            for detail in details.get(str(model_key), [])
        }
        for case in cases:
            if not isinstance(case, dict):
                continue
            case["trace_ref"] = detail_map.get((case.get("test_name"), case.get("case_id")))

    return details


def _build_trace_summary(results: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    trace_summary: dict[str, dict[str, Any]] = {}
    for model_key, model_payload in _coerce_mapping(results.get("models")).items():
        tests = _coerce_mapping(_coerce_mapping(model_payload).get("tests"))
        aggregate = {
            "trace_count": 0,
            "trace_cases_with_failures": 0,
            "total_spans": 0,
            "failed_spans": 0,
            "partial_spans": 0,
            "tool_failures": 0,
            "span_types": {},
            "tool_failure_names": {},
            "tool_failure_error_types": {},
            "path_signatures": {},
        }

        for test_payload in tests.values():
            test_results = _coerce_mapping(test_payload).get("results")
            if not isinstance(test_results, list):
                continue

            for raw_result in test_results:
                if not isinstance(raw_result, Mapping):
                    continue

                result = dict(raw_result)
                trace = _coerce_mapping(result.get("trace"))
                if not trace:
                    continue

                spans = trace.get("spans") if isinstance(trace.get("spans"), list) else []
                trace_summary_map = _coerce_mapping(trace.get("summary"))
                span_types_map = _coerce_mapping(trace_summary_map.get("span_types"))

                total_spans = trace_summary_map.get("total_spans")
                if not isinstance(total_spans, (int, float)):
                    total_spans = len(spans)

                failed_spans = trace_summary_map.get("failed_spans")
                if not isinstance(failed_spans, (int, float)):
                    failed_spans = sum(
                        1
                        for span in spans
                        if isinstance(span, Mapping) and span.get("status") == "failed"
                    )

                partial_spans = sum(
                    1
                    for span in spans
                    if isinstance(span, Mapping) and span.get("status") == "partial"
                )
                tool_failures = sum(
                    1
                    for span in spans
                    if isinstance(span, Mapping)
                    and span.get("span_type") == "tool"
                    and span.get("status") == "failed"
                )

                if not span_types_map and spans:
                    for span in spans:
                        if not isinstance(span, Mapping):
                            continue
                        span_type = str(span.get("span_type") or "unknown")
                        span_types_map[span_type] = span_types_map.get(span_type, 0) + 1

                aggregate["trace_count"] += 1
                aggregate["total_spans"] += int(total_spans)
                aggregate["failed_spans"] += int(failed_spans)
                aggregate["partial_spans"] += partial_spans
                aggregate["tool_failures"] += tool_failures
                if int(failed_spans) > 0 or partial_spans > 0:
                    aggregate["trace_cases_with_failures"] += 1

                path_signature = _build_trace_path_signature(spans)
                aggregate["path_signatures"][path_signature] = (
                    aggregate["path_signatures"].get(path_signature, 0) + 1
                )

                for span_type, count in span_types_map.items():
                    if isinstance(count, (int, float)):
                        aggregate["span_types"][str(span_type)] = (
                            aggregate["span_types"].get(str(span_type), 0) + int(count)
                        )

                for span in spans:
                    if not isinstance(span, Mapping):
                        continue
                    if span.get("span_type") != "tool" or span.get("status") != "failed":
                        continue

                    tool_name = str(span.get("name") or "unknown")
                    aggregate["tool_failure_names"][tool_name] = (
                        aggregate["tool_failure_names"].get(tool_name, 0) + 1
                    )

                    metadata = _coerce_mapping(span.get("metadata"))
                    error_type = metadata.get("error_type") or span.get("error") or "unknown"
                    error_type_label = str(error_type)
                    aggregate["tool_failure_error_types"][error_type_label] = (
                        aggregate["tool_failure_error_types"].get(error_type_label, 0) + 1
                    )

        if aggregate["trace_count"] > 0:
            aggregate["avg_spans_per_trace"] = aggregate["total_spans"] / aggregate["trace_count"]
            trace_summary[model_key] = aggregate

    return trace_summary


def _build_report_sections(results: Mapping[str, Any]) -> dict[str, Any]:
    run_metadata = _coerce_mapping(results.get("run_metadata"))
    summary = _coerce_mapping(results.get("summary"))
    model_comparison = _coerce_mapping(summary.get("model_comparison"))
    best_performers = _coerce_mapping(summary.get("best_performers"))
    trends = _coerce_mapping(results.get("trends"))
    config_snapshot = _coerce_mapping(run_metadata.get("config_snapshot"))
    snapshot_parameters = _coerce_mapping(config_snapshot.get("parameters"))
    snapshot_env_vars = _coerce_mapping(config_snapshot.get("env_vars_present"))
    runtime_overrides = _coerce_mapping(snapshot_parameters.get("runtime_overrides"))
    snapshot_models = snapshot_parameters.get("model_keys") or []
    case_panels = _build_case_panels(results)
    trace_details = _build_trace_detail_sections(results, case_panels)

    return {
        "timestamp": results.get("timestamp") or run_metadata.get("timestamp"),
        "run_id": run_metadata.get("run_id") or config_snapshot.get("run_id"),
        "suite": run_metadata.get("test_suite"),
        "prompt_version": run_metadata.get("prompt_version") or run_metadata.get("judge_prompt_version"),
        "schema_version": run_metadata.get("schema_version") or results.get("version"),
        "metric_version": run_metadata.get("metric_version"),
        "selected_tests": run_metadata.get("selected_tests") or [],
        "result_hash": run_metadata.get("result_hash"),
        "config_checksum": run_metadata.get("config_checksum"),
        "tests_config_checksum": run_metadata.get("tests_config_checksum"),
        "snapshot_models": snapshot_models if isinstance(snapshot_models, list) else [],
        "snapshot_runtime_overrides": runtime_overrides,
        "snapshot_env_vars_present": snapshot_env_vars,
        "model_comparison": model_comparison,
        "best_performers": best_performers,
        "policy": build_policy_summary_from_results(results),
        "policy_audit": build_policy_audit_summary_from_results(results),
        "trends": trends,
        "case_panels": case_panels,
        "trace_summary": _build_trace_summary(results),
        "trace_details": trace_details,
    }


def render_terminal_summary(results: Mapping[str, Any]) -> str:
    sections = _build_report_sections(results)
    lines = ["", "=" * 80, "EVALUATION SUMMARY", "=" * 80, ""]

    suite = sections.get("suite")
    run_id = sections.get("run_id")
    result_hash = sections.get("result_hash")
    if suite or run_id or result_hash:
        lines.append("Run Metadata:")
        if suite:
            lines.append(f"  Suite: {suite}")
        if run_id:
            lines.append(f"  Run ID: {run_id}")
        if sections.get("timestamp"):
            lines.append(f"  Timestamp: {sections['timestamp']}")
        if sections.get("prompt_version"):
            lines.append(f"  Prompt Version: {sections['prompt_version']}")
        if sections.get("schema_version"):
            lines.append(f"  Schema Version: {sections['schema_version']}")
        if sections.get("metric_version"):
            lines.append(f"  Metric Version: {sections['metric_version']}")
        if result_hash:
            lines.append(f"  Result Hash: {result_hash}")
        selected_tests = sections.get("selected_tests") or []
        if selected_tests:
            lines.append(f"  Selected Tests: {', '.join(str(item) for item in selected_tests)}")
        snapshot_models = sections.get("snapshot_models") or []
        if snapshot_models:
            lines.append(f"  Snapshot Models: {', '.join(str(item) for item in snapshot_models)}")
        runtime_overrides = sections.get("snapshot_runtime_overrides") or {}
        if runtime_overrides:
            lines.append(f"  Runtime Overrides: {_truncate_text(str(runtime_overrides), SNAPSHOT_VALUE_LIMIT)}")
        env_vars_present = sections.get("snapshot_env_vars_present") or {}
        present_keys = [key for key, is_present in env_vars_present.items() if is_present]
        if env_vars_present:
            lines.append(f"  Env Vars Present: {', '.join(present_keys) if present_keys else 'none'}")
        lines.append("")

    for model_key, comparison in sections["model_comparison"].items():
        comparison_map = _coerce_mapping(comparison)
        lines.append(f"{model_key}:")
        lines.append(f"  Overall Score: {_format_decimal(comparison_map.get('overall_score'))}")
        lines.append(f"  Avg Latency: {_format_latency(comparison_map.get('avg_latency'))}")
        lines.append(f"  P95 Latency: {_format_latency(comparison_map.get('latency_p95'))}")
        lines.append(f"  Throughput: {_format_decimal(comparison_map.get('tokens_per_second'), 1, ' tokens/s')}")
        lines.append(
            f"  Tokens: in {_format_integer(comparison_map.get('total_input_tokens'))} | out {_format_integer(comparison_map.get('total_output_tokens'))} | total {_format_integer(comparison_map.get('total_tokens'))}"
        )
        lines.append(f"  Cost: {_format_currency(comparison_map.get('total_cost'))}")
        if comparison_map.get("quality_latency_efficiency") is not None:
            lines.append(
                f"  Quality/Latency: {_format_decimal(comparison_map.get('quality_latency_efficiency'))}"
            )
        if comparison_map.get("judge_agreement_rate") is not None:
            lines.append(
                f"  Judge Agreement: {_format_percent(comparison_map.get('judge_agreement_rate'))}"
            )
        if comparison_map.get("judge_disagreement_mean") is not None:
            lines.append(
                f"  Judge Disagreement: {_format_decimal(comparison_map.get('judge_disagreement_mean'))}"
            )
        lines.append("")

    lines.extend(["=" * 80, "BEST PERFORMERS BY CATEGORY", "=" * 80, ""])
    for category, data in sections["best_performers"].items():
        item = _coerce_mapping(data)
        lines.append(f"{category}: {item.get('model', 'n/a')} (score: {_format_decimal(item.get('score'))})")

    policy = _coerce_mapping(sections.get("policy"))
    if int(policy.get("total_policy_cases") or 0) > 0:
        lines.extend(["", "=" * 80, "POLICY SUMMARY", "=" * 80, ""])
        lines.append(f"Policy Cases: {_format_integer(policy.get('total_policy_cases'))}")
        lines.append(f"Flagged Cases: {_format_integer(policy.get('flagged_case_count'))}")
        lines.append(f"High Severity Cases: {_format_integer(policy.get('high_severity_case_count'))}")
        lines.append(f"Queue Candidates: {_format_integer(policy.get('queue_candidate_count'))}")
        lines.append(f"Avg Severity: {_format_decimal(policy.get('avg_severity'))}")
        lines.append(f"Risk Levels: {_format_count_breakdown(policy.get('risk_level_counts'))}")
        for item in policy.get("by_policy_type") or []:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "  - {policy_type}: total {total} | flagged {flagged} | high {high} | avg severity {avg}".format(
                    policy_type=str(item.get("policy_type") or "policy_safety"),
                    total=_format_integer(item.get("total_cases")),
                    flagged=_format_integer(item.get("flagged_cases")),
                    high=_format_integer(item.get("high_severity_cases")),
                    avg=_format_decimal(item.get("avg_severity")),
                )
            )
        top_cases = policy.get("top_cases") or []
        if isinstance(top_cases, list) and top_cases:
            lines.append("")
            lines.append("Top Policy Cases:")
            for case in top_cases[:5]:
                if not isinstance(case, Mapping):
                    continue
                lines.append(
                    "  - [{test}] {case_id} | {policy_type} | {risk_level} | severity {severity}".format(
                        test=str(case.get("test_name") or "unknown"),
                        case_id=str(case.get("test_id") or "unknown"),
                        policy_type=str(case.get("policy_type") or "policy_safety"),
                        risk_level=str(case.get("risk_level") or "low"),
                        severity=_format_decimal(case.get("severity")),
                    )
                )
                if case.get("queue_reason"):
                    lines.append(f"    queue: {case['queue_reason']}")
                if case.get("question"):
                    lines.append(f"    prompt: {_truncate_text(case['question'], CASE_TEXT_LIMIT)}")

    policy_audit = _coerce_mapping(sections.get("policy_audit"))
    if int(policy_audit.get("total_reviews") or 0) > 0:
        lines.extend(["", "=" * 80, "POLICY REVIEW AUDIT", "=" * 80, ""])
        lines.append(f"Total Reviews: {_format_integer(policy_audit.get('total_reviews'))}")
        lines.append(f"Confirmed Violations: {_format_integer(policy_audit.get('confirmed_violation_count'))}")
        lines.append(f"False Positives: {_format_integer(policy_audit.get('false_positive_count'))}")
        lines.append(f"Needs Follow-Up: {_format_integer(policy_audit.get('needs_follow_up_count'))}")
        lines.append(f"Latest Review: {policy_audit.get('latest_review_at') or 'n/a'}")
        recent_reviews = policy_audit.get("recent_reviews") or []
        if isinstance(recent_reviews, list) and recent_reviews:
            lines.append("")
            lines.append("Recent Decisions:")
            for review in recent_reviews[:5]:
                if not isinstance(review, Mapping):
                    continue
                lines.append(
                    "  - [{test}] {case_id} | {decision} | {annotator} | {timestamp}".format(
                        test=str(review.get("test_name") or "unknown"),
                        case_id=str(review.get("test_id") or "unknown"),
                        decision=_humanize_policy_decision(review.get("decision")),
                        annotator=str(review.get("annotator_id") or "unknown"),
                        timestamp=str(review.get("timestamp") or "n/a"),
                    )
                )
                if review.get("queue_reason"):
                    lines.append(f"    queue: {review['queue_reason']}")
                if review.get("notes"):
                    lines.append(f"    notes: {_truncate_text(review['notes'], CASE_REASON_LIMIT)}")

    if sections["case_panels"]:
        lines.extend(["", "=" * 80, "FAIL-FIRST CASE PANELS", "=" * 80, "", f"Top {CASE_PANEL_LIMIT} cases per model, sorted failed-first. Prompt/answer text is truncated for scanability."])
        for model_key, cases in sections["case_panels"].items():
            lines.append("")
            lines.append(f"{model_key}:")
            for case in cases:
                lines.append(
                    f"  - [{case['test_name']}] {case['case_id']} | score {_format_decimal(case.get('score'))} | {'failed' if case.get('failed') else 'review'}"
                )
                if case.get("trace_ref"):
                    lines.append(f"    trace ref: {case['trace_ref']}")
                if case.get("prompt"):
                    lines.append(f"    prompt: {case['prompt']}")
                if case.get("reason"):
                    lines.append(f"    reason: {case['reason']}")
                elif case.get("answer"):
                    lines.append(f"    answer: {case['answer']}")

    if sections["trace_summary"]:
        lines.extend(["", "=" * 80, "TRACE COVERAGE", "=" * 80, ""])
        for model_key, trace_info in sections["trace_summary"].items():
            span_types = ", ".join(
                f"{span_type}={count}"
                for span_type, count in sorted(
                    _coerce_mapping(trace_info.get("span_types")).items(),
                    key=lambda item: (-int(item[1]) if isinstance(item[1], (int, float)) else 0, item[0]),
                )
            )
            lines.append(f"{model_key}:")
            lines.append(f"  Trace Cases: {_format_integer(trace_info.get('trace_count'))}")
            lines.append(f"  Avg Spans/Trace: {_format_decimal(trace_info.get('avg_spans_per_trace'), 1)}")
            lines.append(f"  Failed Trace Cases: {_format_integer(trace_info.get('trace_cases_with_failures'))}")
            lines.append(f"  Failed Spans: {_format_integer(trace_info.get('failed_spans'))}")
            lines.append(f"  Partial Spans: {_format_integer(trace_info.get('partial_spans'))}")
            lines.append(f"  Tool Failures: {_format_integer(trace_info.get('tool_failures'))}")
            if span_types:
                lines.append(f"  Span Mix: {span_types}")
            if trace_info.get("tool_failures"):
                lines.append(
                    f"  Top Tool Failures: {_format_count_breakdown(trace_info.get('tool_failure_names'))}"
                )
                lines.append(
                    f"  Tool Error Types: {_format_count_breakdown(trace_info.get('tool_failure_error_types'))}"
                )
            lines.append(
                f"  Top Agent Paths: {_format_count_breakdown(trace_info.get('path_signatures'), limit=2)}"
            )
            lines.append("")

    if sections["trace_details"]:
        lines.extend(["", "=" * 80, "TRACE DETAIL INDEX", "=" * 80, ""])
        for model_key, detail_items in sections["trace_details"].items():
            lines.append(f"{model_key}:")
            for detail in detail_items:
                lines.append(
                    f"  - {detail['trace_ref']} | [{detail['test_name']}] {detail['case_id']} | spans {_format_integer(detail.get('span_count'))} | failed {_format_integer(detail.get('failed_spans'))}"
                )
                lines.append(f"    path: {detail['path_signature']}")
                for span in detail.get("span_rows", []):
                    lines.append(
                        f"    span: {span['span_type']}::{span['name']} | {span['status']} | error_type {span['error_type']}"
                    )
            lines.append("")

    if sections["trends"]:
        lines.extend(["", "=" * 80, "TRENDS & REGRESSIONS", "=" * 80, ""])
        for model_key, trend_data in sections["trends"].items():
            trend_info = _coerce_mapping(_coerce_mapping(trend_data).get("trend"))
            regressions = _coerce_mapping(trend_data).get("regressions") or []
            lines.append(f"{model_key}:")
            trend_label = trend_info.get("trend", "unknown")
            if trend_label == "insufficient_history":
                lines.append(
                    f"  Trend: insufficient_history (need >=2 runs, got {len(trend_info.get('values', []))})"
                )
            else:
                lines.append(
                    f"  Trend: {trend_label} ({_format_decimal(trend_info.get('change_pct'), 1, '%')})"
                )
            if isinstance(regressions, list) and regressions:
                lines.append(f"  Regressions detected: {len(regressions)}")
                for regression in regressions[:2]:
                    reg = _coerce_mapping(regression)
                    lines.append(
                        f"    - {reg.get('metric', 'unknown')}: {_format_decimal(reg.get('drop_percentage'), 1, '%')} drop"
                    )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_markdown_report(results: Mapping[str, Any]) -> str:
    sections = _build_report_sections(results)
    lines = ["# Evaluation Summary", ""]

    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- Timestamp: {sections.get('timestamp') or 'n/a'}")
    lines.append(f"- Suite: {sections.get('suite') or 'n/a'}")
    lines.append(f"- Run ID: {sections.get('run_id') or 'n/a'}")
    lines.append(f"- Prompt Version: {sections.get('prompt_version') or 'n/a'}")
    lines.append(f"- Schema Version: {sections.get('schema_version') or 'n/a'}")
    lines.append(f"- Metric Version: {sections.get('metric_version') or 'n/a'}")
    lines.append(f"- Result Hash: {sections.get('result_hash') or 'n/a'}")
    lines.append(f"- Config Checksum: {sections.get('config_checksum') or 'n/a'}")
    lines.append(f"- Tests Config Checksum: {sections.get('tests_config_checksum') or 'n/a'}")
    selected_tests = sections.get("selected_tests") or []
    lines.append(
        f"- Selected Tests: {', '.join(str(item) for item in selected_tests) if selected_tests else 'all suite tests'}"
    )
    snapshot_models = sections.get("snapshot_models") or []
    lines.append(
        f"- Snapshot Models: {', '.join(str(item) for item in snapshot_models) if snapshot_models else 'n/a'}"
    )
    runtime_overrides = sections.get("snapshot_runtime_overrides") or {}
    lines.append(
        f"- Runtime Overrides: {_truncate_text(str(runtime_overrides), SNAPSHOT_VALUE_LIMIT) if runtime_overrides else 'none'}"
    )
    env_vars_present = sections.get("snapshot_env_vars_present") or {}
    present_keys = [key for key, is_present in env_vars_present.items() if is_present]
    lines.append(f"- Env Vars Present: {', '.join(present_keys) if present_keys else 'none'}")
    lines.append("")

    lines.append("## Model Comparison")
    lines.append("")
    lines.append("| Model | Overall Score | Avg Latency | P95 Latency | Throughput | Tokens | Cost | Judge Agreement | Judge Disagreement |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model_key, comparison in sections["model_comparison"].items():
        comparison_map = _coerce_mapping(comparison)
        lines.append(
            "| {model} | {score} | {avg_latency} | {p95_latency} | {throughput} | {tokens} | {cost} | {agreement} | {disagreement} |".format(
                model=model_key,
                score=_format_decimal(comparison_map.get("overall_score")),
                avg_latency=_format_latency(comparison_map.get("avg_latency")),
                p95_latency=_format_latency(comparison_map.get("latency_p95")),
                throughput=_format_decimal(comparison_map.get("tokens_per_second"), 1),
                tokens=_format_integer(comparison_map.get("total_tokens")),
                cost=_format_currency(comparison_map.get("total_cost")),
                agreement=_format_percent(comparison_map.get("judge_agreement_rate")),
                disagreement=_format_decimal(comparison_map.get("judge_disagreement_mean")),
            )
        )
    lines.append("")

    lines.append("## Best Performers")
    lines.append("")
    for category, data in sections["best_performers"].items():
        item = _coerce_mapping(data)
        lines.append(f"- {category}: {item.get('model', 'n/a')} ({_format_decimal(item.get('score'))})")
    if not sections["best_performers"]:
        lines.append("- No category winners available")
    lines.append("")

    policy = _coerce_mapping(sections.get("policy"))
    if int(policy.get("total_policy_cases") or 0) > 0:
        lines.append("## Policy Summary")
        lines.append("")
        lines.append(f"- Policy Cases: {_format_integer(policy.get('total_policy_cases'))}")
        lines.append(f"- Flagged Cases: {_format_integer(policy.get('flagged_case_count'))}")
        lines.append(f"- High Severity Cases: {_format_integer(policy.get('high_severity_case_count'))}")
        lines.append(f"- Queue Candidates: {_format_integer(policy.get('queue_candidate_count'))}")
        lines.append(f"- Average Severity: {_format_decimal(policy.get('avg_severity'))}")
        lines.append(f"- Risk Levels: {_format_count_breakdown(policy.get('risk_level_counts'))}")
        lines.append("")
        lines.append("| Policy Type | Total Cases | Flagged | High Severity | Avg Severity |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for item in policy.get("by_policy_type") or []:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "| {policy_type} | {total} | {flagged} | {high} | {avg} |".format(
                    policy_type=str(item.get("policy_type") or "policy_safety"),
                    total=_format_integer(item.get("total_cases")),
                    flagged=_format_integer(item.get("flagged_cases")),
                    high=_format_integer(item.get("high_severity_cases")),
                    avg=_format_decimal(item.get("avg_severity")),
                )
            )
        lines.append("")
        top_cases = policy.get("top_cases") or []
        if isinstance(top_cases, list) and top_cases:
            lines.append("### Top Policy Cases")
            lines.append("")
            for case in top_cases[:5]:
                if not isinstance(case, Mapping):
                    continue
                lines.append(
                    "- [{test}] {case_id} | {policy_type} | {risk_level} | severity {severity}".format(
                        test=str(case.get("test_name") or "unknown"),
                        case_id=str(case.get("test_id") or "unknown"),
                        policy_type=str(case.get("policy_type") or "policy_safety"),
                        risk_level=str(case.get("risk_level") or "low"),
                        severity=_format_decimal(case.get("severity")),
                    )
                )
                if case.get("queue_reason"):
                    lines.append(f"- Queue Reason: {case['queue_reason']}")
                if case.get("question"):
                    lines.append(f"- Prompt: {_truncate_text(case['question'], CASE_TEXT_LIMIT)}")
            lines.append("")

    policy_audit = _coerce_mapping(sections.get("policy_audit"))
    if int(policy_audit.get("total_reviews") or 0) > 0:
        lines.append("## Policy Review Audit")
        lines.append("")
        lines.append(f"- Total Reviews: {_format_integer(policy_audit.get('total_reviews'))}")
        lines.append(f"- Confirmed Violations: {_format_integer(policy_audit.get('confirmed_violation_count'))}")
        lines.append(f"- False Positives: {_format_integer(policy_audit.get('false_positive_count'))}")
        lines.append(f"- Needs Follow-Up: {_format_integer(policy_audit.get('needs_follow_up_count'))}")
        lines.append(f"- Latest Review: {policy_audit.get('latest_review_at') or 'n/a'}")
        lines.append("")
        recent_reviews = policy_audit.get("recent_reviews") or []
        if isinstance(recent_reviews, list) and recent_reviews:
            lines.append("### Recent Review Decisions")
            lines.append("")
            for review in recent_reviews[:5]:
                if not isinstance(review, Mapping):
                    continue
                lines.append(
                    "- [{test}] {case_id} | {decision} | {annotator} | {timestamp}".format(
                        test=str(review.get("test_name") or "unknown"),
                        case_id=str(review.get("test_id") or "unknown"),
                        decision=_humanize_policy_decision(review.get("decision")),
                        annotator=str(review.get("annotator_id") or "unknown"),
                        timestamp=str(review.get("timestamp") or "n/a"),
                    )
                )
                if review.get("queue_reason"):
                    lines.append(f"- Queue Reason: {review['queue_reason']}")
                if review.get("notes"):
                    lines.append(f"- Notes: {_truncate_text(review['notes'], CASE_REASON_LIMIT)}")
            lines.append("")

    if sections["case_panels"]:
        lines.append("## Fail-First Case Panels")
        lines.append("")
        lines.append(f"Top {CASE_PANEL_LIMIT} cases per model, sorted failed-first. Prompt, answer and reason text is truncated for scanability.")
        lines.append("")
        for model_key, cases in sections["case_panels"].items():
            lines.append(f"### {model_key}")
            lines.append("")
            for case in cases:
                lines.append(
                    "- [{test}] {case_id} | score {score} | {state}".format(
                        test=case["test_name"],
                        case_id=case["case_id"],
                        score=_format_decimal(case.get("score")),
                        state="failed" if case.get("failed") else "review",
                    )
                )
                if case.get("trace_ref"):
                    lines.append(f"- Trace Ref: [{case['trace_ref']}](#{case['trace_ref']})")
                if case.get("prompt"):
                    lines.append(f"- Prompt: {case['prompt']}")
                if case.get("reason"):
                    lines.append(f"- Reason: {case['reason']}")
                elif case.get("answer"):
                    lines.append(f"- Answer: {case['answer']}")
            lines.append("")

    if sections["trace_summary"]:
        lines.append("## Trace Coverage")
        lines.append("")
        lines.append("| Model | Trace Cases | Avg Spans / Trace | Failed Trace Cases | Failed Spans | Partial Spans | Tool Failures | Span Mix |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for model_key, trace_info in sections["trace_summary"].items():
            span_mix = ", ".join(
                f"{span_type}={count}"
                for span_type, count in sorted(
                    _coerce_mapping(trace_info.get("span_types")).items(),
                    key=lambda item: (-int(item[1]) if isinstance(item[1], (int, float)) else 0, item[0]),
                )
            ) or "n/a"
            lines.append(
                "| {model} | {trace_cases} | {avg_spans} | {failed_cases} | {failed_spans} | {partial_spans} | {tool_failures} | {span_mix} |".format(
                    model=model_key,
                    trace_cases=_format_integer(trace_info.get("trace_count")),
                    avg_spans=_format_decimal(trace_info.get("avg_spans_per_trace"), 1),
                    failed_cases=_format_integer(trace_info.get("trace_cases_with_failures")),
                    failed_spans=_format_integer(trace_info.get("failed_spans")),
                    partial_spans=_format_integer(trace_info.get("partial_spans")),
                    tool_failures=_format_integer(trace_info.get("tool_failures")),
                    span_mix=span_mix,
                )
            )
        lines.append("")
        for model_key, trace_info in sections["trace_summary"].items():
            if not trace_info.get("tool_failures"):
                lines.append(f"### {model_key} Agent Path Breakdown")
                lines.append("")
                lines.append(
                    f"- Top Agent Paths: {_format_count_breakdown(trace_info.get('path_signatures'), limit=3)}"
                )
                continue
            lines.append(f"### {model_key} Tool Failure Breakdown")
            lines.append("")
            lines.append(
                f"- Top Tool Failures: {_format_count_breakdown(trace_info.get('tool_failure_names'))}"
            )
            lines.append(
                f"- Tool Error Types: {_format_count_breakdown(trace_info.get('tool_failure_error_types'))}"
            )
            lines.append(
                f"- Top Agent Paths: {_format_count_breakdown(trace_info.get('path_signatures'), limit=3)}"
            )
        lines.append("")

    if sections["trace_details"]:
        lines.append("## Trace Detail Index")
        lines.append("")
        for model_key, detail_items in sections["trace_details"].items():
            lines.append(f"### {model_key}")
            lines.append("")
            for detail in detail_items:
                lines.append(f"#### {detail['trace_ref']}")
                lines.append("")
                lines.append(
                    f"- Case: [{detail['test_name']}] {detail['case_id']}"
                )
                lines.append(f"- Trace ID: {detail['trace_id']}")
                lines.append(f"- Path: {detail['path_signature']}")
                lines.append(f"- Span Count: {_format_integer(detail.get('span_count'))}")
                lines.append(f"- Failed Spans: {_format_integer(detail.get('failed_spans'))}")
                for span in detail.get("span_rows", []):
                    lines.append(
                        f"- Span: {span['span_type']}::{span['name']} | {span['status']} | error_type {span['error_type']}"
                    )
                lines.append("")

    if sections["trends"]:
        lines.append("## Trends")
        lines.append("")
        for model_key, trend_data in sections["trends"].items():
            trend_info = _coerce_mapping(_coerce_mapping(trend_data).get("trend"))
            regressions = _coerce_mapping(trend_data).get("regressions") or []
            lines.append(f"### {model_key}")
            if trend_info.get("trend") == "insufficient_history":
                lines.append(f"- Trend: insufficient_history ({len(trend_info.get('values', []))} run)")
            else:
                lines.append(
                    f"- Trend: {trend_info.get('trend', 'unknown')} ({_format_decimal(trend_info.get('change_pct'), 1, '%')})"
                )
            if isinstance(regressions, list) and regressions:
                lines.append(f"- Regressions: {len(regressions)}")
                for regression in regressions[:3]:
                    reg = _coerce_mapping(regression)
                    lines.append(
                        f"- {reg.get('metric', 'unknown')}: {_format_decimal(reg.get('drop_percentage'), 1, '%')} drop"
                    )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_html_report(results: Mapping[str, Any]) -> str:
    sections = _build_report_sections(results)

    metadata_rows = [
        ("Timestamp", sections.get("timestamp") or "n/a"),
        ("Suite", sections.get("suite") or "n/a"),
        ("Run ID", sections.get("run_id") or "n/a"),
        ("Prompt Version", sections.get("prompt_version") or "n/a"),
        ("Schema Version", sections.get("schema_version") or "n/a"),
        ("Metric Version", sections.get("metric_version") or "n/a"),
        ("Result Hash", sections.get("result_hash") or "n/a"),
        ("Config Checksum", sections.get("config_checksum") or "n/a"),
        ("Tests Config Checksum", sections.get("tests_config_checksum") or "n/a"),
        (
            "Selected Tests",
            ", ".join(str(item) for item in (sections.get("selected_tests") or [])) or "all suite tests",
        ),
        (
            "Snapshot Models",
            ", ".join(str(item) for item in (sections.get("snapshot_models") or [])) or "n/a",
        ),
        (
            "Runtime Overrides",
            _truncate_text(str(sections.get("snapshot_runtime_overrides") or {}), SNAPSHOT_VALUE_LIMIT)
            if sections.get("snapshot_runtime_overrides")
            else "none",
        ),
        (
            "Env Vars Present",
            ", ".join(
                key
                for key, is_present in (sections.get("snapshot_env_vars_present") or {}).items()
                if is_present
            )
            or "none",
        ),
    ]

    model_cards = []
    for model_key, comparison in sections["model_comparison"].items():
        comparison_map = _coerce_mapping(comparison)
        model_cards.append(
            """
            <article class="card">
              <h3>{model}</h3>
              <div class="metrics-grid">
                <div><span>Overall Score</span><strong>{overall_score}</strong></div>
                <div><span>Avg Latency</span><strong>{avg_latency}</strong></div>
                <div><span>P95 Latency</span><strong>{latency_p95}</strong></div>
                <div><span>Throughput</span><strong>{throughput}</strong></div>
                <div><span>Total Tokens</span><strong>{tokens}</strong></div>
                <div><span>Cost</span><strong>{cost}</strong></div>
              </div>
            </article>
            """.format(
                model=escape(str(model_key)),
                overall_score=escape(_format_decimal(comparison_map.get("overall_score"))),
                avg_latency=escape(_format_latency(comparison_map.get("avg_latency"))),
                latency_p95=escape(_format_latency(comparison_map.get("latency_p95"))),
                throughput=escape(_format_decimal(comparison_map.get("tokens_per_second"), 1, " tokens/s")),
                tokens=escape(_format_integer(comparison_map.get("total_tokens"))),
                cost=escape(_format_currency(comparison_map.get("total_cost"))),
            )
        )

    best_performer_items = "".join(
        "<li><strong>{category}</strong>: {model} ({score})</li>".format(
            category=escape(str(category)),
            model=escape(str(_coerce_mapping(data).get("model", "n/a"))),
            score=escape(_format_decimal(_coerce_mapping(data).get("score"))),
        )
        for category, data in sections["best_performers"].items()
    ) or "<li>No category winners available</li>"

    policy_summary = _coerce_mapping(sections.get("policy"))
    policy_audit_summary = _coerce_mapping(sections.get("policy_audit"))
    policy_cards = []
    for item in policy_summary.get("by_policy_type") or []:
        if not isinstance(item, Mapping):
            continue
        policy_cards.append(
            """
            <article class="card">
              <h3>{policy_type}</h3>
              <div class="metrics-grid">
                <div><span>Total Cases</span><strong>{total}</strong></div>
                <div><span>Flagged</span><strong>{flagged}</strong></div>
                <div><span>High Severity</span><strong>{high}</strong></div>
                <div><span>Avg Severity</span><strong>{avg}</strong></div>
              </div>
            </article>
            """.format(
                policy_type=escape(str(item.get("policy_type") or "policy_safety")),
                total=escape(_format_integer(item.get("total_cases"))),
                flagged=escape(_format_integer(item.get("flagged_cases"))),
                high=escape(_format_integer(item.get("high_severity_cases"))),
                avg=escape(_format_decimal(item.get("avg_severity"))),
            )
        )

    policy_case_items = []
    for case in (policy_summary.get("top_cases") or [])[:5]:
        if not isinstance(case, Mapping):
            continue
        policy_case_items.append(
            """
            <li class="case-item {state}">
              <div class="case-head">
                <strong>[{test_name}] {case_id}</strong>
                <span>{policy_type} · {risk_level} · severity {severity}</span>
              </div>
              {prompt}
              {queue_reason}
            </li>
            """.format(
                state="failed" if case.get("flagged") else "review",
                test_name=escape(str(case.get("test_name") or "unknown")),
                case_id=escape(str(case.get("test_id") or "unknown")),
                policy_type=escape(str(case.get("policy_type") or "policy_safety")),
                risk_level=escape(str(case.get("risk_level") or "low")),
                severity=escape(_format_decimal(case.get("severity"))),
                prompt=(
                    f"<p><span>Prompt:</span> {escape(_truncate_text(case.get('question'), CASE_TEXT_LIMIT))}</p>"
                    if case.get("question")
                    else ""
                ),
                queue_reason=(
                    f"<p><span>Queue Reason:</span> {escape(str(case.get('queue_reason')))}</p>"
                    if case.get("queue_reason")
                    else ""
                ),
            )
        )

    policy_audit_items = []
    for review in (policy_audit_summary.get("recent_reviews") or [])[:5]:
        if not isinstance(review, Mapping):
            continue
        policy_audit_items.append(
            """
            <li class="case-item review">
              <div class="case-head">
                <strong>[{test_name}] {case_id}</strong>
                <span>{decision} · {annotator}</span>
              </div>
              <p><span>When:</span> {timestamp}</p>
              {prompt}
              {queue_reason}
              {notes}
            </li>
            """.format(
                test_name=escape(str(review.get("test_name") or "unknown")),
                case_id=escape(str(review.get("test_id") or "unknown")),
                decision=escape(_humanize_policy_decision(review.get("decision"))),
                annotator=escape(str(review.get("annotator_id") or "unknown")),
                timestamp=escape(str(review.get("timestamp") or "n/a")),
                prompt=(
                    f"<p><span>Prompt:</span> {escape(_truncate_text(review.get('question'), CASE_TEXT_LIMIT))}</p>"
                    if review.get("question")
                    else ""
                ),
                queue_reason=(
                    f"<p><span>Queue Reason:</span> {escape(str(review.get('queue_reason')))}</p>"
                    if review.get("queue_reason")
                    else ""
                ),
                notes=(
                    f"<p><span>Notes:</span> {escape(_truncate_text(review.get('notes'), CASE_REASON_LIMIT))}</p>"
                    if review.get("notes")
                    else ""
                ),
            )
        )

    case_sections = []
    for model_key, cases in sections["case_panels"].items():
        case_items = []
        for case in cases:
            case_items.append(
                """
                <li class="case-item {state}">
                  <div class="case-head">
                    <strong>[{test_name}] {case_id}</strong>
                    <span>score {score}</span>
                  </div>
                                    {trace_ref}
                  {prompt}
                  {reason}
                  {answer}
                </li>
                """.format(
                    state="failed" if case.get("failed") else "review",
                    test_name=escape(str(case.get("test_name") or "unknown")),
                    case_id=escape(str(case.get("case_id") or "unknown")),
                    score=escape(_format_decimal(case.get("score"))),
                    trace_ref=(
                        f'<p><span>Trace Ref:</span> <a href="#{escape(str(case["trace_ref"]))}">{escape(str(case["trace_ref"]))}</a></p>'
                        if case.get("trace_ref")
                        else ""
                    ),
                    prompt=(
                        f"<p><span>Prompt:</span> {escape(str(case['prompt']))}</p>" if case.get("prompt") else ""
                    ),
                    reason=(
                        f"<p><span>Reason:</span> {escape(str(case['reason']))}</p>" if case.get("reason") else ""
                    ),
                    answer=(
                        f"<p><span>Answer:</span> {escape(str(case['answer']))}</p>"
                        if case.get("answer") and not case.get("reason")
                        else ""
                    ),
                )
            )
        case_sections.append(
            "<section class=\"card\"><h3>{model}</h3><ul class=\"case-list\">{items}</ul></section>".format(
                model=escape(str(model_key)),
                items="".join(case_items),
            )
        )

    trace_detail_sections = []
    for model_key, detail_items in sections["trace_details"].items():
        detail_cards = []
        for detail in detail_items:
            span_items = "".join(
                "<li>{span_type}::{name} | {status} | error_type {error_type}</li>".format(
                    span_type=escape(str(span.get("span_type") or "unknown")),
                    name=escape(str(span.get("name") or "unknown")),
                    status=escape(str(span.get("status") or "unknown")),
                    error_type=escape(str(span.get("error_type") or "none")),
                )
                for span in detail.get("span_rows", [])
            ) or "<li>No span rows available</li>"
            detail_cards.append(
                """
                <article id="{trace_ref}" class="card">
                  <h3>{trace_ref}</h3>
                  <p><span>Case:</span> [{test_name}] {case_id}</p>
                  <p><span>Trace ID:</span> {trace_id}</p>
                  <p><span>Path:</span> {path}</p>
                  <p><span>Span Count:</span> {span_count} | <span>Failed Spans:</span> {failed_spans}</p>
                  <ul>{span_items}</ul>
                </article>
                """.format(
                    trace_ref=escape(str(detail.get("trace_ref") or "trace")),
                    test_name=escape(str(detail.get("test_name") or "unknown")),
                    case_id=escape(str(detail.get("case_id") or "unknown")),
                    trace_id=escape(str(detail.get("trace_id") or "unknown")),
                    path=escape(str(detail.get("path_signature") or "unknown")),
                    span_count=escape(_format_integer(detail.get("span_count"))),
                    failed_spans=escape(_format_integer(detail.get("failed_spans"))),
                    span_items=span_items,
                )
            )
        trace_detail_sections.append(
            '<section><h2>{model} Trace Details</h2><div class="grid">{items}</div></section>'.format(
                model=escape(str(model_key)),
                items="".join(detail_cards),
            )
        )

    trend_sections = []
    for model_key, trend_data in sections["trends"].items():
        trend_info = _coerce_mapping(_coerce_mapping(trend_data).get("trend"))
        regressions = _coerce_mapping(trend_data).get("regressions") or []
        regression_items = "".join(
            "<li>{metric}: {drop}</li>".format(
                metric=escape(str(_coerce_mapping(regression).get("metric", "unknown"))),
                drop=escape(_format_decimal(_coerce_mapping(regression).get("drop_percentage"), 1, "%")),
            )
            for regression in regressions[:3]
        )
        trend_sections.append(
            """
            <section class="card">
              <h3>{model}</h3>
              <p>Trend: <strong>{trend}</strong></p>
              {regressions}
            </section>
            """.format(
                model=escape(str(model_key)),
                trend=escape(
                    "insufficient_history"
                    if trend_info.get("trend") == "insufficient_history"
                    else f"{trend_info.get('trend', 'unknown')} ({_format_decimal(trend_info.get('change_pct'), 1, '%')})"
                ),
                regressions=(f"<ul>{regression_items}</ul>" if regression_items else ""),
            )
        )

    trace_sections = []
    for model_key, trace_info in sections["trace_summary"].items():
        span_mix = "".join(
            "<li><strong>{span_type}</strong>: {count}</li>".format(
                span_type=escape(str(span_type)),
                count=escape(_format_integer(count)),
            )
            for span_type, count in sorted(
                _coerce_mapping(trace_info.get("span_types")).items(),
                key=lambda item: (-int(item[1]) if isinstance(item[1], (int, float)) else 0, item[0]),
            )
        ) or "<li>No span mix available</li>"
        trace_sections.append(
            """
            <section class="card">
              <h3>{model}</h3>
              <div class="metrics-grid">
                <div><span>Trace Cases</span><strong>{trace_cases}</strong></div>
                <div><span>Avg Spans / Trace</span><strong>{avg_spans}</strong></div>
                <div><span>Failed Trace Cases</span><strong>{failed_cases}</strong></div>
                <div><span>Failed Spans</span><strong>{failed_spans}</strong></div>
                <div><span>Partial Spans</span><strong>{partial_spans}</strong></div>
                <div><span>Tool Failures</span><strong>{tool_failures}</strong></div>
              </div>
                            <p><span>Top Agent Paths:</span> {path_breakdown}</p>
                            {tool_breakdown}
              <ul>{span_mix}</ul>
            </section>
            """.format(
                model=escape(str(model_key)),
                trace_cases=escape(_format_integer(trace_info.get("trace_count"))),
                avg_spans=escape(_format_decimal(trace_info.get("avg_spans_per_trace"), 1)),
                failed_cases=escape(_format_integer(trace_info.get("trace_cases_with_failures"))),
                failed_spans=escape(_format_integer(trace_info.get("failed_spans"))),
                partial_spans=escape(_format_integer(trace_info.get("partial_spans"))),
                tool_failures=escape(_format_integer(trace_info.get("tool_failures"))),
                path_breakdown=escape(_format_count_breakdown(trace_info.get("path_signatures"), limit=2)),
                tool_breakdown=(
                    "<p><span>Top Tool Failures:</span> {names}</p><p><span>Tool Error Types:</span> {errors}</p>".format(
                        names=escape(_format_count_breakdown(trace_info.get("tool_failure_names"))),
                        errors=escape(_format_count_breakdown(trace_info.get("tool_failure_error_types"))),
                    )
                    if trace_info.get("tool_failures")
                    else ""
                ),
                span_mix=span_mix,
            )
        )

    metadata_table = "".join(
        "<tr><th>{label}</th><td>{value}</td></tr>".format(
            label=escape(str(label)),
            value=escape(str(value)),
        )
        for label, value in metadata_rows
    )

    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Evaluation Summary</title>
    <style>
      :root {{ color-scheme: light; --bg: #f6f1e8; --panel: #fffaf2; --ink: #2f2418; --muted: #6b5b49; --border: #d6c4ad; --accent: #9c6b2f; --danger: #9b3d2e; }}
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; font-family: Georgia, "Times New Roman", serif; background: linear-gradient(180deg, #f8f3ea 0%, #efe4d3 100%); color: var(--ink); }}
      main {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 48px; }}
      h1, h2, h3, p {{ margin: 0; }}
      h1 {{ font-size: 2.4rem; margin-bottom: 8px; }}
      h2 {{ font-size: 1.1rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; }}
      h3 {{ font-size: 1.1rem; margin-bottom: 14px; }}
      section {{ margin-top: 28px; }}
      .lede {{ color: var(--muted); max-width: 70ch; line-height: 1.6; }}
      .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
      .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 18px; box-shadow: 0 10px 30px rgba(65, 42, 13, 0.08); }}
      table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); border-radius: 18px; overflow: hidden; }}
      th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }}
      th {{ width: 240px; color: var(--muted); font-weight: 600; }}
      tr:last-child th, tr:last-child td {{ border-bottom: 0; }}
      .metrics-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .metrics-grid span {{ display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 4px; }}
      .metrics-grid strong {{ font-size: 1.05rem; }}
      ul {{ margin: 0; padding-left: 18px; }}
      .case-list {{ list-style: none; padding: 0; display: grid; gap: 12px; }}
      .case-item {{ border: 1px solid var(--border); border-radius: 14px; padding: 12px; background: rgba(255,255,255,0.55); }}
      .case-item.failed {{ border-color: rgba(155, 61, 46, 0.35); background: rgba(155, 61, 46, 0.06); }}
      .case-head {{ display: flex; justify-content: space-between; gap: 16px; margin-bottom: 8px; }}
      .case-item span {{ color: var(--muted); font-size: 0.88rem; }}
      .case-item p {{ margin-top: 8px; line-height: 1.55; }}
      @media (max-width: 640px) {{ .metrics-grid {{ grid-template-columns: 1fr; }} .case-head {{ flex-direction: column; gap: 6px; }} th {{ width: auto; }} }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>Evaluation Summary</h1>
        <p class="lede">Unified file report generated from the shared reporting contract across console, markdown and HTML outputs.</p>
      </header>

      <section>
        <h2>Run Metadata</h2>
        <table>
          <tbody>{metadata_table}</tbody>
        </table>
      </section>

      <section>
        <h2>Model Comparison</h2>
        <div class="grid">{model_cards}</div>
      </section>

      <section>
        <h2>Best Performers</h2>
        <div class="card"><ul>{best_performer_items}</ul></div>
      </section>

    {policy_section}
    {policy_audit_section}

            {trace_panels}
      {case_panels}
            {trace_details}
      {trends}
    </main>
  </body>
</html>
""".format(
        metadata_table=metadata_table,
        model_cards="".join(model_cards) or '<div class="card"><p>No model comparison available.</p></div>',
        best_performer_items=best_performer_items,
        policy_section=(
            '<section><h2>Policy Summary</h2><div class="grid">'
            '<article class="card"><div class="metrics-grid">'
            '<div><span>Policy Cases</span><strong>{total}</strong></div>'
            '<div><span>Flagged Cases</span><strong>{flagged}</strong></div>'
            '<div><span>High Severity</span><strong>{high}</strong></div>'
            '<div><span>Queue Candidates</span><strong>{queue}</strong></div>'
            '<div><span>Avg Severity</span><strong>{avg}</strong></div>'
            '<div><span>Risk Levels</span><strong>{risk_levels}</strong></div>'
            '</div></article>{policy_cards}{top_cases}</div></section>'.format(
                total=escape(_format_integer(policy_summary.get("total_policy_cases"))),
                flagged=escape(_format_integer(policy_summary.get("flagged_case_count"))),
                high=escape(_format_integer(policy_summary.get("high_severity_case_count"))),
                queue=escape(_format_integer(policy_summary.get("queue_candidate_count"))),
                avg=escape(_format_decimal(policy_summary.get("avg_severity"))),
                risk_levels=escape(_format_count_breakdown(policy_summary.get("risk_level_counts"))),
                policy_cards="".join(policy_cards),
                top_cases=(
                    '<article class="card"><h3>Top Policy Cases</h3><ul class="case-list">{items}</ul></article>'.format(
                        items="".join(policy_case_items) or '<li class="case-item review">No policy queue candidates available.</li>'
                    )
                ),
            )
            if int(policy_summary.get("total_policy_cases") or 0) > 0
            else ""
        ),
        policy_audit_section=(
            '<section><h2>Policy Review Audit</h2><div class="grid">'
            '<article class="card"><div class="metrics-grid">'
            '<div><span>Total Reviews</span><strong>{total}</strong></div>'
            '<div><span>Confirmed Violations</span><strong>{confirmed}</strong></div>'
            '<div><span>False Positives</span><strong>{false_positive}</strong></div>'
            '<div><span>Needs Follow-Up</span><strong>{follow_up}</strong></div>'
            '<div><span>Latest Review</span><strong>{latest}</strong></div>'
            '</div></article><article class="card"><h3>Recent Decisions</h3><ul class="case-list">{items}</ul></article></div></section>'.format(
                total=escape(_format_integer(policy_audit_summary.get("total_reviews"))),
                confirmed=escape(_format_integer(policy_audit_summary.get("confirmed_violation_count"))),
                false_positive=escape(_format_integer(policy_audit_summary.get("false_positive_count"))),
                follow_up=escape(_format_integer(policy_audit_summary.get("needs_follow_up_count"))),
                latest=escape(str(policy_audit_summary.get("latest_review_at") or "n/a")),
                items="".join(policy_audit_items) or '<li class="case-item review">No policy review decisions available.</li>',
            )
            if int(policy_audit_summary.get("total_reviews") or 0) > 0
            else ""
        ),
        trace_panels=(
            '<section><h2>Trace Coverage</h2><div class="grid">{}</div></section>'.format("".join(trace_sections))
            if trace_sections
            else ""
        ),
        case_panels=(
            '<section><h2>Fail-First Case Panels</h2><div class="grid">{}</div></section>'.format("".join(case_sections))
            if case_sections
            else ""
        ),
        trace_details="".join(trace_detail_sections),
        trends=(
            '<section><h2>Trends</h2><div class="grid">{}</div></section>'.format("".join(trend_sections))
            if trend_sections
            else ""
        ),
    )