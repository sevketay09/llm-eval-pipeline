"""Service for loading and querying evaluation reports."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from api.config import get_settings
from api.schemas.evaluations import ReportListItem, ReportSummary
from utils.human_annotations import HIGH_DISAGREEMENT_THRESHOLD, _extract_judge_signal
from utils.policy_reporting import build_policy_audit_summary_from_results, build_policy_summary_from_models
from utils.report_renderer import render_html_report, render_markdown_report


class ReportService:
    """Read-only access to evaluation reports."""

    def __init__(self):
        settings = get_settings()
        self._dir = Path(settings.reports_dir)

    def _count_total_items(self, model_info: dict[str, Any]) -> int:
        tests = model_info.get("tests", {})
        if not isinstance(tests, dict):
            return 0

        total_items = 0
        for test_result in tests.values():
            if not isinstance(test_result, dict):
                continue
            summary = test_result.get("summary", {})
            if not isinstance(summary, dict):
                continue
            total_tests = summary.get("total_tests")
            if isinstance(total_tests, (int, float)):
                total_items += int(total_tests)

        return total_items

    def _normalize_version_metadata(self, metadata: dict[str, Any], report_data: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(metadata)
        prompt_version = normalized.get("prompt_version") or normalized.get("judge_prompt_version")
        normalized["prompt_version"] = prompt_version or "legacy"
        normalized.setdefault("judge_prompt_version", normalized["prompt_version"])

        schema_version = normalized.get("schema_version") or report_data.get("version")
        normalized["schema_version"] = schema_version or "legacy"
        normalized.setdefault("metric_version", "legacy")

        metric_pack_versions = normalized.get("metric_pack_versions")
        if not isinstance(metric_pack_versions, dict):
            normalized["metric_pack_versions"] = {}

        return normalized

    def _extract_metric_observed_cost(self, metric_result: Mapping[str, Any]) -> float | None:
        raw_payload = metric_result.get("raw_payload")
        metadata = metric_result.get("metadata")
        payloads = []
        if isinstance(raw_payload, Mapping):
            payloads.append(raw_payload)
        if isinstance(metadata, Mapping):
            payloads.append(metadata)

        for payload in payloads:
            for key in ("total_cost", "cost", "cost_usd", "usd_cost"):
                value = payload.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    def _extract_metric_observed_tokens(self, metric_result: Mapping[str, Any]) -> int:
        raw_payload = metric_result.get("raw_payload")
        metadata = metric_result.get("metadata")
        payloads = []
        if isinstance(raw_payload, Mapping):
            payloads.append(raw_payload)
        if isinstance(metadata, Mapping):
            payloads.append(metadata)

        for payload in payloads:
            usage = payload.get("usage")
            if isinstance(usage, Mapping):
                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
                if isinstance(input_tokens, (int, float)) or isinstance(output_tokens, (int, float)):
                    return int(input_tokens or 0) + int(output_tokens or 0)

            direct_tokens = payload.get("tokens")
            if isinstance(direct_tokens, (int, float)):
                return int(direct_tokens)

            input_tokens = payload.get("input_tokens", payload.get("prompt_tokens", 0))
            output_tokens = payload.get("output_tokens", payload.get("completion_tokens", 0))
            if isinstance(input_tokens, (int, float)) or isinstance(output_tokens, (int, float)):
                return int(input_tokens or 0) + int(output_tokens or 0)

        return 0

    def _build_evaluator_efficiency_breakdown(
        self,
        models_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        providers: dict[str, dict[str, Any]] = {}
        total_metrics = 0

        for model_key, model_info in models_data.items():
            if not isinstance(model_info, dict):
                continue

            tests = model_info.get("tests", {})
            if not isinstance(tests, dict):
                continue

            for test_name, test_info in tests.items():
                if not isinstance(test_info, dict):
                    continue

                results = test_info.get("results", [])
                if not isinstance(results, list):
                    continue

                for index, result in enumerate(results):
                    if not isinstance(result, dict):
                        continue

                    case_id = str(result.get("id") or result.get("test_id") or result.get("case_id") or index)
                    metric_results = result.get("metric_results", [])
                    if not isinstance(metric_results, list):
                        continue

                    for metric_result in metric_results:
                        if not isinstance(metric_result, Mapping):
                            continue

                        provider = metric_result.get("provider")
                        if not isinstance(provider, str) or not provider.strip():
                            provider = "unknown"

                        bucket = providers.setdefault(
                            provider,
                            {
                                "provider": provider,
                                "metric_count": 0,
                                "cases": set(),
                                "models": set(),
                                "score_total": 0.0,
                                "score_count": 0,
                                "success_total": 0,
                                "success_count": 0,
                                "observed_cost": 0.0,
                                "observed_cost_count": 0,
                                "observed_tokens": 0,
                            },
                        )

                        total_metrics += 1
                        bucket["metric_count"] += 1
                        bucket["cases"].add(f"{model_key}:{test_name}:{case_id}")
                        bucket["models"].add(model_key)

                        score = metric_result.get("normalized_value", metric_result.get("value"))
                        if isinstance(score, (int, float)):
                            bucket["score_total"] += float(score)
                            bucket["score_count"] += 1

                        success = metric_result.get("success")
                        if isinstance(success, bool):
                            bucket["success_total"] += 1 if success else 0
                            bucket["success_count"] += 1

                        observed_cost = self._extract_metric_observed_cost(metric_result)
                        if observed_cost is not None:
                            bucket["observed_cost"] += observed_cost
                            bucket["observed_cost_count"] += 1

                        bucket["observed_tokens"] += self._extract_metric_observed_tokens(metric_result)

        rows: list[dict[str, Any]] = []
        for bucket in providers.values():
            observed_cost = (
                bucket["observed_cost"] if bucket["observed_cost_count"] > 0 else None
            )
            observed_tokens = int(bucket["observed_tokens"])
            rows.append(
                {
                    "provider": bucket["provider"],
                    "metric_count": int(bucket["metric_count"]),
                    "case_count": len(bucket["cases"]),
                    "model_count": len(bucket["models"]),
                    "avg_score": (
                        bucket["score_total"] / bucket["score_count"]
                        if bucket["score_count"] > 0
                        else None
                    ),
                    "success_rate": (
                        bucket["success_total"] / bucket["success_count"]
                        if bucket["success_count"] > 0
                        else None
                    ),
                    "observed_cost": observed_cost,
                    "observed_tokens": observed_tokens,
                    "cost_per_1k_tokens": (
                        (observed_cost * 1000) / observed_tokens
                        if observed_cost is not None and observed_tokens > 0
                        else None
                    ),
                    "metric_share": (bucket["metric_count"] / total_metrics) if total_metrics > 0 else 0.0,
                }
            )

        rows.sort(
            key=lambda item: (
                -(item.get("observed_cost") or 0.0),
                -(item.get("metric_count") or 0),
                -(item.get("avg_score") or 0.0),
            )
        )
        return rows

    def _build_efficiency_summary(
        self,
        models_data: dict[str, Any],
        model_scores: dict[str, float],
    ) -> dict[str, Any]:
        leaderboard: list[dict[str, Any]] = []

        for model_key, model_info in models_data.items():
            if not isinstance(model_info, dict):
                continue

            overall_metrics = model_info.get("overall_metrics", {})
            if not isinstance(overall_metrics, dict):
                overall_metrics = {}

            input_tokens = int(overall_metrics.get("total_input_tokens") or 0)
            output_tokens = int(overall_metrics.get("total_output_tokens") or 0)
            total_tokens = input_tokens + output_tokens
            total_requests = int(overall_metrics.get("total_requests") or 0)
            total_items = self._count_total_items(model_info)
            overall_score = float(
                model_scores.get(model_key, overall_metrics.get("weighted_score") or 0.0)
            )

            avg_tokens_per_eval = None
            if total_items > 0 and total_tokens > 0:
                avg_tokens_per_eval = total_tokens / total_items
            elif total_requests > 0 and total_tokens > 0:
                avg_tokens_per_eval = total_tokens / total_requests

            quality_points = (overall_score * total_items) if total_items > 0 else None
            quality_per_1k_tokens = None
            if quality_points is not None and total_tokens > 0:
                quality_per_1k_tokens = (quality_points * 1000) / total_tokens

            tokens_per_quality_point = None
            if quality_points is not None and quality_points > 0:
                tokens_per_quality_point = total_tokens / quality_points

            leaderboard.append(
                {
                    "model": model_key,
                    "overall_score": overall_score,
                    "total_tokens": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_requests": total_requests,
                    "total_items": total_items,
                    "avg_tokens_per_eval": avg_tokens_per_eval,
                    "quality_points": quality_points,
                    "quality_per_1k_tokens": quality_per_1k_tokens,
                    "tokens_per_quality_point": tokens_per_quality_point,
                    "frontier": False,
                }
            )

        frontier_candidates = [
            item for item in leaderboard if isinstance(item.get("avg_tokens_per_eval"), (int, float))
        ]

        for item in frontier_candidates:
            item["frontier"] = True
            for other in frontier_candidates:
                if other["model"] == item["model"]:
                    continue

                dominates = (
                    other["overall_score"] >= item["overall_score"]
                    and other["avg_tokens_per_eval"] <= item["avg_tokens_per_eval"]
                    and (
                        other["overall_score"] > item["overall_score"]
                        or other["avg_tokens_per_eval"] < item["avg_tokens_per_eval"]
                    )
                )
                if dominates:
                    item["frontier"] = False
                    break

        leaderboard.sort(
            key=lambda item: (
                -1 if item.get("quality_per_1k_tokens") is None else 0,
                -(item.get("quality_per_1k_tokens") or 0.0),
                -(item.get("overall_score") or 0.0),
                item.get("avg_tokens_per_eval")
                if item.get("avg_tokens_per_eval") is not None
                else float("inf"),
            )
        )

        frontier_models = [item["model"] for item in leaderboard if item.get("frontier")]
        leanest_entry = min(
            (item for item in leaderboard if item.get("avg_tokens_per_eval") is not None),
            key=lambda item: item["avg_tokens_per_eval"],
            default=None,
        )
        strongest_frontier = max(
            (item for item in leaderboard if item.get("frontier")),
            key=lambda item: (
                item.get("overall_score") or 0.0,
                item.get("quality_per_1k_tokens") or 0.0,
            ),
            default=None,
        )

        return {
            "leaderboard": leaderboard,
            "frontier_models": frontier_models,
            "best_quality_yield_model": leaderboard[0]["model"] if leaderboard else None,
            "leanest_model": leanest_entry["model"] if leanest_entry else None,
            "strongest_frontier_model": strongest_frontier["model"] if strongest_frontier else None,
            "evaluator_breakdown": self._build_evaluator_efficiency_breakdown(models_data),
        }

    def _build_judge_disagreement_summary(self, models_data: dict[str, Any]) -> dict[str, Any]:
        cases: list[dict[str, Any]] = []
        by_model: dict[str, dict[str, Any]] = {}

        for model_key, model_info in models_data.items():
            if not isinstance(model_info, dict):
                continue

            tests = model_info.get("tests", {})
            if not isinstance(tests, dict):
                continue

            model_cases: list[float] = []
            high_disagreement_count = 0

            for test_name, test_info in tests.items():
                if not isinstance(test_info, dict):
                    continue

                for result in test_info.get("results", []):
                    if not isinstance(result, dict):
                        continue

                    judge_signal = _extract_judge_signal(result)
                    disagreement = judge_signal.get("judge_disagreement")
                    if not isinstance(disagreement, (int, float)):
                        continue

                    model_cases.append(float(disagreement))
                    if disagreement >= HIGH_DISAGREEMENT_THRESHOLD:
                        high_disagreement_count += 1

                    cases.append(
                        {
                            "model": model_key,
                            "test_name": test_name,
                            "test_id": str(result.get("id") or result.get("test_id") or "unknown"),
                            "question": str(
                                result.get("question")
                                or result.get("input")
                                or result.get("prompt")
                                or result.get("task")
                                or ""
                            ),
                            "llm_judge_score": judge_signal["llm_judge_score"],
                            "llm_judge_label": judge_signal["llm_judge_label"],
                            "primary_judge_score": judge_signal.get("primary_judge_score"),
                            "primary_judge_label": judge_signal.get("primary_judge_label"),
                            "secondary_judge_score": judge_signal.get("secondary_judge_score"),
                            "secondary_judge_label": judge_signal.get("secondary_judge_label"),
                            "judge_disagreement": float(disagreement),
                            "judge_agreement": judge_signal.get("judge_agreement"),
                            "review_priority": judge_signal.get("review_priority") or 0.0,
                            "queue_reason": judge_signal.get("queue_reason") or "",
                        }
                    )

            if model_cases:
                by_model[model_key] = {
                    "model": model_key,
                    "panel_case_count": len(model_cases),
                    "high_disagreement_count": high_disagreement_count,
                    "mean_disagreement": sum(model_cases) / len(model_cases),
                }

        if not cases:
            return {
                "total_panel_cases": 0,
                "high_disagreement_cases": 0,
                "mean_disagreement": None,
                "max_disagreement": None,
                "strongest_split_model": None,
                "recommended_queue_size": 0,
                "by_model": [],
                "top_cases": [],
            }

        cases.sort(
            key=lambda case: (
                -(case.get("review_priority") or 0.0),
                -(case.get("judge_disagreement") or 0.0),
                case.get("model") or "",
                case.get("test_name") or "",
            )
        )

        model_summaries = sorted(
            by_model.values(),
            key=lambda item: (
                -(item.get("mean_disagreement") or 0.0),
                -(item.get("high_disagreement_count") or 0),
                item.get("model") or "",
            ),
        )
        strongest_split_model = model_summaries[0]["model"] if model_summaries else None
        high_disagreement_cases = [case for case in cases if case["judge_disagreement"] >= HIGH_DISAGREEMENT_THRESHOLD]

        return {
            "total_panel_cases": len(cases),
            "high_disagreement_cases": len(high_disagreement_cases),
            "mean_disagreement": sum(case["judge_disagreement"] for case in cases) / len(cases),
            "max_disagreement": max(case["judge_disagreement"] for case in cases),
            "strongest_split_model": strongest_split_model,
            "recommended_queue_size": len(high_disagreement_cases),
            "by_model": model_summaries,
            "top_cases": cases[:8],
        }

    def _build_continuity_summary(self, models_data: dict[str, Any]) -> dict[str, Any]:
        by_model: list[dict[str, Any]] = []

        for model_key, model_info in models_data.items():
            if not isinstance(model_info, dict):
                continue

            tests = model_info.get("tests", {})
            if not isinstance(tests, dict):
                continue

            multi_turn = tests.get("multi_turn", {})
            if not isinstance(multi_turn, dict):
                continue

            summary = multi_turn.get("summary", {})
            if not isinstance(summary, dict):
                continue

            avg_scores = summary.get("avg_scores", {})
            unresolved_summary = summary.get("unresolved_intent_summary", {})
            if not isinstance(avg_scores, dict):
                avg_scores = {}
            if not isinstance(unresolved_summary, dict):
                unresolved_summary = {}

            intent_resolution = avg_scores.get("intent_resolution")
            unresolved_turn_rate = unresolved_summary.get("unresolved_turn_rate")
            unresolved_turns = unresolved_summary.get("unresolved_turns")
            unresolved_intent_total = unresolved_summary.get("unresolved_intent_total")

            if not any(
                isinstance(value, (int, float))
                for value in (intent_resolution, unresolved_turn_rate, unresolved_turns, unresolved_intent_total)
            ):
                continue

            by_model.append({
                "model": model_key,
                "intent_resolution": float(intent_resolution) if isinstance(intent_resolution, (int, float)) else None,
                "unresolved_turn_rate": float(unresolved_turn_rate) if isinstance(unresolved_turn_rate, (int, float)) else None,
                "unresolved_turns": int(unresolved_turns) if isinstance(unresolved_turns, (int, float)) else 0,
                "unresolved_intent_total": int(unresolved_intent_total) if isinstance(unresolved_intent_total, (int, float)) else 0,
            })

        if not by_model:
            return {
                "by_model": [],
                "best_intent_resolution_model": None,
                "highest_unresolved_rate_model": None,
            }

        best_intent_resolution = max(
            (item for item in by_model if isinstance(item.get("intent_resolution"), (int, float))),
            key=lambda item: item["intent_resolution"],
            default=None,
        )
        highest_unresolved_rate = max(
            (item for item in by_model if isinstance(item.get("unresolved_turn_rate"), (int, float))),
            key=lambda item: item["unresolved_turn_rate"],
            default=None,
        )

        return {
            "by_model": by_model,
            "best_intent_resolution_model": best_intent_resolution.get("model") if best_intent_resolution else None,
            "highest_unresolved_rate_model": highest_unresolved_rate.get("model") if highest_unresolved_rate else None,
        }

    def list_reports(self, limit: int = 50) -> list[ReportListItem]:
        if not self._dir.exists():
            return []

        files = sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        # The cumulative store is not a per-eval report (schema: {version, runs, ...});
        # exclude it so it doesn't surface in the Results list as "0 models".
        files = [f for f in files if f.name != "evaluations_store.json"]
        items = []
        for f in files[:limit]:
            stat = f.stat()
            encoded_name = quote(f.name)
            # Quick peek for metadata
            model_count, suite = None, None
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                model_count = len(data.get("models", {}))
                suite = data.get("run_metadata", {}).get("test_suite")
            except Exception:
                pass

            items.append(ReportListItem(
                filename=f.name,
                path=str(f),
                modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                size_kb=stat.st_size // 1024,
                model_count=model_count,
                suite=suite,
                export_links=ReportListItem.ExportLinks(
                    raw=f"/api/results/reports/{encoded_name}/raw",
                    markdown=f"/api/results/reports/{encoded_name}/markdown",
                    html=f"/api/results/reports/{encoded_name}/html",
                ),
            ))
        return items

    def _build_statistical_comparison(self, data: dict[str, Any]) -> dict[str, Any]:
        """Bootstrap CIs + pairwise significance over the report (fail-soft)."""
        try:
            from analysis.significance import compute_significance

            return compute_significance(data)
        except Exception:
            # Significance is an enrichment; never let it break report loading.
            return {}

    def get_report(self, filename: str) -> ReportSummary | None:
        path = self._dir / filename
        if not path.exists() or not path.is_file():
            return None

        # Prevent path traversal
        if not path.resolve().is_relative_to(self._dir.resolve()):
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = dict(data.get("run_metadata", {}) or {})
        if data.get("timestamp") and "timestamp" not in metadata:
            metadata["timestamp"] = data.get("timestamp")
        metadata = self._normalize_version_metadata(metadata, data)

        # Compute per-model average scores
        model_scores: dict[str, float] = {}
        models_data = data.get("models", {})
        for model_key, model_info in models_data.items():
            if not isinstance(model_info, dict):
                continue
            tests = model_info.get("tests", {})
            scores = []
            for test_result in tests.values():
                summary = test_result.get("summary", {})
                if "overall_score" in summary:
                    scores.append(summary["overall_score"])
            if scores:
                model_scores[model_key] = sum(scores) / len(scores)

        efficiency = self._build_efficiency_summary(models_data, model_scores)
        disagreement = self._build_judge_disagreement_summary(models_data)
        continuity = self._build_continuity_summary(models_data)
        policy = build_policy_summary_from_models(models_data)
        policy_audit = build_policy_audit_summary_from_results(data)
        statistical_comparison = self._build_statistical_comparison(data)

        return ReportSummary(
            filename=filename,
            metadata=metadata,
            models=models_data,
            model_scores=model_scores,
            model_comparison=dict((data.get("summary", {}) or {}).get("model_comparison", {}) or {}),
            trends=dict(data.get("trends", {}) or {}),
            continuity=continuity,
            efficiency=efficiency,
            disagreement=disagreement,
            policy=policy,
            policy_audit=policy_audit,
            statistical_comparison=statistical_comparison,
        )

    def get_report_raw(self, filename: str) -> dict[str, Any] | None:
        path = self._dir / filename
        if not path.exists() or not path.is_file():
            return None
        if not path.resolve().is_relative_to(self._dir.resolve()):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_report_markdown(self, filename: str) -> str | None:
        path = self._dir / filename
        if not path.exists() or not path.is_file():
            return None
        if not path.resolve().is_relative_to(self._dir.resolve()):
            return None

        markdown_path = path.with_suffix(".md")
        if markdown_path.exists() and markdown_path.is_file():
            with open(markdown_path, "r", encoding="utf-8") as f:
                return f.read()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return render_markdown_report(data)

    def get_report_html(self, filename: str) -> str | None:
        path = self._dir / filename
        if not path.exists() or not path.is_file():
            return None
        if not path.resolve().is_relative_to(self._dir.resolve()):
            return None

        html_path = path.with_suffix(".html")
        if html_path.exists() and html_path.is_file():
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return render_html_report(data)

    def compare_reports(self, filenames: list[str]) -> dict[str, Any]:
        """Compare multiple reports side-by-side."""
        reports = {}
        for fn in filenames:
            summary = self.get_report(fn)
            if summary:
                metadata = summary.metadata if isinstance(summary.metadata, dict) else {}
                reports[fn] = {
                    "model_scores": summary.model_scores,
                    "continuity": summary.continuity,
                    "model_comparison": summary.model_comparison,
                    "prompt_version": metadata.get("prompt_version"),
                    "schema_version": metadata.get("schema_version"),
                    "metric_version": metadata.get("metric_version"),
                }
        return reports
