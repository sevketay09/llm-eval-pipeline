"""
Human Feedback Evaluator
Uses human annotations to evaluate models or calibrate LLM-as-Judge
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from adapters.unified_adapter import UnifiedLLMAdapter
from analysis.judge_reliability import compute_judge_reliability
from utils.human_annotations import AnnotationManager, HumanAnnotation
import statistics


def _feedback_excerpt(value: str, limit: int = 160) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


_RUBRIC_KEYWORDS = (
    "rubric",
    "criteria",
    "criterion",
    "guideline",
    "policy",
    "instruction",
    "constraint",
    "format",
    "schema",
    "missing",
    "incomplete",
    "incorrect",
)


def _taxonomy_label_to_title(label: str) -> str:
    return label.replace("_", " ")


def _build_disagreement_taxonomy(annotation: HumanAnnotation) -> Dict[str, Any]:
    score_diff = abs(annotation.llm_judge_score - annotation.human_score)
    direction = "llm_too_high" if annotation.llm_judge_score > annotation.human_score else "llm_too_low"
    verdict = annotation.verdict or {}
    human_feedback = (annotation.human_feedback or "").strip()
    feedback_lower = human_feedback.casefold()

    tags: List[str] = [direction]
    primary_reason = "judge_over_scoring" if direction == "llm_too_high" else "judge_under_scoring"

    if score_diff >= 0.75:
        severity = "critical"
        tags.append("extreme_split")
    elif score_diff >= 0.45:
        severity = "high"
        tags.append("strong_split")
    else:
        severity = "medium"
        tags.append("moderate_split")

    if annotation.correction_type == "reject":
        tags.append("human_reject")
        if direction == "llm_too_high":
            primary_reason = "judge_missed_rejection"
    elif annotation.correction_type == "approve":
        tags.append("human_approve")
        if direction == "llm_too_low":
            primary_reason = "judge_missed_acceptance"
    else:
        tags.append("human_adjust")
        if score_diff >= 0.45:
            primary_reason = "rubric_boundary_mismatch"

    if min(abs(annotation.human_score - 0.5), abs(annotation.llm_judge_score - 0.5)) <= 0.1:
        tags.append("boundary_case")
        if primary_reason in {"judge_over_scoring", "judge_under_scoring"}:
            primary_reason = "rubric_boundary_mismatch"

    if isinstance(verdict.get("requires_follow_up"), bool) and verdict["requires_follow_up"]:
        tags.append("requires_follow_up")
        if primary_reason in {"judge_over_scoring", "judge_under_scoring"}:
            primary_reason = "follow_up_signal_gap"

    if human_feedback and any(keyword in feedback_lower for keyword in _RUBRIC_KEYWORDS):
        tags.append("rubric_feedback")
        if primary_reason in {
            "judge_over_scoring",
            "judge_under_scoring",
            "judge_missed_acceptance",
            "judge_missed_rejection",
        }:
            primary_reason = "rubric_signal_gap"

    return {
        "primary_reason": primary_reason,
        "severity": severity,
        "direction": direction,
        "tags": tags,
        "title": _taxonomy_label_to_title(primary_reason),
    }


def _summarize_disagreement_taxonomy(disagreements: List[Dict[str, Any]]) -> Dict[str, Any]:
    reason_counts = Counter()
    severity_counts = Counter()
    direction_counts = Counter()
    tag_counts = Counter()

    for case in disagreements:
        taxonomy = case.get("reason_taxonomy") or {}
        primary_reason = taxonomy.get("primary_reason")
        severity = taxonomy.get("severity")
        direction = taxonomy.get("direction")
        tags = taxonomy.get("tags") or []

        if primary_reason:
            reason_counts[str(primary_reason)] += 1
        if severity:
            severity_counts[str(severity)] += 1
        if direction:
            direction_counts[str(direction)] += 1
        for tag in tags:
            tag_counts[str(tag)] += 1

    return {
        "reason_counts": dict(reason_counts),
        "severity_counts": dict(severity_counts),
        "direction_counts": dict(direction_counts),
        "tag_counts": dict(tag_counts),
        "top_reasons": [
            {
                "reason": reason,
                "label": _taxonomy_label_to_title(reason),
                "count": count,
            }
            for reason, count in reason_counts.most_common(5)
        ],
    }


class HumanFeedbackEvaluator:
    """
    Evaluator that uses human annotations.
    
    Can be used to:
    1. Compare model performance with human-validated scores
    2. Measure LLM-Judge accuracy vs. human judgments
    3. Calibrate LLM-Judge using human feedback
    """
    
    def __init__(self, annotation_manager: AnnotationManager = None, reports_dir: str = "reports"):
        self.annotation_manager = annotation_manager or AnnotationManager()
        self.reports_dir = Path(reports_dir)
        self._prompt_version_cache: Dict[str, Optional[str]] = {}

    def _resolve_prompt_version_from_source_report(self, source_report: str) -> Optional[str]:
        report_ref = str(source_report or "").strip()
        if not report_ref:
            return None
        if report_ref in self._prompt_version_cache:
            return self._prompt_version_cache[report_ref]

        report_name, _, run_id = report_ref.partition("::")
        report_path = self.reports_dir / report_name
        prompt_version: Optional[str] = None

        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
                runs = payload.get("runs") or []
                if run_id:
                    run_payload = next(
                        (
                            run for run in runs
                            if (run.get("run_metadata") or {}).get("run_id") == run_id
                        ),
                        None,
                    )
                else:
                    run_payload = runs[-1] if runs else None
                if isinstance(run_payload, dict):
                    prompt_version = (run_payload.get("run_metadata") or {}).get("judge_prompt_version")
            elif isinstance(payload, dict):
                prompt_version = (payload.get("run_metadata") or {}).get("judge_prompt_version")
        except Exception:
            prompt_version = None

        self._prompt_version_cache[report_ref] = prompt_version
        return prompt_version

    def _resolve_annotation_prompt_version(self, annotation: HumanAnnotation) -> str:
        metadata = annotation.metadata or {}
        prompt_version = metadata.get("judge_prompt_version") or metadata.get("prompt_version")
        if isinstance(prompt_version, str) and prompt_version.strip():
            return prompt_version.strip()

        source_report = metadata.get("source_report")
        resolved = self._resolve_prompt_version_from_source_report(str(source_report or ""))
        if isinstance(resolved, str) and resolved.strip():
            return resolved.strip()

        return "unknown"

    def summarize_prompt_versions(
        self,
        annotations: Optional[List[HumanAnnotation]] = None,
        test_category: str = None,
    ) -> Dict[str, Any]:
        """Compare judge calibration quality across prompt versions."""
        review_set = annotations if annotations is not None else self.annotation_manager.load_all_annotations(
            status="completed",
            test_category=test_category,
        )
        if not review_set:
            return {
                "versions": [],
                "known_version_count": 0,
                "unknown_count": 0,
                "best_agreement_version": None,
                "lowest_mae_version": None,
            }

        grouped: Dict[str, List[HumanAnnotation]] = defaultdict(list)
        for annotation in review_set:
            grouped[self._resolve_annotation_prompt_version(annotation)].append(annotation)

        version_rows = []
        for version, items in grouped.items():
            agreements = [1 - abs(item.llm_judge_score - item.human_score) for item in items]
            absolute_errors = [abs(item.llm_judge_score - item.human_score) for item in items]
            score_diffs = [item.llm_judge_score - item.human_score for item in items]
            version_rows.append({
                "prompt_version": version,
                "reviewed_cases": len(items),
                "average_agreement": statistics.mean(agreements) if agreements else 0.0,
                "mean_absolute_error": statistics.mean(absolute_errors) if absolute_errors else 0.0,
                "judge_bias": statistics.mean(score_diffs) if score_diffs else 0.0,
            })

        version_rows.sort(
            key=lambda row: (
                row["prompt_version"] == "unknown",
                -row["average_agreement"],
                row["mean_absolute_error"],
                -row["reviewed_cases"],
                row["prompt_version"],
            )
        )

        known_versions = [row for row in version_rows if row["prompt_version"] != "unknown"]
        best_agreement = max(
            version_rows,
            key=lambda row: (row["average_agreement"], row["reviewed_cases"]),
            default=None,
        )
        lowest_mae = min(
            version_rows,
            key=lambda row: (row["mean_absolute_error"], -row["reviewed_cases"]),
            default=None,
        )

        return {
            "versions": version_rows,
            "known_version_count": len(known_versions),
            "unknown_count": sum(1 for row in version_rows if row["prompt_version"] == "unknown"),
            "best_agreement_version": best_agreement["prompt_version"] if best_agreement else None,
            "lowest_mae_version": lowest_mae["prompt_version"] if lowest_mae else None,
        }
    
    def evaluate_model_with_human_feedback(
        self,
        model_name: str,
        test_category: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model using human-validated scores.
        
        Args:
            model_name: Model to evaluate
            test_category: Optional test category filter
        
        Returns:
            Evaluation metrics based on human scores
        """
        # Load annotations for this model
        annotations = self.annotation_manager.load_all_annotations(
            status="completed",
            model_name=model_name,
            test_category=test_category
        )
        
        if not annotations:
            return {
                "error": f"No human annotations found for model {model_name}",
                "model_name": model_name,
                "test_category": test_category
            }
        
        # Calculate metrics
        human_scores = [ann.human_score for ann in annotations]
        
        # Breakdown by correction type
        approved = [ann for ann in annotations if ann.correction_type == "approve"]
        adjusted = [ann for ann in annotations if ann.correction_type == "adjust"]
        rejected = [ann for ann in annotations if ann.correction_type == "reject"]
        
        approval_rate = len(approved) / len(annotations) if annotations else 0
        adjustment_rate = len(adjusted) / len(annotations) if annotations else 0
        rejection_rate = len(rejected) / len(annotations) if annotations else 0
        
        return {
            "model_name": model_name,
            "test_category": test_category,
            "total_annotations": len(annotations),
            "human_validated_score": statistics.mean(human_scores) if human_scores else 0,
            "score_std": statistics.stdev(human_scores) if len(human_scores) > 1 else 0,
            "score_range": {
                "min": min(human_scores) if human_scores else 0,
                "max": max(human_scores) if human_scores else 0
            },
            "correction_breakdown": {
                "approved": len(approved),
                "adjusted": len(adjusted),
                "rejected": len(rejected),
                "approval_rate": approval_rate,
                "adjustment_rate": adjustment_rate,
                "rejection_rate": rejection_rate
            },
            "annotations": [
                {
                    "test_id": ann.test_id,
                    "question": ann.question[:100] + "..." if len(ann.question) > 100 else ann.question,
                    "human_score": ann.human_score,
                    "correction_type": ann.correction_type
                }
                for ann in annotations
            ]
        }
    
    def evaluate_judge_accuracy(
        self,
        test_category: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate LLM-as-Judge accuracy compared to human judgments.
        
        Args:
            test_category: Optional test category filter
        
        Returns:
            Judge accuracy metrics
        """
        annotations = self.annotation_manager.load_all_annotations(
            status="completed",
            test_category=test_category
        )
        
        if not annotations:
            return {
                "error": "No annotations found",
                "test_category": test_category
            }
        
        # Calculate agreement metrics
        agreements = []
        absolute_errors = []
        
        for ann in annotations:
            agreement = 1 - abs(ann.llm_judge_score - ann.human_score)
            agreements.append(agreement)
            absolute_errors.append(abs(ann.llm_judge_score - ann.human_score))
        
        # Breakdown by correction type
        by_correction_type = {
            "approve": [],
            "adjust": [],
            "reject": []
        }
        
        for ann in annotations:
            error = abs(ann.llm_judge_score - ann.human_score)
            by_correction_type[ann.correction_type].append(error)
        
        # Calculate bias (LLM-Judge tendency to over/under score)
        score_diffs = [ann.llm_judge_score - ann.human_score for ann in annotations]
        bias = statistics.mean(score_diffs) if score_diffs else 0
        
        return {
            "test_category": test_category,
            "total_comparisons": len(annotations),
            "average_agreement": statistics.mean(agreements) if agreements else 0,
            "mean_absolute_error": statistics.mean(absolute_errors) if absolute_errors else 0,
            "median_absolute_error": statistics.median(absolute_errors) if absolute_errors else 0,
            "judge_bias": bias,
            "bias_interpretation": "LLM-Judge scores too high" if bias > 0.1 else "LLM-Judge scores too low" if bias < -0.1 else "LLM-Judge is well-calibrated",
            "error_by_correction_type": {
                correction_type: {
                    "count": len(errors),
                    "mean_error": statistics.mean(errors) if errors else 0
                }
                for correction_type, errors in by_correction_type.items()
            },
            "high_disagreement_cases": [
                {
                    "test_id": ann.test_id,
                    "test_category": ann.test_category,
                    "llm_score": ann.llm_judge_score,
                    "human_score": ann.human_score,
                    "difference": abs(ann.llm_judge_score - ann.human_score),
                    "human_feedback": ann.human_feedback[:200] + "..." if len(ann.human_feedback) > 200 else ann.human_feedback
                }
                for ann in annotations
                if abs(ann.llm_judge_score - ann.human_score) > 0.3
            ][:10]  # Top 10 disagreements
        }
    
    def get_calibration_insights(self) -> Dict[str, Any]:
        """
        Get insights for calibrating LLM-as-Judge.
        
        Returns:
            Calibration recommendations
        """
        judge_accuracy = self.evaluate_judge_accuracy()
        
        if "error" in judge_accuracy:
            return judge_accuracy
        
        recommendations = []
        
        # Check bias
        bias = judge_accuracy.get("judge_bias", 0)
        if abs(bias) > 0.1:
            if bias > 0:
                recommendations.append({
                    "issue": "Judge scores consistently too high",
                    "recommendation": "Consider adding stricter evaluation criteria or adjusting the system prompt to be more critical"
                })
            else:
                recommendations.append({
                    "issue": "Judge scores consistently too low",
                    "recommendation": "Consider relaxing evaluation criteria or adjusting the system prompt to be more generous"
                })
        
        # Check error rate
        mae = judge_accuracy.get("mean_absolute_error", 0)
        if mae > 0.2:
            recommendations.append({
                "issue": f"High mean absolute error ({mae:.2f})",
                "recommendation": "Consider fine-tuning the judge model with human annotations or providing more detailed evaluation guidelines"
            })
        
        # Check high disagreement cases
        high_disagreements = judge_accuracy.get("high_disagreement_cases", [])
        if len(high_disagreements) > 5:
            categories = {}
            for case in high_disagreements:
                cat = case['test_category']
                if cat not in categories:
                    categories[cat] = 0
                categories[cat] += 1
            
            problematic_cats = [cat for cat, count in categories.items() if count >= 3]
            if problematic_cats:
                recommendations.append({
                    "issue": f"High disagreement in categories: {', '.join(problematic_cats)}",
                    "recommendation": f"Review evaluation criteria for these categories and provide category-specific examples"
                })
        
        annotations = self.annotation_manager.load_all_annotations(status="completed")
        disagreements = self.get_disagreement_cases(threshold=0.3)
        prompt_version_comparison = self.summarize_prompt_versions(annotations=annotations)

        reliability = compute_judge_reliability([
            (ann.llm_judge_score, ann.human_score)
            for ann in annotations
            if isinstance(ann.llm_judge_score, (int, float))
            and isinstance(ann.human_score, (int, float))
        ])
        if reliability["verdict"] == "needs_calibration" and reliability["n"] >= 5:
            recommendations.append({
                "issue": f"Low chance-corrected agreement (kappa={reliability['cohens_kappa']}, {reliability['kappa_interpretation']})",
                "recommendation": "Judge and human raters disagree beyond chance level; review judge prompts against annotated examples"
            })

        return {
            "overall_metrics": {
                "average_agreement": judge_accuracy.get("average_agreement", 0),
                "mean_absolute_error": mae,
                "judge_bias": bias,
                "spearman_rho": reliability["spearman_rho"],
                "cohens_kappa": reliability["cohens_kappa"],
                "kappa_interpretation": reliability["kappa_interpretation"],
                "reliability_verdict": reliability["verdict"],
                "reliability_n": reliability["n"],
            },
            "recommendations": recommendations,
            "disagreement_taxonomy": _summarize_disagreement_taxonomy(disagreements),
            "prompt_version_comparison": prompt_version_comparison,
            "training_data_available": judge_accuracy.get("total_comparisons", 0),
            "ready_for_finetuning": judge_accuracy.get("total_comparisons", 0) >= 50
        }

    def compute_inter_rater_reliability(self, tolerance: float = 0.15) -> Dict[str, Any]:
        """
        Measure agreement between distinct human reviewers on the same case.

        Only meaningful when a case has been annotated by 2+ distinct annotator_ids —
        single-reviewer workflows are a fully supported usage mode, so this returns
        `applicable: False` rather than an error when no such overlap exists.
        """
        annotations = self.annotation_manager.load_all_annotations(status="completed")

        grouped: Dict[tuple, List[HumanAnnotation]] = defaultdict(list)
        for ann in annotations:
            grouped[(ann.test_id, ann.model_name)].append(ann)

        multi_reviewer_cases = []
        for (test_id, model_name), items in grouped.items():
            by_annotator: Dict[str, HumanAnnotation] = {}
            for item in sorted(items, key=lambda a: a.timestamp):
                by_annotator[item.annotator_id] = item
            if len(by_annotator) < 2:
                continue

            scores = [ann.human_score for ann in by_annotator.values()]
            pairwise_diffs = [
                abs(a - b)
                for i, a in enumerate(scores)
                for b in scores[i + 1:]
            ]
            multi_reviewer_cases.append({
                "test_id": test_id,
                "model_name": model_name,
                "reviewer_count": len(scores),
                "mean_pairwise_absolute_difference": statistics.mean(pairwise_diffs),
            })

        if not multi_reviewer_cases:
            return {
                "applicable": False,
                "multi_reviewer_case_count": 0,
                "total_reviewed_cases": len(grouped),
            }

        all_diffs = [case["mean_pairwise_absolute_difference"] for case in multi_reviewer_cases]
        within_tolerance = sum(1 for diff in all_diffs if diff <= tolerance)

        return {
            "applicable": True,
            "multi_reviewer_case_count": len(multi_reviewer_cases),
            "total_reviewed_cases": len(grouped),
            "average_agreement": statistics.mean(1 - diff for diff in all_diffs),
            "mean_pairwise_absolute_difference": statistics.mean(all_diffs),
            "within_tolerance_rate": within_tolerance / len(multi_reviewer_cases),
            "tolerance": tolerance,
            "distinct_annotators": len({ann.annotator_id for ann in annotations}),
        }

    def get_disagreement_cases(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Collect cases where human and LLM-judge scores diverge beyond a threshold."""
        annotations = self.annotation_manager.load_all_annotations(status="completed")

        disagreements = []
        for ann in annotations:
            score_diff = abs(ann.llm_judge_score - ann.human_score)
            if score_diff >= threshold:
                taxonomy = _build_disagreement_taxonomy(ann)
                disagreements.append({
                    "test_id": ann.test_id,
                    "test_category": ann.test_category,
                    "model_name": ann.model_name,
                    "question": ann.question,
                    "model_response": ann.model_response,
                    "llm_judge": {
                        "score": ann.llm_judge_score,
                        "reasoning": ann.llm_judge_reasoning
                    },
                    "human": {
                        "score": ann.human_score,
                        "feedback": ann.human_feedback,
                        "correction_type": ann.correction_type,
                        "verdict": ann.verdict,
                    },
                    "score_difference": score_diff,
                    "direction": "LLM too high" if ann.llm_judge_score > ann.human_score else "LLM too low",
                    "reason_taxonomy": taxonomy,
                    "timestamp": ann.timestamp
                })

        return disagreements

    def summarize_disagreement_taxonomy(
        self,
        disagreements: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Summarize disagreement reasons for reviewed annotations."""
        cases = disagreements if disagreements is not None else self.get_disagreement_cases(threshold=threshold)
        return _summarize_disagreement_taxonomy(cases)

    def build_calibration_sample_set(self, sample_size: int = 12) -> Dict[str, Any]:
        """Build a balanced reviewed sample set for judge calibration work."""
        annotations = self.annotation_manager.load_all_annotations(status="completed")
        if not annotations:
            return {
                "total_samples": 0,
                "bucket_counts": {},
                "samples": [],
            }

        target_size = max(3, sample_size)
        disagreements = sorted(
            annotations,
            key=lambda ann: abs(ann.llm_judge_score - ann.human_score),
            reverse=True,
        )
        anchors = sorted(
            annotations,
            key=lambda ann: (
                abs(ann.llm_judge_score - ann.human_score),
                -max(ann.human_score, ann.llm_judge_score),
            ),
        )
        boundary = sorted(
            annotations,
            key=lambda ann: min(abs(ann.human_score - 0.5), abs(ann.llm_judge_score - 0.5)),
        )

        selected_ids = set()
        samples: List[Dict[str, Any]] = []

        def append_from_bucket(bucket: str, items: List[HumanAnnotation], bucket_size: int) -> None:
            for ann in items:
                if len(samples) >= target_size:
                    return
                if ann.annotation_id in selected_ids:
                    continue
                samples.append({
                    "bucket": bucket,
                    "test_id": ann.test_id,
                    "test_category": ann.test_category,
                    "model_name": ann.model_name,
                    "llm_score": ann.llm_judge_score,
                    "human_score": ann.human_score,
                    "score_difference": abs(ann.llm_judge_score - ann.human_score),
                    "correction_type": ann.correction_type,
                    "question": _feedback_excerpt(ann.question, limit=120),
                    "human_feedback": _feedback_excerpt(ann.human_feedback),
                })
                selected_ids.add(ann.annotation_id)
                if sum(1 for item in samples if item["bucket"] == bucket) >= bucket_size:
                    return

        per_bucket = max(1, target_size // 3)
        append_from_bucket("high_disagreement", disagreements, per_bucket)
        append_from_bucket("boundary_case", boundary, per_bucket)
        append_from_bucket("agreement_anchor", anchors, per_bucket)

        for ann in disagreements:
            if len(samples) >= target_size:
                break
            if ann.annotation_id in selected_ids:
                continue
            samples.append({
                "bucket": "overflow",
                "test_id": ann.test_id,
                "test_category": ann.test_category,
                "model_name": ann.model_name,
                "llm_score": ann.llm_judge_score,
                "human_score": ann.human_score,
                "score_difference": abs(ann.llm_judge_score - ann.human_score),
                "correction_type": ann.correction_type,
                "question": _feedback_excerpt(ann.question, limit=120),
                "human_feedback": _feedback_excerpt(ann.human_feedback),
            })
            selected_ids.add(ann.annotation_id)

        bucket_counts: Dict[str, int] = {}
        for sample in samples:
            bucket = sample["bucket"]
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        return {
            "total_samples": len(samples),
            "bucket_counts": bucket_counts,
            "samples": samples,
        }
    
    def export_disagreement_cases(
        self,
        threshold: float = 0.3,
        output_file: str = "disagreement_cases.json"
    ) -> str:
        """
        Export cases where LLM-Judge and humans disagree significantly.
        Useful for analysis and improving judge prompts.
        
        Args:
            threshold: Minimum score difference to consider as disagreement
            output_file: Output filename
        
        Returns:
            Path to exported file
        """
        import json
        from pathlib import Path
        
        disagreements = self.get_disagreement_cases(threshold=threshold)
        
        output_path = Path("annotations") / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_disagreements": len(disagreements),
                "threshold": threshold,
                "reason_taxonomy": self.summarize_disagreement_taxonomy(disagreements=disagreements),
                "cases": disagreements
            }, f, ensure_ascii=False, indent=2)
        
        return str(output_path)


def evaluate_human_feedback(
    model_name: str,
    test_category: str = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate model with human feedback.
    
    Args:
        model_name: Model to evaluate
        test_category: Optional test category filter
    
    Returns:
        Evaluation results
    """
    evaluator = HumanFeedbackEvaluator()
    return evaluator.evaluate_model_with_human_feedback(model_name, test_category)


def evaluate_judge_with_human_feedback() -> Dict[str, Any]:
    """
    Convenience function to evaluate LLM-as-Judge accuracy.
    
    Returns:
        Judge accuracy metrics
    """
    evaluator = HumanFeedbackEvaluator()
    return evaluator.evaluate_judge_accuracy()
