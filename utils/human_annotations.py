"""
Human Annotation Management System
Stores and manages human feedback for LLM-as-Judge evaluations
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
from utils.logger import get_logger
from utils.trend_analysis import TrendAnalyzer

logger = get_logger(__name__)


@dataclass
class HumanAnnotation:
    """Single human annotation"""
    annotation_id: str
    test_id: str
    test_category: str
    model_name: str
    question: str
    model_response: str
    llm_judge_score: float
    llm_judge_reasoning: str
    human_score: float
    human_feedback: str
    correction_type: str  # "approve", "adjust", "reject"
    annotator_id: str
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HumanAnnotation':
        """Create from dictionary"""
        return cls(**data)


class AnnotationManager:
    """
    Manages human annotations for evaluation results.
    Stores annotations in JSONL format for easy streaming and analysis.
    """
    
    def __init__(self, annotations_dir: str = "annotations"):
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.annotations_dir / "pending").mkdir(exist_ok=True)
        (self.annotations_dir / "completed").mkdir(exist_ok=True)
        (self.annotations_dir / "training_data").mkdir(exist_ok=True)
    
    def generate_annotation_id(self, test_id: str, model_name: str) -> str:
        """Generate unique annotation ID"""
        unique_string = f"{test_id}_{model_name}_{datetime.now().isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def save_annotation(
        self,
        annotation: HumanAnnotation,
        status: str = "completed"
    ) -> None:
        """
        Save a single annotation.
        
        Args:
            annotation: HumanAnnotation object
            status: "pending" or "completed"
        """
        filepath = self.annotations_dir / status / f"{annotation.annotation_id}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(annotation.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Annotation saved: {annotation.annotation_id} (status: {status})")
    
    def save_annotation_batch(
        self,
        annotations: List[HumanAnnotation],
        batch_name: str = None
    ) -> None:
        """Save multiple annotations as JSONL"""
        if batch_name is None:
            batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        filepath = self.annotations_dir / "completed" / f"{batch_name}.jsonl"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for annotation in annotations:
                f.write(json.dumps(annotation.to_dict(), ensure_ascii=False) + '\n')
    
    def load_annotation(self, annotation_id: str, status: str = "completed") -> Optional[HumanAnnotation]:
        """Load a single annotation by ID"""
        filepath = self.annotations_dir / status / f"{annotation_id}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return HumanAnnotation.from_dict(data)
    
    def load_all_annotations(
        self,
        status: str = "completed",
        test_category: str = None,
        model_name: str = None
    ) -> List[HumanAnnotation]:
        """
        Load all annotations with optional filtering.
        
        Args:
            status: "pending" or "completed"
            test_category: Filter by test category
            model_name: Filter by model name
        
        Returns:
            List of HumanAnnotation objects
        """
        annotations = []
        
        status_dir = self.annotations_dir / status
        
        # Load from JSON files
        for filepath in status_dir.glob("*.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                annotation = HumanAnnotation.from_dict(data)
                
                # Apply filters
                if test_category and annotation.test_category != test_category:
                    continue
                if model_name and annotation.model_name != model_name:
                    continue
                
                annotations.append(annotation)
        
        # Load from JSONL files
        for filepath in status_dir.glob("*.jsonl"):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        annotation = HumanAnnotation.from_dict(data)
                        
                        # Apply filters
                        if test_category and annotation.test_category != test_category:
                            continue
                        if model_name and annotation.model_name != model_name:
                            continue
                        
                        annotations.append(annotation)
        
        return annotations
    
    def get_pending_items(
        self,
        limit: int = None,
        test_category: str = None,
        source_report: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get items pending human review.
        These are created from evaluation results.
        """
        pending_file = self.annotations_dir / "pending" / "pending_reviews.jsonl"
        
        if not pending_file.exists():
            return []
        
        items = []
        with open(pending_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    
                    # Filter
                    if test_category and item.get('test_category') != test_category:
                        continue
                    if source_report:
                        item_source = (item.get('metadata', {}) or {}).get('source_report')
                        if item_source != source_report and not str(item_source).startswith(f"{source_report}::"):
                            continue
                    
                    items.append(item)
                    
                    if limit and len(items) >= limit:
                        break
        
        return items
    
    def add_pending_item(self, item: Dict[str, Any]) -> None:
        """Add an item for pending review"""
        pending_file = self.annotations_dir / "pending" / "pending_reviews.jsonl"
        
        with open(pending_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def remove_pending_item(self, item_id: str) -> None:
        """Remove item from pending (after annotation)"""
        pending_file = self.annotations_dir / "pending" / "pending_reviews.jsonl"
        
        if not pending_file.exists():
            return
        
        # Read all items
        items = []
        with open(pending_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get('item_id') != item_id:
                        items.append(item)
        
        # Write back
        with open(pending_file, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def apply_annotation_to_report(self, annotation: HumanAnnotation, reports_dir: str = "reports") -> bool:
        """Apply human-reviewed score back to source report and recompute affected summaries."""
        metadata = annotation.metadata or {}
        source_report = metadata.get("source_report")
        if not source_report:
            return False

        report_path = Path(reports_dir) / source_report
        if not report_path.exists():
            return False

        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)

        model_block = report_data.get("models", {}).get(annotation.model_name)
        if not isinstance(model_block, dict):
            return False

        test_block = model_block.get("tests", {}).get(annotation.test_category)
        if not isinstance(test_block, dict):
            return False

        results = test_block.get("results", [])
        if not isinstance(results, list):
            return False

        target = None
        for result in results:
            if not isinstance(result, dict):
                continue
            candidate_id = result.get("id", result.get("test_id"))
            if str(candidate_id) == str(annotation.test_id):
                target = result
                break

        if target is None:
            return False

        target.setdefault("scores", {})
        if isinstance(target.get("scores"), dict):
            target["scores"]["overall"] = float(annotation.human_score)
            target["scores"]["human_override"] = float(annotation.human_score)

        target["human_review"] = {
            "human_score": float(annotation.human_score),
            "correction_type": annotation.correction_type,
            "human_feedback": annotation.human_feedback,
            "annotator_id": annotation.annotator_id,
            "timestamp": annotation.timestamp
        }

        summary = test_block.get("summary", {})
        if isinstance(summary, dict):
            effective_scores = []
            for result in results:
                extracted = _extract_llm_score(result)
                if isinstance(extracted, (int, float)):
                    effective_scores.append(float(extracted))
            if effective_scores:
                summary["overall_score"] = sum(effective_scores) / len(effective_scores)

        # Recompute model-level overall score used by UI comparisons
        model_tests = model_block.get("tests", {})
        test_overalls = []
        avg_latencies = []
        for tdata in model_tests.values():
            if not isinstance(tdata, dict):
                continue
            tsummary = tdata.get("summary", {})
            if not isinstance(tsummary, dict):
                continue
            if isinstance(tsummary.get("overall_score"), (int, float)):
                test_overalls.append(float(tsummary["overall_score"]))
            if isinstance(tsummary.get("avg_latency"), (int, float)):
                avg_latencies.append(float(tsummary["avg_latency"]))

        if test_overalls:
            model_overall = sum(test_overalls) / len(test_overalls)
            report_data.setdefault("summary", {}).setdefault("model_comparison", {}).setdefault(annotation.model_name, {})
            report_data["summary"]["model_comparison"][annotation.model_name]["overall_score"] = model_overall
            model_block.setdefault("overall_metrics", {})
            model_block["overall_metrics"]["weighted_score"] = model_overall

        if avg_latencies:
            avg_latency = sum(avg_latencies) / len(avg_latencies)
            report_data.setdefault("summary", {}).setdefault("model_comparison", {}).setdefault(annotation.model_name, {})
            report_data["summary"]["model_comparison"][annotation.model_name]["avg_latency"] = avg_latency

        self._recompute_report_summary(report_data)
        self._recompute_report_trends(report_data, reports_dir)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return True

    def _recompute_report_summary(self, report_data: Dict[str, Any]) -> None:
        """Recompute summary.model_comparison, best_performers and recommendations."""
        models = report_data.get("models", {})
        summary = report_data.setdefault("summary", {})
        existing_model_cmp = summary.get("model_comparison", {})
        model_comparison: Dict[str, Dict[str, Any]] = {}

        for model_key, model_data in models.items():
            tests = model_data.get("tests", {}) if isinstance(model_data, dict) else {}
            test_scores = []
            test_latencies = []

            for test_data in tests.values():
                if not isinstance(test_data, dict):
                    continue
                test_summary = test_data.get("summary", {})
                if not isinstance(test_summary, dict):
                    continue

                score = test_summary.get("overall_score")
                latency = test_summary.get("avg_latency")
                if isinstance(score, (int, float)):
                    test_scores.append(float(score))
                if isinstance(latency, (int, float)):
                    test_latencies.append(float(latency))

            overall_score = (sum(test_scores) / len(test_scores)) if test_scores else 0.0
            avg_latency = (sum(test_latencies) / len(test_latencies)) if test_latencies else 0.0

            cmp_entry = dict(existing_model_cmp.get(model_key, {}))
            cmp_entry["overall_score"] = overall_score
            cmp_entry["avg_latency"] = avg_latency
            model_comparison[model_key] = cmp_entry

            if isinstance(model_data, dict):
                model_data.setdefault("overall_metrics", {})
                model_data["overall_metrics"]["weighted_score"] = overall_score
                model_data["overall_metrics"]["latency_avg"] = avg_latency

        summary["model_comparison"] = model_comparison

        best_performers: Dict[str, Dict[str, Any]] = {}
        all_tests = set()
        for model_data in models.values():
            if isinstance(model_data, dict):
                all_tests.update(model_data.get("tests", {}).keys())

        for test_name in sorted(all_tests):
            best_model = None
            best_score = float("-inf")
            for model_key, model_data in models.items():
                if not isinstance(model_data, dict):
                    continue
                test_data = model_data.get("tests", {}).get(test_name, {})
                if not isinstance(test_data, dict):
                    continue
                test_summary = test_data.get("summary", {})
                if not isinstance(test_summary, dict):
                    continue
                score = test_summary.get("overall_score")
                if isinstance(score, (int, float)) and score > best_score:
                    best_score = float(score)
                    best_model = model_key

            if best_model is not None:
                best_performers[test_name] = {
                    "model": best_model,
                    "score": best_score
                }

        summary["best_performers"] = best_performers

        recommendations: List[str] = []
        if model_comparison:
            ranking = sorted(
                model_comparison.items(),
                key=lambda item: item[1].get("overall_score", 0),
                reverse=True
            )
            top_model, top_metrics = ranking[0]
            recommendations.append(
                f"🏆 En iyi model: {top_model} (overall_score: {top_metrics.get('overall_score', 0):.3f})"
            )

            if len(ranking) > 1:
                second_model, second_metrics = ranking[1]
                delta = top_metrics.get("overall_score", 0) - second_metrics.get("overall_score", 0)
                recommendations.append(
                    f"📊 Lider farkı: {top_model} vs {second_model} = {delta:.3f}"
                )

            fastest = min(
                model_comparison.items(),
                key=lambda item: item[1].get("avg_latency", float("inf"))
            )
            recommendations.append(
                f"⚡ En düşük gecikme: {fastest[0]} (avg_latency: {fastest[1].get('avg_latency', 0):.2f}s)"
            )

        summary["recommendations"] = recommendations

    def _recompute_report_trends(self, report_data: Dict[str, Any], reports_dir: str) -> None:
        """Recompute report trends based on historical runs for the same suite."""
        trend_analyzer = TrendAnalyzer(reports_dir=reports_dir)
        trends: Dict[str, Any] = {}

        current_ts = report_data.get("timestamp")
        current_suite = report_data.get("run_metadata", {}).get("test_suite")
        models = report_data.get("models", {})

        for model_key, model_data in models.items():
            historical_all = trend_analyzer.load_historical_results(
                model_key,
                limit=6,
                suite_filter=current_suite
            )
            historical = [h for h in historical_all if h.get("timestamp") != current_ts]

            current_score = None
            if isinstance(model_data, dict):
                current_score = model_data.get("overall_metrics", {}).get("weighted_score")

            if historical:
                trend_input = historical + [{
                    "timestamp": current_ts,
                    "file": "<current>",
                    "results": model_data,
                }]
                trend_data = trend_analyzer.calculate_trend(
                    trend_input,
                    "overall_metrics.weighted_score"
                )
                regressions = trend_analyzer.detect_regressions(model_data, historical)
                trends[model_key] = {
                    "trend": trend_data,
                    "regressions": regressions
                }
            elif isinstance(current_score, (int, float)):
                trends[model_key] = {
                    "trend": {
                        "values": [float(current_score)],
                        "timestamps": [current_ts],
                        "trend": "insufficient_history",
                        "change_pct": 0.0,
                        "history_runs": 0
                    },
                    "regressions": []
                }

        report_data["trends"] = trends
    
    def export_for_training(
        self,
        output_file: str = None,
        min_agreement_threshold: float = 0.2
    ) -> str:
        """
        Export annotations as training data.
        
        Format for fine-tuning LLM-as-Judge:
        {
            "messages": [
                {"role": "system", "content": "Sen bir değerlendirme uzmanısın..."},
                {"role": "user", "content": "Soru: ... Cevap: ... Değerlendir."},
                {"role": "assistant", "content": "score: X, reasoning: ..."}
            ]
        }
        
        Args:
            output_file: Output filename (default: training_data_TIMESTAMP.jsonl)
            min_agreement_threshold: Minimum agreement (1 - |llm_score - human_score|)
        
        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        output_path = self.annotations_dir / "training_data" / output_file
        
        annotations = self.load_all_annotations(status="completed")
        
        training_examples = []
        for annotation in annotations:
            # Calculate agreement
            agreement = 1 - abs(annotation.llm_judge_score - annotation.human_score)

            if agreement < min_agreement_threshold:
                continue

            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Sen bir LLM yanıt değerlendirme uzmanısın. Yanıtları 0-1 arası skorla ve gerekçelendir."
                    },
                    {
                        "role": "user",
                        "content": f"Soru: {annotation.question}\n\nModel Yanıtı: {annotation.model_response}\n\nBu yanıtı değerlendir."
                    },
                    {
                        "role": "assistant",
                        "content": f"Skor: {annotation.human_score:.2f}\n\nDeğerlendirme: {annotation.human_feedback}"
                    }
                ],
                "metadata": {
                    "annotation_id": annotation.annotation_id,
                    "test_category": annotation.test_category,
                    "correction_type": annotation.correction_type,
                    "original_llm_score": annotation.llm_judge_score,
                    "human_score": annotation.human_score,
                    "agreement": agreement
                }
            }
            training_examples.append(training_example)
        
        # Write training data
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        return str(output_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get annotation statistics"""
        completed = self.load_all_annotations(status="completed")
        pending = self.get_pending_items()
        
        if not completed:
            return {
                "total_completed": 0,
                "total_pending": len(pending),
                "message": "No completed annotations yet"
            }
        
        # Calculate agreement stats
        agreements = []
        corrections_by_type = {"approve": 0, "adjust": 0, "reject": 0}
        
        for annotation in completed:
            agreement = 1 - abs(annotation.llm_judge_score - annotation.human_score)
            agreements.append(agreement)
            corrections_by_type[annotation.correction_type] = corrections_by_type.get(annotation.correction_type, 0) + 1
        
        avg_agreement = sum(agreements) / len(agreements) if agreements else 0
        
        # Category breakdown
        by_category = {}
        for annotation in completed:
            if annotation.test_category not in by_category:
                by_category[annotation.test_category] = {"count": 0, "avg_human_score": 0}
            by_category[annotation.test_category]["count"] += 1
            by_category[annotation.test_category]["avg_human_score"] += annotation.human_score
        
        for category in by_category:
            count = by_category[category]["count"]
            by_category[category]["avg_human_score"] /= count
        
        return {
            "total_completed": len(completed),
            "total_pending": len(pending),
            "average_agreement": avg_agreement,
            "corrections_by_type": corrections_by_type,
            "by_category": by_category,
            "annotators": list(set(a.annotator_id for a in completed))
        }


def create_pending_from_results(
    results_file: str,
    annotation_manager: AnnotationManager,
    sample_per_test: int = 5,
    run_id: Optional[str] = None
) -> int:
    """
    Create pending review items from evaluation results.

    Args:
        results_file: Path to evaluation results JSON
        annotation_manager: AnnotationManager instance
        sample_per_test: Number of samples to take per test
        run_id: Specific run ID to use from a unified store file.
                If None, the latest run is used.

    Returns:
        Number of items added for review
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        loaded = json.load(f)

    source_report = Path(results_file).name

    # Backward compatibility: unified store format keeps all runs in one file.
    if isinstance(loaded, dict) and isinstance(loaded.get('runs'), list):
        runs = loaded.get('runs', [])
        if not runs:
            return 0

        if run_id:
            # Find the specific run requested by run_id
            matched = [r for r in runs if (r.get('run_metadata') or {}).get('run_id') == run_id]
            results = matched[0] if matched else None
            if results is None:
                logger.warning(f"Run ID '{run_id}' not found in {results_file}. Falling back to latest.")
                results = sorted(runs, key=lambda r: r.get('timestamp', ''), reverse=True)[0]
                actual_run_id = (results.get('run_metadata') or {}).get('run_id', 'latest')
            else:
                actual_run_id = run_id
        else:
            # Default: use latest run by timestamp
            results = sorted(runs, key=lambda r: r.get('timestamp', ''), reverse=True)[0]
            actual_run_id = (results.get('run_metadata') or {}).get('run_id', 'latest')

        source_report = f"{source_report}::{actual_run_id}"
    else:
        results = loaded

    pending_file = annotation_manager.annotations_dir / "pending" / "pending_reviews.jsonl"
    existing_items: List[Dict[str, Any]] = []
    existing_ids = set()

    if pending_file.exists():
        with open(pending_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    existing_item = json.loads(line)
                    existing_items.append(existing_item)
                    existing_id = existing_item.get("item_id")
                    if existing_id:
                        existing_ids.add(existing_id)
                except json.JSONDecodeError:
                    continue
    
    added_count = 0
    
    new_items: List[Dict[str, Any]] = []

    for model_key, model_data in results.get('models', {}).items():
        for test_name, test_data in model_data.get('tests', {}).items():
            if 'results' not in test_data:
                continue
            
            # Sample results
            test_results = test_data['results'][:sample_per_test]
            
            for result_idx, result in enumerate(test_results):
                # Extract model response based on test type
                model_response = _extract_model_response(result, test_name)
                
                # Extract question
                question = (
                    result.get('question') or 
                    result.get('input') or 
                    result.get('prompt') or
                    result.get('task') or
                    ''
                )
                
                # Extract LLM judge score - try multiple paths
                llm_score = _extract_llm_score(result)

                # Derive categorical label from score
                if llm_score >= 0.9:
                    llm_judge_label = "TAM_DOGRU"
                elif llm_score >= 0.4:
                    llm_judge_label = "KISMEN_DOGRU"
                else:
                    llm_judge_label = "YANLIS"
                
                # Extract reasoning
                reasoning = (
                    result.get('reasoning') or
                    result.get('llm_judge_reasoning') or
                    result.get('judge_reasoning') or
                    result.get('evaluation', {}).get('reasoning') or
                    ''
                )

                stable_result_id = result.get('id', result.get('test_id', f"idx_{result_idx}"))
                item_id = f"{source_report}|{model_key}|{test_name}|{stable_result_id}"

                if item_id in existing_ids:
                    continue

                item = {
                    "item_id": item_id,
                    "model_name": model_key,
                    "test_category": test_name,
                    "test_id": stable_result_id,
                    "question": question,
                    "model_response": model_response,
                    "llm_judge_score": llm_score,
                    "llm_judge_label": llm_judge_label,
                    "llm_judge_reasoning": reasoning,
                    "metadata": {
                        "latency": result.get('latency', 0),
                        "source_report": source_report,
                        "full_result": result
                    }
                }

                new_items.append(item)
                existing_ids.add(item_id)
                added_count += 1

    if new_items:
        with open(pending_file, 'w', encoding='utf-8') as f:
            for existing_item in existing_items:
                f.write(json.dumps(existing_item, ensure_ascii=False) + '\n')
            for item in new_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return added_count


def _extract_model_response(result: Dict[str, Any], test_name: str) -> str:
    """
    Extract model response from result based on test type.
    Different tests store responses in different fields.
    """
    # 1. Self-consistency tests: responses are in by_temperature.*.sample_responses
    if 'by_temperature' in result:
        temp_sections = []
        # Sort by temperature key
        sorted_temps = sorted(result.get('by_temperature', {}).items())
        for temp_key, temp_data in sorted_temps:
            if isinstance(temp_data, dict):
                sample_responses = temp_data.get('sample_responses', [])
                if sample_responses:
                    temp_value = temp_data.get('temperature', temp_key)
                    # Show all responses for this temperature
                    resp_lines = [f"[Temp {temp_value}]"]
                    for i, resp in enumerate(sample_responses, 1):
                        resp_lines.append(f"  {i}. {resp}")
                    temp_sections.append("\n".join(resp_lines))
        if temp_sections:
            return "\n\n".join(temp_sections)
    
    # 2. Multi-turn tests: responses are in turns[].response
    if 'turns' in result:
        turn_responses = []
        for turn in result.get('turns', []):
            if isinstance(turn, dict) and turn.get('response'):
                turn_responses.append(f"Turn {turn.get('turn', '?')}: {turn['response']}")
        if turn_responses:
            return "\n".join(turn_responses)
    
    # 3. Function calling tests: tool_calls or selected_tool + parameters
    if 'tool_calls' in result or 'selected_tool' in result:
        tool_info = []
        if result.get('selected_tool'):
            tool_info.append(f"Tool: {result['selected_tool']}")
        if result.get('parameters'):
            params_str = json.dumps(result['parameters'], ensure_ascii=False)
            tool_info.append(f"Parameters: {params_str}")
        if result.get('tool_calls'):
            for tc in result.get('tool_calls', []):
                if isinstance(tc, dict):
                    tool_info.append(f"Tool Call: {tc.get('name', 'unknown')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)})")
        if tool_info:
            return "\n".join(tool_info)
    
    # 4. Agentic tests: plan or reasoning + final_answer
    if 'plan' in result or ('reasoning' in result and 'final_answer' in result):
        parts = []
        if result.get('plan'):
            parts.append(f"Plan: {result['plan']}")
        if result.get('reasoning'):
            parts.append(f"Reasoning: {result['reasoning']}")
        if result.get('final_answer'):
            parts.append(f"Answer: {result['final_answer']}")
        if parts:
            return "\n".join(parts)
    
    # 5. PII detection tests: detected_pii or detection_result
    if 'detected_pii' in result or 'pii_found' in result:
        pii_info = []
        if result.get('model_response'):
            pii_info.append(f"Response: {result['model_response']}")
        if result.get('detected_pii'):
            pii_info.append(f"Detected PII: {json.dumps(result['detected_pii'], ensure_ascii=False)}")
        if result.get('pii_found'):
            pii_info.append(f"PII Found: {result['pii_found']}")
        if pii_info:
            return "\n".join(pii_info)
    
    # 6. Standard fields - try common field names
    standard_fields = [
        'model_answer',
        'model_response', 
        'response',
        'output',
        'predicted_value',
        'answer',
        'content',
        'text',
        'completion'
    ]
    
    for field in standard_fields:
        if result.get(field):
            return str(result[field])
    
    # 7. Check nested structures
    if 'evaluation' in result and isinstance(result['evaluation'], dict):
        for field in standard_fields:
            if result['evaluation'].get(field):
                return str(result['evaluation'][field])
    
    # 8. Last resort: return scores summary if available
    if 'scores' in result:
        return f"Scores: {json.dumps(result['scores'], ensure_ascii=False)}"
    
    return ''


def _extract_llm_score(result: Dict[str, Any]) -> float:
    """Extract LLM judge score from result."""
    # Try direct score fields
    score_fields = ['overall', 'score', 'llm_judge_score', 'judge_score', 'consistency_score']
    
    # Check in scores dict
    scores = result.get('scores', {})
    if isinstance(scores, dict):
        for field in score_fields:
            if field in scores:
                return float(scores[field])
        # Average of all scores if no specific one found
        numeric_scores = [v for v in scores.values() if isinstance(v, (int, float))]
        if numeric_scores:
            return sum(numeric_scores) / len(numeric_scores)
    
    # Check direct fields
    for field in score_fields:
        if field in result and isinstance(result[field], (int, float)):
            return float(result[field])
    
    # Check evaluation dict
    if 'evaluation' in result and isinstance(result['evaluation'], dict):
        for field in score_fields:
            if field in result['evaluation']:
                return float(result['evaluation'][field])
    
    return 0.5  # Default

