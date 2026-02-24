"""
Human Feedback Evaluator
Uses human annotations to evaluate models or calibrate LLM-as-Judge
"""
from typing import Dict, List, Any, Optional
from adapters.unified_adapter import UnifiedLLMAdapter
from utils.human_annotations import AnnotationManager, HumanAnnotation
import statistics


class HumanFeedbackEvaluator:
    """
    Evaluator that uses human annotations.
    
    Can be used to:
    1. Compare model performance with human-validated scores
    2. Measure LLM-Judge accuracy vs. human judgments
    3. Calibrate LLM-Judge using human feedback
    """
    
    def __init__(self, annotation_manager: AnnotationManager = None):
        self.annotation_manager = annotation_manager or AnnotationManager()
    
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
        
        return {
            "overall_metrics": {
                "average_agreement": judge_accuracy.get("average_agreement", 0),
                "mean_absolute_error": mae,
                "judge_bias": bias
            },
            "recommendations": recommendations,
            "training_data_available": judge_accuracy.get("total_comparisons", 0),
            "ready_for_finetuning": judge_accuracy.get("total_comparisons", 0) >= 50
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
        
        annotations = self.annotation_manager.load_all_annotations(status="completed")
        
        disagreements = []
        for ann in annotations:
            score_diff = abs(ann.llm_judge_score - ann.human_score)
            if score_diff >= threshold:
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
                        "correction_type": ann.correction_type
                    },
                    "score_difference": score_diff,
                    "direction": "LLM too high" if ann.llm_judge_score > ann.human_score else "LLM too low",
                    "timestamp": ann.timestamp
                })
        
        output_path = Path("annotations") / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_disagreements": len(disagreements),
                "threshold": threshold,
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
