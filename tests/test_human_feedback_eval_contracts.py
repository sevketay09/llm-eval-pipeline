"""Contract tests for HumanFeedbackEvaluator.compute_inter_rater_reliability."""
import pytest

from evaluators.human_feedback_eval import HumanFeedbackEvaluator
from utils.human_annotations import AnnotationManager, HumanAnnotation


def _annotation(annotation_id, test_id, annotator_id, human_score, timestamp="2026-01-01T00:00:00"):
    return HumanAnnotation(
        annotation_id=annotation_id,
        test_id=test_id,
        test_category="cat",
        model_name="model-a",
        question="q",
        model_response="r",
        llm_judge_score=0.5,
        llm_judge_reasoning="",
        human_score=human_score,
        human_feedback="",
        correction_type="approve",
        annotator_id=annotator_id,
        timestamp=timestamp,
    )


def _evaluator(tmp_path):
    manager = AnnotationManager(annotations_dir=str(tmp_path / "annotations"))
    return HumanFeedbackEvaluator(annotation_manager=manager), manager


def test_single_reviewer_workflow_reports_not_applicable(tmp_path):
    evaluator, manager = _evaluator(tmp_path)
    manager.save_annotation(_annotation("a1", "case-1", "reviewer-a", 0.8), status="completed")
    manager.save_annotation(_annotation("a2", "case-2", "reviewer-a", 0.4), status="completed")

    result = evaluator.compute_inter_rater_reliability()

    assert result["applicable"] is False
    assert result["multi_reviewer_case_count"] == 0
    assert result["total_reviewed_cases"] == 2


def test_no_annotations_reports_not_applicable(tmp_path):
    evaluator, _ = _evaluator(tmp_path)

    result = evaluator.compute_inter_rater_reliability()

    assert result["applicable"] is False
    assert result["total_reviewed_cases"] == 0


def test_overlapping_reviewers_compute_agreement(tmp_path):
    evaluator, manager = _evaluator(tmp_path)
    manager.save_annotation(_annotation("a1", "case-1", "reviewer-a", 0.8), status="completed")
    manager.save_annotation(_annotation("a2", "case-1", "reviewer-b", 0.7), status="completed")
    manager.save_annotation(_annotation("a3", "case-2", "reviewer-a", 0.2), status="completed")

    result = evaluator.compute_inter_rater_reliability(tolerance=0.15)

    assert result["applicable"] is True
    assert result["multi_reviewer_case_count"] == 1
    assert result["total_reviewed_cases"] == 2
    assert result["mean_pairwise_absolute_difference"] == pytest.approx(0.1)
    assert result["average_agreement"] == pytest.approx(0.9)
    assert result["within_tolerance_rate"] == 1.0
    assert result["distinct_annotators"] == 2


def test_same_annotator_reannotating_does_not_count_as_multi_reviewer(tmp_path):
    evaluator, manager = _evaluator(tmp_path)
    manager.save_annotation(_annotation("a1", "case-1", "reviewer-a", 0.8, timestamp="2026-01-01T00:00:00"), status="completed")
    manager.save_annotation(_annotation("a2", "case-1", "reviewer-a", 0.6, timestamp="2026-01-02T00:00:00"), status="completed")

    result = evaluator.compute_inter_rater_reliability()

    assert result["applicable"] is False
