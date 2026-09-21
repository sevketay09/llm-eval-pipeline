"""Contract tests for the Jev cascade -> HITL wiring in utils/human_annotations.py
and evaluators/human_feedback_eval.py. Offline only, no network.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from utils.human_annotations import (
    HITL_LOW_CONFIDENCE,
    AnnotationManager,
    _extract_judge_signal,
    create_pending_from_results,
)


def _result(score=0.9, judge_meta=None) -> dict:
    r = {
        "scores": {"judge_label": "TAM_DOGRU", "judge_score": score},
        "structured_output": {"is_valid": True},
    }
    if judge_meta is not None:
        r["judge_meta"] = judge_meta
    return r


class ExtractJudgeSignalContractTests(unittest.TestCase):
    def test_low_confidence_sets_uncertain_queue_reason_and_boosts_priority(self):
        signal = _extract_judge_signal(_result(judge_meta={"backend": "decision", "min_confidence": 0.4, "escalated": False}))
        self.assertIn("uncertain", signal["queue_reason"].lower())
        self.assertGreaterEqual(signal["review_priority"], 40.0)
        self.assertEqual(signal["judge_backend"], "decision")
        self.assertEqual(signal["judge_confidence"], 0.4)

    def test_high_confidence_does_not_trigger_uncertain_reason(self):
        signal = _extract_judge_signal(_result(judge_meta={"backend": "decision", "min_confidence": 0.95, "escalated": False}))
        self.assertNotIn("uncertain", signal["queue_reason"].lower())

    def test_missing_judge_meta_matches_pre_existing_behavior(self):
        signal = _extract_judge_signal(_result())
        self.assertIsNone(signal["judge_backend"])
        self.assertIsNone(signal["judge_confidence"])
        self.assertEqual(signal["queue_reason"], "Representative review sample")

    def test_custom_hitl_below_threshold(self):
        signal = _extract_judge_signal(
            _result(judge_meta={"backend": "decision", "min_confidence": 0.5, "escalated": False}),
            hitl_below=0.3,
        )
        self.assertNotIn("uncertain", signal["queue_reason"].lower())


class CreatePendingFromResultsContractTests(unittest.TestCase):
    def _write_report(self, tmp_path: Path, confidences) -> Path:
        results = []
        for i, conf in enumerate(confidences):
            results.append({
                "id": f"case_{i}",
                "question": "q",
                "scores": {"judge_label": "TAM_DOGRU", "judge_score": 0.9},
                "structured_output": {"is_valid": True},
                "judge_meta": {"backend": "decision", "min_confidence": conf, "escalated": False},
            })
        report = {
            "run_metadata": {"judge_cascade": {"accept_confidence": 0.85, "hitl_below": HITL_LOW_CONFIDENCE}},
            "models": {"demo-model": {"tests": {"turkish_grammar": {"results": results}}}},
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))
        return path

    def test_low_confidence_only_filters_candidates(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = self._write_report(tmp_path, [0.3, 0.9, 0.95])
            manager = AnnotationManager(annotations_dir=str(tmp_path / "annotations"))

            added = create_pending_from_results(str(report_path), manager, sample_per_test=10, low_confidence_only=True)
            self.assertEqual(added, 1)

            items = manager.get_pending_items()
            self.assertEqual(len(items), 1)
            self.assertLess(items[0]["judge_confidence"], HITL_LOW_CONFIDENCE)

    def test_without_filter_keeps_all_candidates(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = self._write_report(tmp_path, [0.3, 0.9, 0.95])
            manager = AnnotationManager(annotations_dir=str(tmp_path / "annotations"))

            added = create_pending_from_results(str(report_path), manager, sample_per_test=10)
            self.assertEqual(added, 3)


class CalibrationByBackendContractTests(unittest.TestCase):
    def test_by_backend_groups_annotations(self):
        import tempfile

        from evaluators.human_feedback_eval import HumanFeedbackEvaluator
        from utils.human_annotations import HumanAnnotation

        with tempfile.TemporaryDirectory() as tmp:
            manager = AnnotationManager(annotations_dir=tmp)
            for i in range(6):
                backend = "decision" if i < 3 else "llm"
                ann = HumanAnnotation(
                    annotation_id=f"a{i}",
                    test_id=f"t{i}",
                    test_category="turkish_grammar",
                    model_name="demo-model",
                    question="q",
                    model_response="r",
                    llm_judge_score=0.8,
                    llm_judge_reasoning="",
                    human_score=0.8 if backend == "decision" else 0.2,
                    human_feedback="",
                    correction_type="approve",
                    verdict={},
                    annotator_id="tester",
                    timestamp="2026-01-01T00:00:00",
                    metadata={"judge_backend": backend},
                )
                manager.save_annotation(ann, status="completed")

            evaluator = HumanFeedbackEvaluator(annotation_manager=manager)
            insights = evaluator.get_calibration_insights()
            self.assertIn("by_backend", insights)
            self.assertIn("decision", insights["by_backend"])
            self.assertIn("llm", insights["by_backend"])
            self.assertEqual(insights["by_backend"]["decision"]["n"], 3)


if __name__ == "__main__":
    unittest.main()
