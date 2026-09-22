"""Contract tests for decisions/dataset_qc.py and its wiring into
CustomDatasetService. Offline only — MockDecisionClient, no network.
"""
from __future__ import annotations

import unittest

from decisions.client import BaseDecisionClient
from decisions.dataset_qc import QCThresholds, qc_cases
from decisions.mock_client import MockDecisionClient
from decisions.types import Decision, DecisionResponse


def _case(qid: str, question: str = "What is the return policy?", answer: str = "30 days.") -> dict:
    return {"id": qid, "question": question, "expected_answer": answer}


class _MarkerClient(BaseDecisionClient):
    """Answers each question from a marker embedded in the question text
    (`AMB:1`, `NONDET:1`, `SUP:0`), so individual cases can be steered
    independently — MockDecisionClient's `overrides` apply the same value to
    every case asking a given question, which can't express "only this one
    case is ambiguous."

    Defaults (no marker): not ambiguous, not nondeterministic, supported,
    medium difficulty — i.e. a case that survives QC untouched.
    """

    def __init__(self) -> None:
        super().__init__("marker-client", "marker-client")

    def _decide_raw(self, state, questions):
        defaults = {"ambiguous": 0.0, "nondeterministic": 0.0, "supported": 1.0, "difficulty": 0.5}
        markers = {"ambiguous": "AMB:1", "nondeterministic": "NONDET:1", "supported": "SUP:0"}
        answers = {}
        for name in questions:
            value = defaults[name]
            if name in markers and markers[name] in state:
                value = 1.0 if name != "supported" else 0.0
            answers[name] = Decision(kind="noul", value=value, confidence=1.0, probabilities={})
        return DecisionResponse(answers=answers, latency_ms=1.0, est_input_tokens=1, model=self.model_name)


class QcCasesContractTests(unittest.TestCase):
    def test_ambiguous_cases_are_dropped_and_counted(self):
        cases = [
            _case("c1", question="AMB:1 case one"),
            _case("c2", question="AMB:1 case two"),
            _case("c3", question="clear case A"),
            _case("c4", question="clear case B"),
            _case("c5", question="clear case C"),
        ]
        client = _MarkerClient()
        kept, summary = qc_cases(client, cases)
        self.assertEqual(len(kept), 3)
        self.assertEqual(summary["qc_ambiguous_removed"], 2)
        self.assertNotIn("qc_skipped_reason", summary)

    def test_would_leave_fewer_than_3_skips_drop_entirely(self):
        cases = [_case("c1"), _case("c2"), _case("c3")]
        client = MockDecisionClient(overrides={"ambiguous": 0.9})
        kept, summary = qc_cases(client, cases)
        self.assertEqual(len(kept), 3)
        self.assertEqual(summary["qc_skipped_reason"], "would_leave_fewer_than_3")
        self.assertEqual(summary["qc_ambiguous_removed"], 0)
        for case in kept:
            self.assertIn("qc", case)
            self.assertNotIn("error", case["qc"])

    def test_clean_cases_are_kept_and_annotated(self):
        cases = [_case("c1"), _case("c2"), _case("c3"), _case("c4")]
        client = MockDecisionClient(overrides={"ambiguous": 0.1, "nondeterministic": 0.1})
        kept, summary = qc_cases(client, cases)
        self.assertEqual(len(kept), 4)
        self.assertEqual(summary["qc_ambiguous_removed"], 0)
        self.assertEqual(summary["qc_nondeterministic_removed"], 0)
        for case in kept:
            self.assertIn(case["qc"]["difficulty"], {"easy", "medium", "hard"})
            self.assertIsNone(case["qc"]["supported"])

    def test_supported_question_only_asked_with_source_material(self):
        cases = [_case("c1")]

        class RecordingClient(MockDecisionClient):
            def _decide_raw(self, state, questions):
                self.last_questions = set(questions)
                return super()._decide_raw(state, questions)

        client = RecordingClient()
        qc_cases(client, cases)
        self.assertNotIn("supported", client.last_questions)

        client2 = RecordingClient()
        qc_cases(client2, cases, source_material="Returns are accepted within 30 days.")
        self.assertIn("supported", client2.last_questions)

    def test_unsupported_case_dropped_when_source_present(self):
        cases = [
            _case("c1", question="SUP:0 case one"),
            _case("c2", question="clear case A"),
            _case("c3", question="clear case B"),
            _case("c4", question="clear case C"),
        ]
        client = _MarkerClient()
        kept, summary = qc_cases(client, cases, source_material="Some source text.")
        self.assertEqual(len(kept), 3)
        self.assertEqual(summary["qc_unsupported_removed"], 1)

    def test_decision_error_marks_case_without_dropping(self):
        cases = [_case("c1"), _case("c2"), _case("c3")]
        class FastFailingClient(MockDecisionClient):
            handles_own_retries = True

        client = FastFailingClient(fail_times=1)
        kept, summary = qc_cases(client, cases)
        self.assertEqual(len(kept), 3)
        self.assertEqual(summary["qc_errors"], 1)
        self.assertTrue(any(case["qc"].get("error") for case in kept))

    def test_custom_thresholds_respected(self):
        cases = [
            _case("c1", question="AMB:1 case one"),
            _case("c2", question="clear case A"),
            _case("c3", question="clear case B"),
            _case("c4", question="clear case C"),
        ]
        client = _MarkerClient()
        kept, summary = qc_cases(client, cases, thresholds=QCThresholds(ambiguous=1.5))
        self.assertEqual(len(kept), 4)
        self.assertEqual(summary["qc_ambiguous_removed"], 0)


class CustomDatasetServiceQcWiringTests(unittest.TestCase):
    def test_qc_model_none_leaves_cases_and_summary_unchanged(self):
        from api.services.custom_dataset_service import CustomDatasetService

        service = CustomDatasetService.__new__(CustomDatasetService)
        cases = [_case("c1"), _case("c2"), _case("c3")]
        summary = service._run_dataset_qc(None, cases, source_material=None, focus_areas=None)
        self.assertEqual(summary, {})
        self.assertEqual(len(cases), 3)
        self.assertNotIn("qc", cases[0])

    def test_qc_model_set_filters_cases_via_injected_factory(self):
        from api.services.custom_dataset_service import CustomDatasetService

        service = CustomDatasetService.__new__(CustomDatasetService)
        service._config_path = "config/models.yaml"
        service._decision_clients = {}
        service.decision_client_factory = lambda model_key, config_path: _MarkerClient()
        cases = [
            _case("c1", question="AMB:1 case one"),
            _case("c2", question="clear case A"),
            _case("c3", question="clear case B"),
            _case("c4", question="clear case C"),
        ]
        summary = service._run_dataset_qc(
            "demo-jev", cases, source_material=None, focus_areas=None
        )
        self.assertEqual(len(cases), 3)
        self.assertEqual(summary["qc_ambiguous_removed"], 1)
        self.assertIn("demo-jev", service._decision_clients)


if __name__ == "__main__":
    unittest.main()
