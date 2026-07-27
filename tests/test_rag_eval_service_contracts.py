"""Contract tests for RagEvalService — the API-facing wrapper around
analysis.rag_eval. evaluate_rag_case() itself is covered by
test_rag_eval_contracts.py; these tests catch service-layer field mapping
bugs (e.g. reading from a key path the underlying function doesn't use)."""
import unittest

from api.schemas.rag_eval import RagContext
from api.services.rag_eval_service import RagEvalService


class RagEvalServiceContractTests(unittest.TestCase):
    def setUp(self):
        self.service = RagEvalService()

    def test_good_case_produces_nonzero_scores(self):
        # A clearly good RAG case (relevant context, faithful+relevant answer)
        # must not come back as all zeros — before the fix, the service read
        # every metric from result["scores"][...]["score"], a key path
        # evaluate_rag_case() never populates, so this always regressed to
        # 0.0 / fault_component="none" regardless of input quality.
        response = self.service.evaluate(
            question="What is the capital of France?",
            contexts=[RagContext(text="Paris is the capital of France.", source="doc1")],
            answer="The capital of France is Paris.",
            expected_answer="Paris",
        )

        self.assertGreater(response.context_precision, 0.0)
        self.assertGreater(response.context_recall, 0.0)
        self.assertGreater(response.faithfulness, 0.0)
        self.assertGreater(response.answer_relevance, 0.0)
        self.assertGreater(response.overall_score, 0.0)

    def test_bad_case_detects_retriever_fault(self):
        response = self.service.evaluate(
            question="What is the capital of France?",
            contexts=[RagContext(text="Bananas are a good source of potassium.", source="doc1")],
            answer="I don't know.",
            expected_answer="Paris",
        )

        self.assertEqual(response.fault_component, "retriever")

    def test_hallucinated_answer_detects_generator_fault(self):
        response = self.service.evaluate(
            question="What is the capital of France?",
            contexts=[RagContext(text="Paris is the capital of France.", source="doc1")],
            answer="The capital of France is Berlin, established in 1850.",
            expected_answer="Paris",
        )

        self.assertEqual(response.fault_component, "generator")

    def test_no_expected_answer_still_returns_a_valid_response(self):
        # context_recall's underlying "recall" is None (not 0.0) when no
        # expected_answer is given — the response schema's field is a plain
        # float, so this must not raise a Pydantic validation error.
        response = self.service.evaluate(
            question="What is the capital of France?",
            contexts=[RagContext(text="Paris is the capital of France.", source="doc1")],
            answer="Paris.",
            expected_answer="",
        )

        self.assertEqual(response.context_recall, 0.0)
        self.assertGreater(response.faithfulness, 0.0)

    def test_overall_score_matches_underlying_weighted_score(self):
        response = self.service.evaluate(
            question="What is the capital of France?",
            contexts=[RagContext(text="Paris is the capital of France.", source="doc1")],
            answer="The capital of France is Paris.",
            expected_answer="Paris",
        )

        self.assertAlmostEqual(response.overall_score, response.details["overall_rag_score"], places=4)


if __name__ == "__main__":
    unittest.main()
