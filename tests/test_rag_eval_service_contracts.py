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

    def test_no_expected_answer_marks_recall_not_applicable(self):
        response = self.service.evaluate(
            question="What is the capital of France?",
            contexts=[RagContext(text="Paris is the capital of France.", source="doc1")],
            answer="Paris.",
            expected_answer="",
        )

        self.assertFalse(response.context_recall_applicable)

    def test_with_expected_answer_marks_recall_applicable_even_when_zero(self):
        # A genuinely bad recall (context doesn't cover the expected answer)
        # must stay distinguishable from "not applicable" — both currently
        # render as context_recall == 0.0.
        response = self.service.evaluate(
            question="What is the capital of France?",
            contexts=[RagContext(text="Paris is the capital of France.", source="doc1")],
            answer="Paris.",
            expected_answer="Completely unrelated ground truth text.",
        )

        self.assertTrue(response.context_recall_applicable)
        self.assertEqual(response.context_recall, 0.0)

    def test_overall_score_matches_underlying_weighted_score(self):
        response = self.service.evaluate(
            question="What is the capital of France?",
            contexts=[RagContext(text="Paris is the capital of France.", source="doc1")],
            answer="The capital of France is Paris.",
            expected_answer="Paris",
        )

        self.assertAlmostEqual(response.overall_score, response.details["overall_rag_score"], places=4)


class _FakeEmbeddingAdapter:
    """Maps each known text to a fixed vector; paraphrases share a vector so
    a test can prove embedding-mode actually scores by meaning, not words."""

    VECTORS = {
        "How long is the return window?": [1.0, 0.0, 0.0],
        "Customers may send back purchased items within thirty days of the purchase date.": [1.0, 0.0, 0.0],
        "You have a one month window to send the product back.": [1.0, 0.0, 0.0],
    }

    def __init__(self):
        self.encode_calls = 0

    def encode(self, texts, normalize=True):
        self.encode_calls += 1
        return {"embeddings": [self.VECTORS.get(t, [0.0, 1.0, 0.0]) for t in texts]}


class RagEvalServiceEmbeddingModeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = _FakeEmbeddingAdapter()
        self.factory_calls = []

        def factory(model_key, config_path):
            self.factory_calls.append(model_key)
            return self.adapter

        self.service = RagEvalService(embedding_adapter_factory=factory)

    def test_embedding_mode_scores_a_paraphrase_as_relevant(self):
        # Same case as the token-overlap paraphrase weakness this feature
        # exists to address: zero shared words, but embedding-mode's fake
        # vectors treat them as identical in meaning, so relevance must be high.
        response = self.service.evaluate(
            question="How long is the return window?",
            contexts=[RagContext(
                text="Customers may send back purchased items within thirty days of the purchase date.",
            )],
            answer="You have a one month window to send the product back.",
            embedding_model="fake-embed",
        )

        self.assertEqual(response.scoring_mode, "embedding")
        self.assertEqual(response.embedding_model, "fake-embed")
        self.assertGreater(response.answer_relevance, 0.9)
        self.assertGreater(response.faithfulness, 0.9)

    def test_no_embedding_model_stays_in_token_overlap_mode(self):
        response = self.service.evaluate(
            question="What is the capital of France?",
            contexts=[RagContext(text="Paris is the capital of France.")],
            answer="Paris.",
        )

        self.assertEqual(response.scoring_mode, "token_overlap")
        self.assertIsNone(response.embedding_model)
        self.assertEqual(self.adapter.encode_calls, 0)

    def test_adapter_is_cached_across_calls_for_the_same_model(self):
        for _ in range(3):
            self.service.evaluate(
                question="q",
                contexts=[RagContext(text="c")],
                answer="a",
                embedding_model="fake-embed",
            )

        self.assertEqual(self.factory_calls, ["fake-embed"])  # built once, reused

    def test_unknown_embedding_model_raises_value_error(self):
        def factory(model_key, config_path):
            raise ValueError(f"Embedding model '{model_key}' not found in config")

        service = RagEvalService(embedding_adapter_factory=factory)
        with self.assertRaises(ValueError):
            service.evaluate(
                question="q",
                contexts=[RagContext(text="c")],
                answer="a",
                embedding_model="does-not-exist",
            )


if __name__ == "__main__":
    unittest.main()
