"""Contract tests for analysis.rag_eval — offline, no real LLM/embedding."""
import math
import unittest

import numpy as np

from analysis.rag_eval import (
    _cosine,
    _tokenize,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_answer_relevance,
    isolate_fault,
    evaluate_rag_case,
    evaluate_rag_report,
)


class TokenizeContractTests(unittest.TestCase):
    """Turkish-specific tokenization: a-z0-9 alone drops every Turkish letter
    (ç ğ ı ö ş ü) and Python's default .lower() mangles the Turkish dotted
    capital İ, both of which used to fragment/lose Turkish words entirely."""

    def test_turkish_letters_are_kept_not_dropped(self):
        self.assertEqual(_tokenize("gündür"), ["gündür"])
        self.assertEqual(_tokenize("kaç"), ["kaç"])

    def test_turkish_dotted_capital_i_does_not_fragment_the_word(self):
        # Before the fix: "İade" -> "i" + combining-dot-above + "ade", which
        # the [a-z...] regex split into two separate tokens ("i", "ade")
        # instead of one ("iade").
        self.assertEqual(_tokenize("İade süresi"), ["iade", "süresi"])

    def test_realistic_turkish_question_is_fully_tokenized(self):
        tokens = _tokenize("İade süresi kaç gündür?")
        self.assertEqual(tokens, ["iade", "süresi", "kaç", "gündür"])


class ContextPrecisionContractTests(unittest.TestCase):
    """Contract tests for compute_context_precision."""

    def test_relevant_context_high_precision(self):
        """Relevant context (high token overlap) should give precision > 0.5."""
        question = "What is Python programming language?"
        contexts = [
            "Python programming is used for data science and web development.",
            "Python language is high level and easy to learn.",
        ]
        result = compute_context_precision(question, contexts)

        self.assertGreater(result["precision"], 0.5)
        self.assertGreater(result["relevant_count"], 0)
        self.assertEqual(result["total_chunks"], 2)
        self.assertEqual(len(result["chunk_scores"]), 2)

    def test_irrelevant_context_low_precision(self):
        """Irrelevant context (low token overlap) should give precision < 0.5."""
        question = "What is Python programming?"
        contexts = [
            "Cooking recipes for French cuisine.",
            "How to bake a chocolate cake.",
        ]
        result = compute_context_precision(question, contexts)

        self.assertLess(result["precision"], 0.5)
        self.assertEqual(result["total_chunks"], 2)

    def test_empty_contexts_zero_precision(self):
        """Empty contexts should give precision 0.0."""
        question = "What is Python?"
        contexts = []
        result = compute_context_precision(question, contexts)

        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["relevant_count"], 0)
        self.assertEqual(result["total_chunks"], 0)
        self.assertEqual(result["chunk_scores"], [])


class ContextRecallContractTests(unittest.TestCase):
    """Contract tests for compute_context_recall."""

    def test_answer_in_context_high_recall(self):
        """When expected answer words are in context, recall should be high."""
        question = "What is data science?"
        contexts = [
            "Data science is an interdisciplinary field.",
            "It uses statistics and machine learning techniques.",
        ]
        expected_answer = "Data science uses statistics and programming."
        result = compute_context_recall(question, contexts, expected_answer)

        self.assertGreater(result["recall"], 0.5)
        self.assertIsNotNone(result["recall"])

    def test_answer_not_in_context_low_recall(self):
        """When expected answer words are not in context, recall should be low."""
        question = "What is cooking?"
        contexts = [
            "Programming is the art of writing code.",
            "Software development requires problem solving.",
        ]
        expected_answer = "Cooking involves preparing food with heat and spices."
        result = compute_context_recall(question, contexts, expected_answer)

        self.assertLess(result["recall"], 0.5)

    def test_no_expected_answer_returns_zero(self):
        """When expected_answer is empty string, recall should be 0.0."""
        question = "What is data?"
        contexts = ["Data is processed information."]
        expected_answer = ""
        result = compute_context_recall(question, contexts, expected_answer)

        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["covered_tokens"], 0)


class FaithfulnessContractTests(unittest.TestCase):
    """Contract tests for compute_faithfulness."""

    def test_grounded_answer_high_faithfulness(self):
        """When answer words are in context, faithfulness should be high."""
        answer = "Python is a programming language used for data science."
        contexts = [
            "Python is a high-level programming language.",
            "Data science uses Python and other tools.",
        ]
        result = compute_faithfulness(answer, contexts)

        self.assertGreater(result["faithfulness"], 0.5)
        self.assertGreater(result["grounded_tokens"], 0)

    def test_hallucinated_answer_low_faithfulness(self):
        """When answer contains words not in context, faithfulness should be low."""
        answer = "Cooking involves sautéing vegetables at high temperature with special spices."
        contexts = [
            "Programming is writing code.",
            "Software development requires problem solving.",
        ]
        result = compute_faithfulness(answer, contexts)

        self.assertLess(result["faithfulness"], 0.5)

    def test_empty_answer_returns_zero(self):
        """When answer is empty, faithfulness should be 0.0."""
        answer = ""
        contexts = ["Some context text."]
        result = compute_faithfulness(answer, contexts)

        self.assertEqual(result["faithfulness"], 0.0)
        self.assertEqual(result["total_answer_tokens"], 0)


class IsolateFaultContractTests(unittest.TestCase):
    """Contract tests for isolate_fault."""

    def _case_result(self, cp, cr, faith, ar):
        """Helper: build a case_result dict from metrics."""
        return {
            "context_precision": {"precision": cp},
            "context_recall": {"recall": cr},
            "faithfulness": {"faithfulness": faith},
            "answer_relevance": {"answer_relevance": ar},
        }

    def test_bad_retrieval_fault(self):
        """Low precision and recall indicate retriever fault."""
        case_result = self._case_result(cp=0.2, cr=0.2, faith=0.8, ar=0.8)
        result = isolate_fault(case_result)

        self.assertEqual(result["fault"], "retriever")
        self.assertIn("fault", result)

    def test_hallucination_fault(self):
        """Low faithfulness indicates generator fault (hallucination)."""
        case_result = self._case_result(cp=0.8, cr=0.8, faith=0.2, ar=0.7)
        result = isolate_fault(case_result)

        self.assertEqual(result["fault"], "generator")

    def test_no_fault_all_good(self):
        """High scores across all metrics indicate no fault."""
        case_result = self._case_result(cp=0.9, cr=0.9, faith=0.9, ar=0.9)
        result = isolate_fault(case_result)

        self.assertEqual(result["fault"], "none")
        self.assertIn("low", result["severity"])


class EvaluateRagCaseContractTests(unittest.TestCase):
    """Contract tests for evaluate_rag_case."""

    def _rag_case(self, q="Python nedir?", contexts=None, answer="Python bir dil.", expected=None):
        """Helper: build a RAG case."""
        c = {
            "question": q,
            "contexts": contexts or ["Python yüksek seviyeli bir dil."],
            "answer": answer,
        }
        if expected:
            c["expected_answer"] = expected
        return c

    def test_required_keys_present(self):
        """Result should contain all required metric keys."""
        case = self._rag_case()
        result = evaluate_rag_case(case)

        self.assertIn("context_precision", result)
        self.assertIn("faithfulness", result)
        self.assertIn("answer_relevance", result)
        self.assertIn("fault_isolation", result)
        self.assertIn("overall_rag_score", result)

    def test_overall_score_in_range(self):
        """overall_rag_score should be between 0.0 and 1.0."""
        case = self._rag_case()
        result = evaluate_rag_case(case)

        self.assertGreaterEqual(result["overall_rag_score"], 0.0)
        self.assertLessEqual(result["overall_rag_score"], 1.0)

    def test_with_expected_answer_includes_recall(self):
        """When expected_answer is provided, context_recall should not be None."""
        case = self._rag_case(expected="Python bir programlama dilidir.")
        result = evaluate_rag_case(case)

        self.assertIsNotNone(result["context_recall"]["recall"])

    def test_turkish_question_shares_the_key_topic_word_with_its_context(self):
        # Regression for the a-z0-9-only tokenizer bug: before the fix, "İade"
        # (return/refund — the one word this question and context are both
        # actually about) got mangled into "ade" for the question while the
        # context correctly kept "iade", so they shared zero tokens even
        # though the context is an exact-topic match. compute_context_precision's
        # 0.5 relevance threshold still isn't cleared here (Turkish suffixes
        # like gün/gündür aren't stemmed, a separate, deeper limitation this
        # fix doesn't address) — this test only locks in the concrete,
        # demonstrable fix: the shared word is recovered at all.
        tokens_q = set(_tokenize("İade süresi kaç gündür?"))
        tokens_c = set(_tokenize(
            "Ürünler satın alma tarihinden itibaren otuz gün içinde iade edilebilir."
        ))
        self.assertIn("iade", tokens_q & tokens_c)


class EvaluateRagReportContractTests(unittest.TestCase):
    """Contract tests for evaluate_rag_report."""

    def _rag_report(self):
        """Helper: build a minimal RAG report."""
        return {
            "models": {
                "gpt-4o": {
                    "tests": {
                        "rag_test": {
                            "results": [
                                {
                                    "case_id": "c1",
                                    "scores": {},
                                    "latency": 1.0,
                                    "question": "Veri nedir?",
                                    "answer": "Veri bilgi birimidir.",
                                    "contexts": ["Veri, işlenmiş bilgidir."],
                                }
                            ]
                        }
                    }
                }
            }
        }

    def test_top_level_keys(self):
        """Result should have top-level keys: total_rag_cases, models, overall_fault_distribution."""
        report = self._rag_report()
        result = evaluate_rag_report(report)

        self.assertIn("total_rag_cases", result)
        self.assertIn("models", result)
        self.assertIn("overall_fault_distribution", result)

    def test_rag_cases_counted(self):
        """total_rag_cases should reflect the number of RAG cases found."""
        report = self._rag_report()
        result = evaluate_rag_report(report)

        self.assertGreaterEqual(result["total_rag_cases"], 1)

    def test_model_present_in_result(self):
        """Model keys from input should appear in result."""
        report = self._rag_report()
        result = evaluate_rag_report(report)

        self.assertIn("gpt-4o", result["models"])
        self.assertGreater(result["models"]["gpt-4o"]["rag_case_count"], 0)


class CosineNumpyContractTests(unittest.TestCase):
    """Regression: real embedding adapters return numpy arrays (multi-element),
    not plain Python lists — `if not v1` raises "truth value of an array with
    more than one element is ambiguous" for those, discovered wiring the RAG
    Eval embedding-scoring mode up to a real (non-mocked) embedding adapter."""

    def test_cosine_accepts_numpy_arrays(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(_cosine(v1, v2), 1.0, places=6)

    def test_cosine_with_orthogonal_numpy_arrays_is_zero(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        self.assertAlmostEqual(_cosine(v1, v2), 0.0, places=6)

    def test_cosine_returns_a_plain_python_float_not_a_numpy_scalar(self):
        # Regression: dot stayed a numpy.float32/64 even after norm1/norm2
        # were coerced to Python floats via math.sqrt, and that numpy scalar
        # ended up inside RagEvalResponse.details — Pydantic's JSON
        # serializer rejects numpy types outright
        # (PydanticSerializationError: Unable to serialize unknown type:
        # <class 'numpy.float32'>), so any real (embedding-mode) RAG Eval
        # call 500'd at the response-serialization stage, after the route
        # handler's own try/except had already returned successfully.
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _cosine(v1, v2)
        self.assertIsInstance(result, float)
        self.assertNotIsInstance(result, np.floating)

    def test_embed_fn_returning_numpy_rows_does_not_crash_compute_faithfulness(self):
        def fake_embed_fn(texts):
            # Shape (n, dim) numpy array, like UnifiedEmbeddingAdapter.encode()
            # returns — not a list of lists.
            return np.array([[1.0, 0.0] for _ in texts])

        result = compute_faithfulness("some answer", ["some context"], embed_fn=fake_embed_fn)
        self.assertAlmostEqual(result["faithfulness"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
