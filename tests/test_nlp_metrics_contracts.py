"""Contract tests: pure-Python NLP metrics (BLEU / ROUGE / token F1)."""
import sys
import types
import unittest

if "datasets" not in sys.modules:  # heavy optional dep; pipeline_runner imports it transitively
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **k: None
    sys.modules["datasets"] = fake_datasets

from evaluators.nlp_metrics import NLPMetricsEvaluator, is_available

KEYS = ("bleu", "rouge_1", "rouge_2", "rouge_l", "f1")


class NlpMetricsContractTests(unittest.TestCase):
    def setUp(self):
        self.ev = NLPMetricsEvaluator()

    def test_always_available(self):
        self.assertTrue(is_available())

    def test_identical_texts_score_one(self):
        scores = self.ev.evaluate("İade süresi 14 gündür.", "İade süresi 14 gündür.")
        for key in KEYS:
            self.assertAlmostEqual(scores[key], 1.0, places=3, msg=key)

    def test_disjoint_texts_score_zero(self):
        scores = self.ev.evaluate("kırmızı elma tatlı", "mavi deniz dalga köpük")
        for key in KEYS:
            self.assertEqual(scores[key], 0.0, msg=key)

    def test_empty_inputs_return_zeros(self):
        for resp, ref in [("", "x"), ("x", ""), ("", ""), ("...", "!!!")]:
            scores = self.ev.evaluate(resp, ref)
            self.assertEqual(set(scores), set(KEYS))
            self.assertTrue(all(v == 0.0 for v in scores.values()), (resp, ref))

    def test_all_scores_in_unit_range(self):
        scores = self.ev.evaluate(
            "Sipariş kargoya verildikten sonra 3 iş günü içinde teslim edilir.",
            "Teslimat, kargoya verildikten sonra genellikle 3 iş günü sürer.",
        )
        for key in KEYS:
            self.assertGreaterEqual(scores[key], 0.0, msg=key)
            self.assertLessEqual(scores[key], 1.0, msg=key)

    def test_partial_overlap_is_between_zero_and_one(self):
        scores = self.ev.evaluate("iade süresi 14 gün", "iade süresi 30 gün")
        self.assertGreater(scores["rouge_1"], 0.0)
        self.assertLess(scores["rouge_1"], 1.0)
        self.assertGreater(scores["f1"], 0.0)
        self.assertLess(scores["f1"], 1.0)

    def test_deterministic(self):
        a = self.ev.evaluate("hızlı kahverengi tilki", "tembel köpeğin üstünden atlar hızlı tilki")
        b = self.ev.evaluate("hızlı kahverengi tilki", "tembel köpeğin üstünden atlar hızlı tilki")
        self.assertEqual(a, b)

    def test_case_insensitive_tokenization(self):
        upper = self.ev.evaluate("MERHABA DÜNYA", "merhaba dünya")
        self.assertAlmostEqual(upper["f1"], 1.0, places=3)

    def test_rouge_l_respects_word_order(self):
        in_order = self.ev.evaluate("bir iki üç dört", "bir iki üç dört")
        shuffled = self.ev.evaluate("dört üç iki bir", "bir iki üç dört")
        self.assertGreater(in_order["rouge_l"], shuffled["rouge_l"])
        # bag-of-words F1 is order-blind — sanity check the contrast
        self.assertAlmostEqual(shuffled["f1"], 1.0, places=3)

    def test_bleu_brevity_penalizes_short_candidate(self):
        short = self.ev.evaluate("iade", "iade süresi on dört gündür ve fatura gerekir")
        self.assertLess(short["bleu"], 0.5)


class PipelineWiringTests(unittest.TestCase):
    def test_qa_metric_results_include_nlp_group(self):
        from pipeline_runner import _build_qa_metric_results

        serialized = _build_qa_metric_results(
            accuracy_judge={"score": 1.0, "label": "TAM_DOGRU"},
            hallucination_score={"score": 1.0},
            geval_scores={},
            quality_scores={},
            nlp_scores={"bleu": 0.42, "rouge_l": 0.5},
        )
        nlp = [m for m in serialized if m.get("provider") == "nlp_metrics"]
        self.assertEqual({m["name"] for m in nlp}, {"bleu", "rouge_l"})
        self.assertTrue(all(m.get("group") == "nlp" for m in nlp))

    def test_qa_metric_results_omit_nlp_when_empty(self):
        from pipeline_runner import _build_qa_metric_results

        serialized = _build_qa_metric_results(
            accuracy_judge={"score": 1.0, "label": "TAM_DOGRU"},
            hallucination_score={"score": 1.0},
            geval_scores={},
            quality_scores={},
            nlp_scores={},
        )
        self.assertFalse([m for m in serialized if m.get("provider") == "nlp_metrics"])


if __name__ == "__main__":
    unittest.main()
