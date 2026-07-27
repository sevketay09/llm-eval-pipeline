"""Contract tests: contamination / test-set leakage probes."""
import unittest

from analysis.contamination import (
    ContaminationChecker,
    continuation_similarity,
    ngram_containment,
    split_for_probe,
    summarize,
)


class StubAdapter:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return {"content": self.reply}


LONG_QUESTION = (
    "Bir müşteri kredi kartı ekstresinde tanımadığı bir işlem gördüğünde "
    "itiraz sürecini başlatmak için hangi adımları sırasıyla izlemelidir"
)


class PureFunctionTests(unittest.TestCase):
    def test_containment_full_verbatim(self):
        ref = "iade süresi on dört gündür ve fatura ibrazı zorunludur"
        self.assertEqual(ngram_containment("cevap: " + ref, ref), 1.0)

    def test_containment_disjoint_is_zero(self):
        self.assertEqual(ngram_containment("kırmızı elma", "mavi deniz dalga köpük sahil"), 0.0)

    def test_containment_empty_inputs(self):
        self.assertEqual(ngram_containment("", "x y z"), 0.0)
        self.assertEqual(ngram_containment("x y z", ""), 0.0)

    def test_split_keeps_all_words(self):
        split = split_for_probe(LONG_QUESTION)
        self.assertIsNotNone(split)
        rejoined = split["prefix"] + " " + split["tail"]
        self.assertEqual(rejoined.split(), LONG_QUESTION.split())

    def test_split_rejects_short_text(self):
        self.assertIsNone(split_for_probe("EVET mi HAYIR mi"))

    def test_similarity_bounds(self):
        split = split_for_probe(LONG_QUESTION)
        self.assertAlmostEqual(continuation_similarity(split["tail"], split["tail"]), 1.0, places=3)
        self.assertEqual(continuation_similarity("tamamen alakasız metin", split["tail"]), 0.0)
        self.assertEqual(continuation_similarity("", split["tail"]), 0.0)


class SummaryTests(unittest.TestCase):
    def _cases(self, sims, threshold=0.6):
        return [{"similarity": s, "flagged": s >= threshold} for s in sims]

    def test_insufficient_data(self):
        summary = summarize(self._cases([0.9, 0.9]))
        self.assertEqual(summary["verdict"], "insufficient_data")
        self.assertIsNone(summary["contamination_rate"])

    def test_clean_verdict(self):
        summary = summarize(self._cases([0.0, 0.1, 0.05, 0.0, 0.2, 0.1]))
        self.assertEqual(summary["verdict"], "clean")
        self.assertEqual(summary["flagged"], 0)

    def test_suspected_verdict(self):
        summary = summarize(self._cases([0.9, 0.8, 0.1, 0.0, 0.7, 0.95]))
        self.assertEqual(summary["verdict"], "contamination_suspected")
        self.assertGreaterEqual(summary["contamination_rate"], 0.2)

    def test_inconclusive_between_bands(self):
        sims = [0.9] + [0.0] * 9  # 10% flagged
        summary = summarize(self._cases(sims))
        self.assertEqual(summary["verdict"], "inconclusive")


class CheckerTests(unittest.TestCase):
    def _case(self):
        return {"id": "c1", "question": LONG_QUESTION, "expected_answer": "Bankayı arayıp itiraz formu doldurmalıdır"}

    def test_memorizing_model_is_flagged(self):
        tail = split_for_probe(LONG_QUESTION)["tail"]
        checker = ContaminationChecker(StubAdapter(tail))
        result = checker.probe_case(self._case())
        self.assertTrue(result["flagged"])
        self.assertGreaterEqual(result["similarity"], 0.99)

    def test_clean_model_not_flagged(self):
        checker = ContaminationChecker(StubAdapter("Üzgünüm, bu metnin devamını bilmiyorum."))
        result = checker.probe_case(self._case())
        self.assertFalse(result["flagged"])

    def test_answer_leak_flags_even_without_tail_match(self):
        checker = ContaminationChecker(StubAdapter("Cevap: Bankayı arayıp itiraz formu doldurmalıdır"))
        result = checker.probe_case(self._case())
        self.assertTrue(result["flagged"])
        self.assertEqual(result["answer_leak"], 1.0)

    def test_short_case_skipped(self):
        checker = ContaminationChecker(StubAdapter("x"))
        self.assertIsNone(checker.probe_case({"id": "s", "question": "EVET mi?"}))

    def test_run_respects_sample_and_summarizes(self):
        tail = split_for_probe(LONG_QUESTION)["tail"]
        checker = ContaminationChecker(StubAdapter(tail))
        cases = [self._case() for _ in range(10)]
        report = checker.run(cases, max_samples=6)
        self.assertEqual(report["summary"]["n_cases"], 6)
        self.assertEqual(report["summary"]["verdict"], "contamination_suspected")

    def test_probe_prompt_never_contains_tail_or_answer(self):
        adapter = StubAdapter("devam")
        ContaminationChecker(adapter).probe_case(self._case())
        sent = " ".join(m["content"] for m in adapter.calls[0])
        tail = split_for_probe(LONG_QUESTION)["tail"]
        self.assertNotIn(tail, sent)
        self.assertNotIn("itiraz formu", sent)


if __name__ == "__main__":
    unittest.main()
