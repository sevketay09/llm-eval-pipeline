"""
Contamination / test-set leakage check — pure Python, zero new dependencies.

Two signals per test case:
1. Continuation probe (guided completion): show the model the first part of the
   test question and ask it to continue verbatim. High similarity between the
   generated continuation and the true tail suggests the case was memorized
   during training (Golchin & Surdeanu-style probe).
2. Verbatim n-gram containment of the expected answer inside the continuation —
   the model completing the *answer* it was never shown is a strong leak signal.
"""
from typing import Any, Dict, List, Optional

from evaluators.nlp_metrics import _ngrams, _rouge_l, _tokenize
from utils.logger import get_logger

logger = get_logger(__name__)

MIN_CASES = 5
MIN_PREFIX_TOKENS = 4
DEFAULT_THRESHOLD = 0.6  # rouge_l between continuation and true tail
PROBE_SYSTEM_PROMPT = (
    "Sana bir metnin başlangıcı verilecek. Bu metni daha önce gördüysen kalanını "
    "birebir aynı kelimelerle tamamla. Sadece devam metnini yaz, açıklama ekleme."
)


def ngram_containment(candidate: str, reference: str, n: int = 8) -> float:
    """Fraction of reference n-grams that appear verbatim in candidate."""
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    n = min(n, len(ref_tokens))
    ref_ngrams = _ngrams(ref_tokens, n)
    cand_ngrams = _ngrams(cand_tokens, n)
    total = sum(ref_ngrams.values())
    if total == 0:
        return 0.0
    overlap = sum(min(count, cand_ngrams[gram]) for gram, count in ref_ngrams.items())
    return round(overlap / total, 4)


def split_for_probe(text: str, ratio: float = 0.6) -> Optional[Dict[str, str]]:
    """Split text into a prefix shown to the model and the tail it must not know.

    Returns None when the text is too short for a meaningful probe.
    """
    words = text.split()
    cut = int(len(words) * ratio)
    if cut < MIN_PREFIX_TOKENS or cut >= len(words):
        return None
    return {"prefix": " ".join(words[:cut]), "tail": " ".join(words[cut:])}


def continuation_similarity(continuation: str, true_tail: str) -> float:
    """Order-aware similarity (ROUGE-L F1) between probe output and true tail."""
    cand = _tokenize(continuation)
    ref = _tokenize(true_tail)
    if not cand or not ref:
        return 0.0
    return round(_rouge_l(cand, ref), 4)


def summarize(case_results: List[Dict[str, Any]], threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """Aggregate per-case probe results into a leak verdict."""
    n = len(case_results)
    if n < MIN_CASES:
        return {
            "n_cases": n,
            "flagged": 0,
            "contamination_rate": None,
            "mean_similarity": None,
            "threshold": threshold,
            "verdict": "insufficient_data",
        }
    flagged = sum(1 for r in case_results if r["flagged"])
    rate = flagged / n
    mean_sim = sum(r["similarity"] for r in case_results) / n
    if rate >= 0.2:
        verdict = "contamination_suspected"
    elif rate < 0.05:
        verdict = "clean"
    else:
        verdict = "inconclusive"
    return {
        "n_cases": n,
        "flagged": flagged,
        "contamination_rate": round(rate, 4),
        "mean_similarity": round(mean_sim, 4),
        "threshold": threshold,
        "verdict": verdict,
    }


def _case_text(case: Dict[str, Any]) -> Optional[str]:
    for key in ("question", "input", "input_text", "prompt"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _case_answer(case: Dict[str, Any]) -> str:
    for key in ("expected_answer", "expected_output", "answer", "ground_truth"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


class ContaminationChecker:
    """Run continuation probes against a model adapter."""

    def __init__(self, adapter, threshold: float = DEFAULT_THRESHOLD, split_ratio: float = 0.6):
        self.adapter = adapter
        self.threshold = threshold
        self.split_ratio = split_ratio

    def probe_case(self, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = _case_text(case)
        if not text:
            return None
        split = split_for_probe(text, self.split_ratio)
        if split is None:
            return None

        messages = [
            {"role": "system", "content": PROBE_SYSTEM_PROMPT},
            {"role": "user", "content": split["prefix"]},
        ]
        response = self.adapter.generate(messages)
        continuation = response.get("content") or ""

        similarity = continuation_similarity(continuation, split["tail"])
        answer = _case_answer(case)
        answer_leak = ngram_containment(continuation, answer) if answer else 0.0
        flagged = similarity >= self.threshold or answer_leak >= self.threshold
        return {
            "id": case.get("id", "unknown"),
            "prefix": split["prefix"],
            "true_tail": split["tail"],
            "continuation": continuation,
            "similarity": similarity,
            "answer_leak": answer_leak,
            "flagged": flagged,
        }

    def run(self, cases: List[Dict[str, Any]], max_samples: Optional[int] = None) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for case in cases:
            if max_samples is not None and len(results) >= max_samples:
                break
            try:
                probed = self.probe_case(case)
            except Exception as exc:
                logger.warning(f"Contamination probe failed for case {case.get('id')}: {exc}")
                continue
            if probed is not None:
                results.append(probed)
        return {
            "summary": summarize(results, self.threshold),
            "cases": results,
        }
