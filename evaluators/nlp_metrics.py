"""
NLP Metrics Evaluator — pure-Python BLEU / ROUGE / token-F1.

Zero external dependencies (no Azure SDK, no sacrebleu, no rouge-score).
Reference-based lexical metrics that cross-check the LLM-as-judge scores
whenever a QA case has an expected_output.
"""
import math
import re
from collections import Counter
from typing import Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def is_available() -> bool:
    """Pure Python — always available."""
    return True


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _bleu(candidate: List[str], reference: List[str], max_n: int = 4) -> float:
    """Sentence BLEU with add-1 smoothing on higher-order precisions."""
    if not candidate or not reference:
        return 0.0

    log_precision_sum = 0.0
    for n in range(1, max_n + 1):
        cand_ngrams = _ngrams(candidate, n)
        ref_ngrams = _ngrams(reference, n)
        total = sum(cand_ngrams.values())
        if total == 0:
            # Candidate shorter than n — smooth as 1/(2*len) like sacrebleu's floor
            log_precision_sum += math.log(1.0 / (2.0 * max(len(candidate), 1)))
            continue
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        if overlap == 0:
            if n == 1:
                return 0.0  # no unigram overlap at all
            precision = 1.0 / (total + 1)  # add-1 smoothing
        else:
            precision = overlap / total
        log_precision_sum += math.log(precision)

    geo_mean = math.exp(log_precision_sum / max_n)
    brevity = 1.0 if len(candidate) >= len(reference) else math.exp(1.0 - len(reference) / len(candidate))
    return brevity * geo_mean


def _f1_from_counts(overlap: float, cand_total: int, ref_total: int) -> float:
    if overlap == 0 or cand_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / cand_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def _rouge_n(candidate: List[str], reference: List[str], n: int) -> float:
    cand_ngrams = _ngrams(candidate, n)
    ref_ngrams = _ngrams(reference, n)
    overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
    return _f1_from_counts(overlap, sum(cand_ngrams.values()), sum(ref_ngrams.values()))


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if token_a == token_b else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _rouge_l(candidate: List[str], reference: List[str]) -> float:
    lcs = _lcs_length(candidate, reference)
    return _f1_from_counts(lcs, len(candidate), len(reference))


def _token_f1(candidate: List[str], reference: List[str]) -> float:
    """SQuAD-style bag-of-tokens F1."""
    overlap = sum((Counter(candidate) & Counter(reference)).values())
    return _f1_from_counts(overlap, len(candidate), len(reference))


class NLPMetricsEvaluator:
    """Reference-based lexical metrics: BLEU, ROUGE-1/2/L, token F1."""

    def evaluate(self, response: str, ground_truth: str) -> Dict[str, float]:
        """Compute all metrics for a single response/ground_truth pair.

        Returns:
            Dict with keys: bleu, rouge_1, rouge_2, rouge_l, f1 — all in [0, 1].
        """
        zeros = {"bleu": 0.0, "rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0, "f1": 0.0}
        if not response or not ground_truth:
            return zeros

        candidate = _tokenize(response)
        reference = _tokenize(ground_truth)
        if not candidate or not reference:
            return zeros

        try:
            return {
                "bleu": round(_bleu(candidate, reference), 4),
                "rouge_1": round(_rouge_n(candidate, reference, 1), 4),
                "rouge_2": round(_rouge_n(candidate, reference, 2), 4),
                "rouge_l": round(_rouge_l(candidate, reference), 4),
                "f1": round(_token_f1(candidate, reference), 4),
            }
        except Exception as e:
            logger.debug(f"NLP metric computation failed: {e}")
            return zeros
